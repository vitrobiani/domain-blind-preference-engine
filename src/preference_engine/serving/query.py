"""
Query interface for generating recommendations.

The main entry point for getting ranked recommendations for a user.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from pyspark.sql import functions as F

from preference_engine.schema import ITEM_ID, USER_ID
from preference_engine.signals import get_scorer, save_scorers, load_scorers
from preference_engine.combiner.combiner import combine

if TYPE_CHECKING:
    from pyspark.sql import DataFrame
    from preference_engine.combiner.heuristics import HeuristicsConfig
    from preference_engine.streaming.feature_store import FeatureStore
    from preference_engine.signals.base import Scorer


DEFAULT_MODELS_PATH = Path(".data/models")


class RecommendationEngine:
    """
    Engine for generating recommendations.

    Manages fitted scorers and provides recommendation queries.
    """

    def __init__(
        self,
        heuristics: "HeuristicsConfig",
        feature_store: "FeatureStore",
        models_path: Path = DEFAULT_MODELS_PATH,
    ) -> None:
        """
        Initialize the recommendation engine.

        Args:
            heuristics: Loaded heuristics configuration.
            feature_store: Initialized feature store.
            models_path: Where fitted scorers are persisted.
        """
        self.heuristics = heuristics
        self.feature_store = feature_store
        self.models_path = Path(models_path)
        self._scorers: dict[str, "Scorer"] = {}
        self._fitted = False

    def fit(self, force: bool = False) -> "RecommendationEngine":
        """
        Load scorers from disk if present, train anything missing, and
        persist newly trained scorers.

        Args:
            force: If True, retrain every active signal from scratch
                   regardless of what exists on disk.

        Returns:
            Self for method chaining.
        """
        active_signals = set(self.heuristics.get_active_signals())

        if force:
            # Retrain from scratch - drop any in-memory scorers so the loop
            # below treats every active signal as "missing" and refits it.
            self._scorers = {}
        else:
            # Load whatever's already on disk
            loaded = load_scorers(self.models_path, self.feature_store)
            self._scorers = {n: s for n, s in loaded.items() if n in active_signals}
            if self._scorers:
                print(f"  Loaded from disk: {sorted(self._scorers.keys())}")

        # Train anything missing. Per-signal failures are logged and skipped
        # so one broken signal (e.g. ALS on a tiny interaction set) doesn't
        # kill the whole retrain - surviving signals still serve.
        newly_trained: dict[str, "Scorer"] = {}
        skipped: dict[str, str] = {}
        for name in sorted(active_signals - self._scorers.keys()):
            print(f"  Training {name}...")

            signal_config = self.heuristics.signals.get(name)
            params = signal_config.params if signal_config else {}

            kwargs = {}
            if name == "als_cf":
                kwargs["implicit"] = False  # Default to explicit

            try:
                scorer = get_scorer(name, **kwargs)
                scorer.fit(self.feature_store, params)
            except Exception as exc:  # noqa: BLE001
                skipped[name] = f"{type(exc).__name__}: {exc}"
                print(f"    ! {name} skipped ({skipped[name]})")
                continue

            self._scorers[name] = scorer
            newly_trained[name] = scorer

        # Persist any newly trained scorers so the next run can skip training
        if newly_trained:
            print(f"  Saving to {self.models_path}: {sorted(newly_trained.keys())}")
            save_scorers(newly_trained, self.models_path)
        if skipped:
            print(f"  Skipped signals: {sorted(skipped.keys())}")

        self._fitted = True
        return self

    def recommend(self, user_id: str) -> "DataFrame":
        """
        Generate ranked recommendations for a user.

        Args:
            user_id: User to generate recommendations for.

        Returns:
            DataFrame[item_id, final_score, breakdown] with top-k recommendations.
        """
        if not self._fitted:
            raise RuntimeError("RecommendationEngine must be fit before recommending")

        # Get all candidate items
        candidates = get_candidates(self.feature_store)

        # Get user's seen items for filtering
        seen_items = get_user_seen_items(self.feature_store, user_id)

        # Score with each active signal
        scores: dict[str, "DataFrame"] = {}
        for name, scorer in self._scorers.items():
            score_df = scorer.score(user_id, candidates)
            if not score_df.isEmpty():
                scores[name] = score_df

        # Combine scores
        weights = self.heuristics.get_weights()

        return combine(
            scores=scores,
            weights=weights,
            filters=self.heuristics.filters,
            top_k=self.heuristics.top_k,
            seen_items=seen_items if self.heuristics.filters.exclude_seen else None,
        )


def recommend(
    user_id: str,
    heuristics: "HeuristicsConfig",
    feature_store: "FeatureStore",
    scorers: dict[str, "Scorer"] | None = None,
) -> "DataFrame":
    """
    Generate ranked recommendations for a user.

    Orchestrates the full recommendation pipeline:
    1. Load candidate items from feature store
    2. Score with each active signal
    3. Combine scores with heuristic weights
    4. Apply filters
    5. Return top-k with breakdown

    Args:
        user_id: User to generate recommendations for.
        heuristics: Loaded heuristics configuration.
        feature_store: Initialized feature store.
        scorers: Pre-fitted scorers (if None, will fit on-the-fly).

    Returns:
        DataFrame[item_id, final_score, breakdown] with top-k recommendations.
    """
    # Get all candidate items
    candidates = get_candidates(feature_store)

    # Get user's seen items for filtering
    seen_items = get_user_seen_items(feature_store, user_id)

    # Get or fit scorers
    if scorers is None:
        scorers = {}
        for name in heuristics.get_active_signals():
            signal_config = heuristics.signals.get(name)
            params = signal_config.params if signal_config else {}

            kwargs = {}
            if name == "als_cf":
                kwargs["implicit"] = False

            scorer = get_scorer(name, **kwargs)
            scorer.fit(feature_store, params)
            scorers[name] = scorer

    # Score with each signal
    scores: dict[str, "DataFrame"] = {}
    for name, scorer in scorers.items():
        score_df = scorer.score(user_id, candidates)
        if not score_df.isEmpty():
            scores[name] = score_df

    # Combine scores
    weights = heuristics.get_weights()

    return combine(
        scores=scores,
        weights=weights,
        filters=heuristics.filters,
        top_k=heuristics.top_k,
        seen_items=seen_items if heuristics.filters.exclude_seen else None,
    )


def get_candidates(
    feature_store: "FeatureStore",
    exclude: list[str] | None = None,
) -> "DataFrame":
    """
    Get all candidate items for scoring.

    Args:
        feature_store: Feature store with item data.
        exclude: Optional list of item IDs to exclude.

    Returns:
        DataFrame with ITEM_ID column of candidate items.
    """
    items = feature_store.item_features().select(ITEM_ID).distinct()

    if exclude:
        items = items.filter(~F.col(ITEM_ID).isin(exclude))

    return items


def get_user_seen_items(
    feature_store: "FeatureStore",
    user_id: str,
) -> "DataFrame":
    """
    Get items the user has already interacted with.

    Args:
        feature_store: Feature store with interactions.
        user_id: User to get seen items for.

    Returns:
        DataFrame with ITEM_ID column of seen items.
    """
    return (
        feature_store.interactions()
        .filter(F.col(USER_ID) == user_id)
        .select(ITEM_ID)
        .distinct()
    )
