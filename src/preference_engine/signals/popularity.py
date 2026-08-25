"""
Popularity signal - baseline scorer using global interaction counts.

The simplest signal: items are ranked by how many interactions they have.
Always works (no cold-start problem), serves as a fallback.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from pyspark.sql import functions as F

from preference_engine.signals.base import Scorer, RAW_SCORE
from preference_engine.schema import ITEM_ID

if TYPE_CHECKING:
    from pyspark.sql import DataFrame
    from preference_engine.streaming.feature_store import FeatureStore


class PopularityScorer(Scorer):
    """
    Scorer based on global item popularity.

    Ranks items by total interaction count. Provides a baseline that
    works for all users (no personalization, but no cold-start issues).
    """

    name = "popularity"

    def __init__(self) -> None:
        self._popularity_df: "DataFrame | None" = None

    def fit(self, ctx: "FeatureStore", params: dict) -> None:
        """
        Prepare popularity scores from the feature store.

        Args:
            ctx: Feature store with popularity aggregates.
            params: Unused for this signal.
        """
        # Load pre-computed popularity from feature store
        self._popularity_df = ctx.popularity()

    def raw_score(self, user_id: str, candidates: "DataFrame") -> "DataFrame":
        """
        Score candidates by their popularity.

        Args:
            user_id: Unused - popularity is not personalized.
            candidates: Items to score.

        Returns:
            DataFrame[item_id, raw_score] with popularity counts.
        """
        if self._popularity_df is None:
            raise RuntimeError("PopularityScorer must be fit before scoring")

        # Join candidates with popularity, defaulting to 0 for unknown items
        return (
            candidates
            .select(ITEM_ID)
            .join(self._popularity_df, on=ITEM_ID, how="left")
            .withColumn(
                RAW_SCORE,
                F.coalesce(F.col("count").cast("double"), F.lit(0.0))
            )
            .select(ITEM_ID, RAW_SCORE)
        )

    @classmethod
    def load(cls, path: Path, ctx: "FeatureStore") -> "PopularityScorer":
        instance = cls()
        instance._popularity_df = ctx.popularity()
        return instance
