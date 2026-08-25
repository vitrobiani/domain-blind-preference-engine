"""
Base scorer interface for all signals.

Every signal implements the Scorer interface. The base class owns
normalization and cold-start behavior (returning empty frames).
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from pyspark.sql import functions as F

from preference_engine.schema import ITEM_ID

if TYPE_CHECKING:
    from pyspark.sql import DataFrame
    from preference_engine.streaming.feature_store import FeatureStore

# Column name for raw scores before normalization
RAW_SCORE = "raw_score"
# Column name for normalized scores
SCORE = "score"


class Scorer(ABC):
    """
    Abstract base class for all scoring signals.

    A scorer computes relevance scores for items given a user.
    Each scorer is backed by a Spark MLlib algorithm or aggregation.

    Subclasses must set:
        name: Unique identifier for this signal.

    Subclasses must implement:
        fit(): Train/prepare using the feature store.
        raw_score(): Return raw scores for candidate items.
    """

    name: str

    @abstractmethod
    def fit(self, ctx: "FeatureStore", params: dict) -> None:
        """
        Train or prepare the scorer using the feature store.

        Called once before scoring. Should be idempotent.

        Args:
            ctx: Feature store with preprocessed data.
            params: Signal-specific parameters from heuristics.
        """
        raise NotImplementedError

    @abstractmethod
    def raw_score(self, user_id: str, candidates: "DataFrame") -> "DataFrame":
        """
        Compute raw scores for candidate items.

        Args:
            user_id: The user to score for.
            candidates: DataFrame with ITEM_ID column of items to score.

        Returns:
            DataFrame[item_id, raw_score] for the candidate items.
            Return an EMPTY DataFrame if this signal cannot score this user
            (e.g., ALS for a cold user) - the combiner handles the gap.
        """
        raise NotImplementedError

    def score(self, user_id: str, candidates: "DataFrame") -> "DataFrame":
        """
        Compute normalized scores for candidate items.

        Calls raw_score() and normalizes to [0, 1] range using min-max scaling
        over the candidate set.

        Args:
            user_id: The user to score for.
            candidates: DataFrame with ITEM_ID column of items to score.

        Returns:
            DataFrame[item_id, score] with scores in [0, 1].
            Returns empty DataFrame if raw_score returns empty.
        """
        raw = self.raw_score(user_id, candidates)

        if raw.isEmpty():
            return raw.withColumnRenamed(RAW_SCORE, SCORE)

        # Min-max normalization to [0, 1]
        stats = raw.agg(
            F.min(RAW_SCORE).alias("min_score"),
            F.max(RAW_SCORE).alias("max_score"),
        ).collect()[0]

        min_val = stats["min_score"]
        max_val = stats["max_score"]

        if min_val == max_val:
            # All same score - normalize to 0.5
            return raw.withColumn(SCORE, F.lit(0.5)).drop(RAW_SCORE)

        return raw.withColumn(
            SCORE,
            (F.col(RAW_SCORE) - F.lit(min_val)) / (F.lit(max_val) - F.lit(min_val)),
        ).drop(RAW_SCORE)

    def save(self, path: "Path") -> None:
        """
        Persist this fitted scorer to `path` (a directory).

        Base implementation writes only the manifest. Subclasses with
        trained state override this, call super().save(path) first,
        then write their artifacts alongside.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        manifest = {"signal": self.name, **self._manifest_extras()}
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def _manifest_extras(self) -> dict:
        """Constructor kwargs needed to rebuild the instance. Override if any."""
        return {}

    @classmethod
    def load(cls, path: "Path", ctx: "FeatureStore") -> "Scorer":
        """
        Rebuild a fitted scorer from disk.

        Subclasses override to restore model artifacts. `ctx` gives
        access to the SparkSession and any feature-store tables the
        scorer references.
        """
        raise NotImplementedError(f"{cls.__name__} does not implement load()")
