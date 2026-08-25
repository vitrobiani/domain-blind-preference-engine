"""
Trends signal - recency-weighted popularity.

Like popularity, but with exponential time decay. Recent interactions
count more than older ones, capturing trending items.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from pyspark.sql import functions as F

from preference_engine.signals.base import Scorer, RAW_SCORE
from preference_engine.schema import ITEM_ID

if TYPE_CHECKING:
    from pyspark.sql import DataFrame
    from preference_engine.streaming.feature_store import FeatureStore


class TrendsScorer(Scorer):
    """
    Scorer based on recency-weighted item popularity.

    Applies exponential decay to interaction counts based on timestamp.
    halflife_days parameter controls how quickly old interactions fade.
    """

    name = "trends"

    def __init__(self) -> None:
        self._recency_df: "DataFrame | None" = None

    def fit(self, ctx: "FeatureStore", params: dict) -> None:
        """
        Compute recency-weighted popularity from interactions.

        Args:
            ctx: Feature store with timestamped interactions.
            params: Trends parameters (halflife_days - unused here as
                    recency is pre-computed in feature store).
        """
        # Load pre-computed recency scores from feature store
        self._recency_df = ctx.recency()

    def raw_score(self, user_id: str, candidates: "DataFrame") -> "DataFrame":
        """
        Score candidates by recency-weighted popularity.

        Args:
            user_id: Unused - trends are global.
            candidates: Items to score.

        Returns:
            DataFrame[item_id, raw_score] with recency-weighted counts.
        """
        if self._recency_df is None:
            raise RuntimeError("TrendsScorer must be fit before scoring")

        # Join candidates with recency scores, defaulting to 0 for unknown items
        return (
            candidates
            .select(ITEM_ID)
            .join(self._recency_df, on=ITEM_ID, how="left")
            .withColumn(
                RAW_SCORE,
                F.coalesce(F.col("recency_score"), F.lit(0.0))
            )
            .select(ITEM_ID, RAW_SCORE)
        )

    @classmethod
    def load(cls, path: Path, ctx: "FeatureStore") -> "TrendsScorer":
        instance = cls()
        instance._recency_df = ctx.recency()
        return instance
