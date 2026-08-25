"""
ALS collaborative filtering signal.

Uses Spark MLlib's Alternating Least Squares for matrix factorization.
Respects interaction_type: explicit mode for ratings, implicit mode for counts.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING

from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS, ALSModel

from preference_engine.signals.base import Scorer, RAW_SCORE
from preference_engine.schema import ITEM_ID, USER_ID, VALUE, USER_ID_INT, ITEM_ID_INT
from preference_engine.ids import IDMapper

if TYPE_CHECKING:
    from pyspark.sql import DataFrame
    from preference_engine.streaming.feature_store import FeatureStore


class ALSScorer(Scorer):
    """
    Collaborative filtering scorer using ALS matrix factorization.

    For explicit feedback: predicts ratings.
    For implicit feedback: uses confidence values with implicitPrefs=True.

    Requires integer IDs (handled by IDMapper).
    """

    name = "als_cf"

    def __init__(self, implicit: bool = False) -> None:
        """
        Initialize the ALS scorer.

        Args:
            implicit: Whether to use implicit feedback mode.
        """
        self._implicit = implicit
        self._model = None
        self._id_mapper: IDMapper | None = None
        self._spark = None

    def fit(self, ctx: "FeatureStore", params: dict) -> None:
        """
        Train the ALS model on interaction data.

        Args:
            ctx: Feature store with interactions.
            params: ALS parameters (rank, reg, max_iter).
        """
        rank = params.get("rank", 32)
        reg = params.get("reg", 0.1)
        max_iter = params.get("max_iter", 10)

        # Load interactions
        interactions = ctx.interactions()
        self._spark = interactions.sparkSession

        # Build ID mapper (ALS needs integer IDs)
        self._id_mapper = IDMapper().fit(interactions)

        # Transform to integer IDs
        int_interactions = self._id_mapper.transform(interactions)

        # Handle null values - replace with 1.0 for implicit
        if self._implicit:
            int_interactions = int_interactions.withColumn(
                VALUE,
                F.coalesce(F.col(VALUE), F.lit(1.0))
            )
        else:
            # For explicit, filter out null values
            int_interactions = int_interactions.filter(F.col(VALUE).isNotNull())

        # Configure ALS
        als = ALS(
            rank=rank,
            regParam=reg,
            maxIter=max_iter,
            userCol=USER_ID_INT,
            itemCol=ITEM_ID_INT,
            ratingCol=VALUE,
            implicitPrefs=self._implicit,
            coldStartStrategy="drop",
            seed=42,
        )

        # Train model
        self._model = als.fit(int_interactions)

    def raw_score(self, user_id: str, candidates: "DataFrame") -> "DataFrame":
        """
        Score candidates using learned user/item factors.

        Args:
            user_id: User to generate predictions for.
            candidates: Items to score.

        Returns:
            DataFrame[item_id, raw_score] with predicted ratings/preferences.
            Returns empty DataFrame for cold users (not in training data).
        """
        if self._model is None or self._id_mapper is None:
            raise RuntimeError("ALSScorer must be fit before scoring")

        # Convert user_id to integer
        user_int = self._id_mapper.user_id_to_int(user_id)
        if user_int is None:
            # Cold user - return empty
            return candidates.select(ITEM_ID).withColumn(RAW_SCORE, F.lit(None)).limit(0)

        # Add integer IDs to candidates
        candidates_with_int = self._id_mapper.transform(
            candidates.select(ITEM_ID)
        )

        # Create user-item pairs for prediction
        user_items = candidates_with_int.withColumn(
            USER_ID_INT,
            F.lit(float(user_int))
        )

        # Get predictions
        predictions = self._model.transform(user_items)

        # Convert back to string IDs and rename prediction column
        result = (
            predictions
            .select(ITEM_ID, F.col("prediction").alias(RAW_SCORE))
            # Handle NaN predictions (cold items)
            .withColumn(
                RAW_SCORE,
                F.when(F.isnan(F.col(RAW_SCORE)), F.lit(0.0)).otherwise(F.col(RAW_SCORE))
            )
        )

        return result

    def _manifest_extras(self) -> dict:
        return {"implicit": self._implicit}

    def save(self, path: Path) -> None:
        if self._model is None or self._id_mapper is None:
            raise RuntimeError("ALSScorer must be fit before save")

        super().save(path)
        path = Path(path)
        self._model.save(str(path / "als_model"))
        self._id_mapper.save(path / "id_mapper")

    @classmethod
    def load(cls, path: Path, ctx: "FeatureStore") -> "ALSScorer":
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text())
        instance = cls(implicit=manifest.get("implicit", False))
        instance._model = ALSModel.load(str(path / "als_model"))
        instance._id_mapper = IDMapper.load(path / "id_mapper", ctx.spark)
        instance._spark = ctx.spark
        return instance
