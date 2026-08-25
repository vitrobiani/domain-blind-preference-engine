"""
Content-based signal using item/user feature similarity.

Computes cosine similarity between user profile and item feature vectors.
User profile is the aggregate of item vectors they've interacted with.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer, Normalizer
from pyspark.ml.linalg import Vectors, VectorUDT
import numpy as np

from preference_engine.signals.base import Scorer, RAW_SCORE
from preference_engine.schema import ITEM_ID, USER_ID, VALUE

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession
    from preference_engine.streaming.feature_store import FeatureStore


class ContentScorer(Scorer):
    """
    Content-based scorer using feature vector similarity.

    Builds item feature vectors from categorical/numeric features.
    User profile = weighted centroid of items they've interacted with.
    Score = cosine similarity between user profile and item vector.
    """

    name = "content"

    def __init__(self) -> None:
        self._item_vectors: "DataFrame | None" = None
        self._user_profiles: "DataFrame | None" = None
        self._spark: "SparkSession | None" = None

    def fit(self, ctx: "FeatureStore", params: dict) -> None:
        """
        Build feature vectors from the feature store.

        Args:
            ctx: Feature store with item features.
            params: May include feature_weights mapping.
        """
        feature_weights = params.get("feature_weights", {})

        # Get item features
        item_features = ctx.item_features()
        self._spark = item_features.sparkSession

        # Get feature columns (exclude item_id)
        feature_cols = [c for c in item_features.columns if c != ITEM_ID]

        # Index categorical features and prepare numeric columns
        indexed_df = item_features
        numeric_cols = []

        for col in feature_cols:
            dtype = str(item_features.schema[col].dataType)
            if "String" in dtype:
                # Index categorical column
                indexer = StringIndexer(
                    inputCol=col,
                    outputCol=f"{col}_idx",
                    handleInvalid="keep"
                )
                indexed_df = indexer.fit(indexed_df).transform(indexed_df)
                numeric_cols.append(f"{col}_idx")
            else:
                numeric_cols.append(col)

        # Assemble features into vector
        assembler = VectorAssembler(
            inputCols=numeric_cols,
            outputCol="features_raw",
            handleInvalid="skip"
        )
        vectorized = assembler.transform(indexed_df)

        # Normalize vectors for cosine similarity
        normalizer = Normalizer(inputCol="features_raw", outputCol="features", p=2.0)
        self._item_vectors = (
            normalizer
            .transform(vectorized)
            .select(ITEM_ID, "features")
        )

        # Build user profiles from interaction history
        interactions = ctx.interactions()

        # Join interactions with item vectors
        user_items = (
            interactions
            .join(self._item_vectors, on=ITEM_ID, how="inner")
        )

        # Compute weighted average of item vectors per user
        # Weight by interaction value (rating)
        @F.udf(VectorUDT())
        def weighted_sum_vectors(features_list, weights_list):
            if not features_list or not weights_list:
                return None
            vectors = [np.array(f.toArray()) for f in features_list]
            weights = np.array(weights_list)
            # Handle None weights
            weights = np.nan_to_num(weights, nan=1.0)
            if weights.sum() == 0:
                weights = np.ones_like(weights)
            weighted = np.average(vectors, axis=0, weights=weights)
            # Normalize
            norm = np.linalg.norm(weighted)
            if norm > 0:
                weighted = weighted / norm
            return Vectors.dense(weighted.tolist())

        self._user_profiles = (
            user_items
            .groupBy(USER_ID)
            .agg(
                F.collect_list("features").alias("features_list"),
                F.collect_list(F.coalesce(F.col(VALUE), F.lit(1.0))).alias("weights_list")
            )
            .withColumn("profile", weighted_sum_vectors("features_list", "weights_list"))
            .select(USER_ID, "profile")
            .filter(F.col("profile").isNotNull())
        )

    def raw_score(self, user_id: str, candidates: "DataFrame") -> "DataFrame":
        """
        Score candidates by similarity to user's content profile.

        Args:
            user_id: User whose profile to match against.
            candidates: Items to score.

        Returns:
            DataFrame[item_id, raw_score] with cosine similarities.
            Returns empty DataFrame if user has no interaction history.
        """
        if self._item_vectors is None or self._user_profiles is None:
            raise RuntimeError("ContentScorer must be fit before scoring")

        # Get user profile
        user_profile = (
            self._user_profiles
            .filter(F.col(USER_ID) == user_id)
            .select("profile")
            .collect()
        )

        if not user_profile or user_profile[0]["profile"] is None:
            # User has no profile - return empty
            return candidates.select(ITEM_ID).withColumn(RAW_SCORE, F.lit(None)).limit(0)

        profile_vec = user_profile[0]["profile"]

        # Compute cosine similarity (dot product of normalized vectors)
        @F.udf("double")
        def cosine_sim(item_vec):
            if item_vec is None:
                return 0.0
            return float(np.dot(profile_vec.toArray(), item_vec.toArray()))

        # Join candidates with item vectors and compute similarity
        return (
            candidates
            .select(ITEM_ID)
            .join(self._item_vectors, on=ITEM_ID, how="left")
            .withColumn(RAW_SCORE, cosine_sim(F.col("features")))
            .withColumn(RAW_SCORE, F.coalesce(F.col(RAW_SCORE), F.lit(0.0)))
            .select(ITEM_ID, RAW_SCORE)
        )

    def save(self, path: Path) -> None:
        if self._item_vectors is None or self._user_profiles is None:
            raise RuntimeError("ContentScorer must be fit before save")

        super().save(path)
        path = Path(path)
        self._item_vectors.write.mode("overwrite").parquet(
            str(path / "item_vectors"))
        self._user_profiles.write.mode("overwrite").parquet(
            str(path / "user_profiles"))

    @classmethod
    def load(cls, path: Path, ctx: "FeatureStore") -> "ContentScorer":
        path = Path(path)
        instance = cls()
        instance._item_vectors = ctx.spark.read.parquet(
            str(path / "item_vectors"))
        instance._user_profiles = ctx.spark.read.parquet(
            str(path / "user_profiles"))
        instance._spark = ctx.spark
        return instance
