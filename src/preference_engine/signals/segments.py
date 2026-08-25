"""
User segments signal using KMeans clustering.

Clusters users by their features, then scores items by cluster preferences.
Useful when users have demographic features but sparse interaction history.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING

from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.clustering import KMeans, KMeansModel

from preference_engine.signals.base import Scorer, RAW_SCORE
from preference_engine.schema import ITEM_ID, USER_ID, VALUE

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession
    from preference_engine.streaming.feature_store import FeatureStore


class SegmentsScorer(Scorer):
    """
    Scorer using user segment (cluster) preferences.

    Clusters users with KMeans on user features.
    Computes each cluster's aggregate item preferences.
    Scores items based on the user's cluster preferences.
    """

    name = "segments"

    def __init__(self, k: int = 8) -> None:
        self._k = k
        self._kmeans_model = None
        self._user_clusters: "DataFrame | None" = None
        self._cluster_preferences: "DataFrame | None" = None
        self._spark: "SparkSession | None" = None

    def fit(self, ctx: "FeatureStore", params: dict) -> None:
        """
        Build user clusters and cluster-item preferences.

        Args:
            ctx: Feature store with user features and interactions.
            params: KMeans parameters (k = number of clusters).
        """
        k = params.get("k", self._k)
        self._k = k

        # Get user features
        user_features = ctx.user_features()
        self._spark = user_features.sparkSession

        # Get numeric feature columns (exclude user_id)
        feature_cols = [c for c in user_features.columns if c != USER_ID]

        # Handle categorical features by indexing them
        indexed_df = user_features
        numeric_cols = []

        for col in feature_cols:
            dtype = str(user_features.schema[col].dataType)
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
            outputCol="features",
            handleInvalid="skip"
        )
        vectorized = assembler.transform(indexed_df)

        # Fit KMeans
        kmeans = KMeans(k=k, seed=42, featuresCol="features", predictionCol="cluster")
        self._kmeans_model = kmeans.fit(vectorized)

        # Get user cluster assignments
        self._user_clusters = (
            self._kmeans_model
            .transform(vectorized)
            .select(USER_ID, "cluster")
        )

        # Cluster preference = avg(value) × support. avg(value) alone works
        # for explicit ratings (movies) but collapses to 1.0 for binary
        # interactions (elections), losing all popularity signal within the
        # cluster. Multiplying by count keeps the "rating quality" axis for
        # explicit data while restoring "how many people picked this" for
        # implicit/binary data.
        interactions = ctx.interactions()

        self._cluster_preferences = (
            interactions
            .join(self._user_clusters, on=USER_ID, how="inner")
            .groupBy("cluster", ITEM_ID)
            .agg((F.avg(VALUE) * F.count("*")).alias("cluster_score"))
        )

    def raw_score(self, user_id: str, candidates: "DataFrame") -> "DataFrame":
        """
        Score candidates by user's cluster preferences.

        Args:
            user_id: User to find cluster for.
            candidates: Items to score.

        Returns:
            DataFrame[item_id, raw_score] with cluster preference scores.
            Returns empty DataFrame if user features unavailable.
        """
        if self._user_clusters is None or self._cluster_preferences is None:
            raise RuntimeError("SegmentsScorer must be fit before scoring")

        # Find user's cluster
        user_cluster = (
            self._user_clusters
            .filter(F.col(USER_ID) == user_id)
            .select("cluster")
            .collect()
        )

        if not user_cluster:
            # User not in training data - return empty
            return candidates.select(ITEM_ID).withColumn(RAW_SCORE, F.lit(None)).limit(0)

        cluster_id = user_cluster[0]["cluster"]

        # Get preferences for this cluster
        cluster_prefs = self._cluster_preferences.filter(F.col("cluster") == cluster_id)

        # Join with candidates
        return (
            candidates
            .select(ITEM_ID)
            .join(cluster_prefs, on=ITEM_ID, how="left")
            .withColumn(
                RAW_SCORE,
                F.coalesce(F.col("cluster_score"), F.lit(0.0))
            )
            .select(ITEM_ID, RAW_SCORE)
        )

    def _manifest_extras(self) -> dict:
        return {"k": self._k}

    def save(self, path: Path) -> None:
        if (self._kmeans_model is None or self._user_clusters is None
                or self._cluster_preferences is None):
            raise RuntimeError("SegmentsScorer must be fit before save")

        super().save(path)
        path = Path(path)
        self._kmeans_model.save(str(path / "kmeans_model"))
        self._user_clusters.write.mode("overwrite").parquet(
            str(path / "user_clusters"))
        self._cluster_preferences.write.mode("overwrite").parquet(
            str(path / "cluster_preferences"))

    @classmethod
    def load(cls, path: Path, ctx: "FeatureStore") -> "SegmentsScorer":
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text())
        instance = cls(k=manifest.get("k", 8))
        instance._kmeans_model = KMeansModel.load(str(path / "kmeans_model"))
        instance._user_clusters = ctx.spark.read.parquet(
            str(path / "user_clusters"))
        instance._cluster_preferences = ctx.spark.read.parquet(
            str(path / "cluster_preferences"))
        instance._spark = ctx.spark
        return instance
