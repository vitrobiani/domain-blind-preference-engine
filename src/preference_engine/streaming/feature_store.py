"""
Feature store builder and accessor.

Provides both streaming (Kafka) and batch paths for building the feature store.
The feature store contains preprocessed data for all signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from datetime import datetime

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    TimestampType,
)

from preference_engine.schema import USER_ID, ITEM_ID, VALUE, TS
from preference_engine.ingest.topics import INTERACTIONS_TOPIC

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession
    from preference_engine.adapter.base import DomainAdapter
    from preference_engine.ingest.topics import InteractionEvent


INTERACTIONS_SCHEMA = StructType([
    StructField(USER_ID, StringType(), False),
    StructField(ITEM_ID, StringType(), False),
    StructField(VALUE, DoubleType(), True),
    StructField(TS, TimestampType(), False),
])


# Default feature store location
DEFAULT_FEATURE_STORE_PATH = Path(".data/feature_store")

# Table names within the feature store
INTERACTIONS_TABLE = "interactions"
POPULARITY_TABLE = "popularity"
RECENCY_TABLE = "recency"
USER_FEATURES_TABLE = "user_features"
ITEM_FEATURES_TABLE = "item_features"


@dataclass
class FeatureStore:
    """
    Interface to the feature store.

    The feature store contains:
    - interactions: Normalized interaction table
    - popularity: Per-item interaction counts
    - recency: Per-item recency-weighted scores
    - user_features: Vectorized user features
    - item_features: Vectorized item features

    Attributes:
        path: Root path of the feature store.
        spark: SparkSession for reading data.
    """

    path: Path
    spark: "SparkSession"
    _cache: dict[str, "DataFrame"] = field(default_factory=dict)

    def _read_table(self, name: str) -> "DataFrame":
        """Read a parquet table from the feature store with caching."""
        if name not in self._cache:
            table_path = self.path / name
            if not table_path.exists():
                raise FileNotFoundError(f"Feature store table not found: {table_path}")
            self._cache[name] = self.spark.read.parquet(str(table_path))
        return self._cache[name]

    def interactions(self) -> "DataFrame":
        """Load the interactions table."""
        return self._read_table(INTERACTIONS_TABLE)

    def popularity(self) -> "DataFrame":
        """Load the popularity aggregates (item_id, count)."""
        return self._read_table(POPULARITY_TABLE)

    def recency(self) -> "DataFrame":
        """Load the recency aggregates (item_id, recency_score)."""
        return self._read_table(RECENCY_TABLE)

    def user_features(self) -> "DataFrame":
        """Load the user features table."""
        return self._read_table(USER_FEATURES_TABLE)

    def item_features(self) -> "DataFrame":
        """Load the item features table."""
        return self._read_table(ITEM_FEATURES_TABLE)

    def clear_cache(self) -> None:
        """Clear the DataFrame cache."""
        self._cache.clear()


def _compute_popularity(interactions: "DataFrame") -> "DataFrame":
    """
    Compute popularity scores (interaction counts per item).

    Args:
        interactions: DataFrame with interaction data.

    Returns:
        DataFrame[item_id, count] with interaction counts.
    """
    return (
        interactions
        .groupBy(ITEM_ID)
        .agg(F.count("*").alias("count"))
    )


def _compute_recency(
    interactions: "DataFrame",
    halflife_days: float = 30.0,
    reference_time: datetime | None = None,
) -> "DataFrame":
    """
    Compute recency-weighted popularity scores.

    Uses exponential decay based on interaction timestamp.
    score = sum(exp(-lambda * age_in_days)) where lambda = ln(2) / halflife

    Args:
        interactions: DataFrame with interaction data including timestamps.
        halflife_days: Half-life for exponential decay.
        reference_time: Reference time for computing age (default: now).

    Returns:
        DataFrame[item_id, recency_score] with recency-weighted scores.
    """
    if reference_time is None:
        reference_time = datetime.now()

    # Decay constant: lambda = ln(2) / halflife
    decay_constant = 0.693147 / halflife_days  # ln(2) ≈ 0.693147

    # Compute age in days and apply exponential decay
    return (
        interactions
        .withColumn(
            "age_days",
            F.datediff(F.lit(reference_time), F.col(TS))
        )
        .withColumn(
            "decay_weight",
            F.exp(-decay_constant * F.col("age_days"))
        )
        .groupBy(ITEM_ID)
        .agg(F.sum("decay_weight").alias("recency_score"))
    )


def build_features_batch(
    adapter: "DomainAdapter",
    spark: "SparkSession",
    output_path: Path = DEFAULT_FEATURE_STORE_PATH,
) -> None:
    """
    Build the feature store from adapter data (batch mode).

    Reads directly from the adapter's DataFrames without Kafka.
    Used for development and testing.

    Args:
        adapter: Domain adapter providing the data.
        spark: Active SparkSession.
        output_path: Where to write the feature store.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load raw data from adapter
    print("  Loading interactions...")
    interactions = adapter.interactions(spark)

    print("  Loading user features...")
    user_features = adapter.user_features(spark)

    print("  Loading item features...")
    item_features = adapter.item_features(spark)

    # Write interactions table
    print(f"  Writing {INTERACTIONS_TABLE}...")
    interactions.write.mode("overwrite").parquet(str(output_path / INTERACTIONS_TABLE))

    # Compute and write popularity
    print(f"  Computing and writing {POPULARITY_TABLE}...")
    popularity = _compute_popularity(interactions)
    popularity.write.mode("overwrite").parquet(str(output_path / POPULARITY_TABLE))

    # Compute and write recency scores
    print(f"  Computing and writing {RECENCY_TABLE}...")
    recency = _compute_recency(interactions)
    recency.write.mode("overwrite").parquet(str(output_path / RECENCY_TABLE))

    # Write user features
    print(f"  Writing {USER_FEATURES_TABLE}...")
    user_features.write.mode("overwrite").parquet(str(output_path / USER_FEATURES_TABLE))

    # Write item features
    print(f"  Writing {ITEM_FEATURES_TABLE}...")
    item_features.write.mode("overwrite").parquet(str(output_path / ITEM_FEATURES_TABLE))

    print(f"  Feature store built at {output_path}")


def bootstrap_features(
    adapter: "DomainAdapter",
    spark: "SparkSession",
    output_path: Path = DEFAULT_FEATURE_STORE_PATH,
) -> None:
    """
    Build a feature store containing only the catalog - no interactions.

    Used for the streaming setup where interactions arrive live via POST /interactions.
    Writes user_features and item_features from the adapter, and empty
    interactions / popularity / recency tables so downstream signals can read them.

    Any existing feature store at output_path is overwritten in place.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print("  Loading user features...")
    user_features = adapter.user_features(spark)
    print("  Loading item features...")
    item_features = adapter.item_features(spark)

    print(f"  Writing empty {INTERACTIONS_TABLE}...")
    empty_interactions = spark.createDataFrame([], INTERACTIONS_SCHEMA)
    empty_interactions.write.mode("overwrite").parquet(str(output_path / INTERACTIONS_TABLE))

    print(f"  Writing empty {POPULARITY_TABLE}...")
    empty_pop = spark.createDataFrame([], f"{ITEM_ID} string, count long")
    empty_pop.write.mode("overwrite").parquet(str(output_path / POPULARITY_TABLE))

    print(f"  Writing empty {RECENCY_TABLE}...")
    empty_rec = spark.createDataFrame([], f"{ITEM_ID} string, recency_score double")
    empty_rec.write.mode("overwrite").parquet(str(output_path / RECENCY_TABLE))

    print(f"  Writing {USER_FEATURES_TABLE}...")
    user_features.write.mode("overwrite").parquet(str(output_path / USER_FEATURES_TABLE))

    print(f"  Writing {ITEM_FEATURES_TABLE}...")
    item_features.write.mode("overwrite").parquet(str(output_path / ITEM_FEATURES_TABLE))

    print(f"  Bootstrap complete at {output_path}")


def append_interactions(
    spark: "SparkSession",
    output_path: Path,
    events: list["InteractionEvent"],
) -> int:
    """
    Append a batch of interaction events to the feature store's interactions table.

    Returns the number of rows appended.
    """
    if not events:
        return 0

    output_path = Path(output_path)
    rows = [(e.user_id, e.item_id, e.value, e.ts) for e in events]
    df = spark.createDataFrame(rows, INTERACTIONS_SCHEMA)
    df.write.mode("append").parquet(str(output_path / INTERACTIONS_TABLE))
    return len(events)


def refresh_aggregates(
    spark: "SparkSession",
    output_path: Path,
) -> None:
    """
    Recompute popularity + recency from the current interactions table
    and overwrite the corresponding parquet tables.

    Call this after append_interactions() so signals see fresh aggregates.
    """
    output_path = Path(output_path)
    interactions = spark.read.parquet(str(output_path / INTERACTIONS_TABLE))

    popularity = _compute_popularity(interactions)
    popularity.write.mode("overwrite").parquet(str(output_path / POPULARITY_TABLE))

    recency = _compute_recency(interactions)
    recency.write.mode("overwrite").parquet(str(output_path / RECENCY_TABLE))


def build_features_streaming(
    spark: "SparkSession",
    bootstrap_servers: str = "localhost:9092",
    output_path: Path = DEFAULT_FEATURE_STORE_PATH,
) -> None:
    """
    Build the feature store from Kafka (streaming mode).

    Runs a Structured Streaming job that consumes interaction events
    and maintains the feature store.

    Args:
        spark: Active SparkSession.
        bootstrap_servers: Kafka bootstrap servers.
        output_path: Where to write the feature store.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define schema for Kafka messages
    interaction_schema = StructType([
        StructField(USER_ID, StringType(), False),
        StructField(ITEM_ID, StringType(), False),
        StructField(VALUE, DoubleType(), True),
        StructField(TS, StringType(), False),  # ISO format string
    ])

    # Read from Kafka
    print(f"  Starting streaming from {INTERACTIONS_TOPIC}...")
    kafka_df = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("subscribe", INTERACTIONS_TOPIC)
        .option("startingOffsets", "earliest")
        .load()
    )

    # Parse JSON messages
    parsed_df = (
        kafka_df
        .selectExpr("CAST(value AS STRING) as json")
        .select(F.from_json(F.col("json"), interaction_schema).alias("data"))
        .select("data.*")
        .withColumn(TS, F.to_timestamp(F.col(TS)))
    )

    # Write interactions as a streaming sink
    interactions_query = (
        parsed_df.writeStream
        .format("parquet")
        .option("path", str(output_path / INTERACTIONS_TABLE))
        .option("checkpointLocation", str(output_path / ".checkpoints" / INTERACTIONS_TABLE))
        .outputMode("append")
        .start()
    )

    print(f"  Streaming to {output_path / INTERACTIONS_TABLE}")
    print("  Press Ctrl+C to stop...")

    # Wait for termination
    interactions_query.awaitTermination()
