"""Tests for the feature store."""

import pytest
from pathlib import Path
from pyspark.sql import SparkSession

from preference_engine.adapter.registry import get_adapter
from preference_engine.streaming.feature_store import (
    FeatureStore,
    build_features_batch,
    _compute_popularity,
    _compute_recency,
    INTERACTIONS_TABLE,
    POPULARITY_TABLE,
    RECENCY_TABLE,
    USER_FEATURES_TABLE,
    ITEM_FEATURES_TABLE,
)
from preference_engine.schema import USER_ID, ITEM_ID, VALUE, TS


class TestBuildFeaturesBatch:
    """Tests for batch feature building."""

    @pytest.fixture
    def adapter(self):
        return get_adapter("synthetic")

    def test_creates_all_tables(
        self, spark: SparkSession, adapter, temp_feature_store: Path
    ) -> None:
        """Test that all tables are created."""
        build_features_batch(adapter, spark, temp_feature_store)

        assert (temp_feature_store / INTERACTIONS_TABLE).exists()
        assert (temp_feature_store / POPULARITY_TABLE).exists()
        assert (temp_feature_store / RECENCY_TABLE).exists()
        assert (temp_feature_store / USER_FEATURES_TABLE).exists()
        assert (temp_feature_store / ITEM_FEATURES_TABLE).exists()

    def test_interactions_schema(
        self, spark: SparkSession, adapter, temp_feature_store: Path
    ) -> None:
        """Test interactions table schema."""
        build_features_batch(adapter, spark, temp_feature_store)

        store = FeatureStore(path=temp_feature_store, spark=spark)
        df = store.interactions()

        assert USER_ID in df.columns
        assert ITEM_ID in df.columns
        assert VALUE in df.columns
        assert TS in df.columns

    def test_interactions_count(
        self, spark: SparkSession, adapter, temp_feature_store: Path
    ) -> None:
        """Test interactions table row count."""
        build_features_batch(adapter, spark, temp_feature_store)

        store = FeatureStore(path=temp_feature_store, spark=spark)
        assert store.interactions().count() == adapter.n_interactions

    def test_popularity_schema(
        self, spark: SparkSession, adapter, temp_feature_store: Path
    ) -> None:
        """Test popularity table schema."""
        build_features_batch(adapter, spark, temp_feature_store)

        store = FeatureStore(path=temp_feature_store, spark=spark)
        df = store.popularity()

        assert ITEM_ID in df.columns
        assert "count" in df.columns

    def test_popularity_count(
        self, spark: SparkSession, adapter, temp_feature_store: Path
    ) -> None:
        """Test popularity covers all items with interactions."""
        build_features_batch(adapter, spark, temp_feature_store)

        store = FeatureStore(path=temp_feature_store, spark=spark)
        # Should have at most n_items rows (some items may have 0 interactions)
        assert store.popularity().count() <= adapter.n_items

    def test_recency_schema(
        self, spark: SparkSession, adapter, temp_feature_store: Path
    ) -> None:
        """Test recency table schema."""
        build_features_batch(adapter, spark, temp_feature_store)

        store = FeatureStore(path=temp_feature_store, spark=spark)
        df = store.recency()

        assert ITEM_ID in df.columns
        assert "recency_score" in df.columns

    def test_user_features_count(
        self, spark: SparkSession, adapter, temp_feature_store: Path
    ) -> None:
        """Test user features count."""
        build_features_batch(adapter, spark, temp_feature_store)

        store = FeatureStore(path=temp_feature_store, spark=spark)
        assert store.user_features().count() == adapter.n_users

    def test_item_features_count(
        self, spark: SparkSession, adapter, temp_feature_store: Path
    ) -> None:
        """Test item features count."""
        build_features_batch(adapter, spark, temp_feature_store)

        store = FeatureStore(path=temp_feature_store, spark=spark)
        assert store.item_features().count() == adapter.n_items


class TestComputePopularity:
    """Tests for popularity computation."""

    def test_counts_interactions(self, spark: SparkSession) -> None:
        """Test that popularity counts interactions correctly."""
        data = [
            ("u1", "i1", 5.0),
            ("u2", "i1", 4.0),
            ("u1", "i2", 3.0),
            ("u3", "i1", 2.0),
        ]
        df = spark.createDataFrame(data, [USER_ID, ITEM_ID, VALUE])

        result = _compute_popularity(df)
        counts = {row[ITEM_ID]: row["count"] for row in result.collect()}

        assert counts["i1"] == 3
        assert counts["i2"] == 1


class TestComputeRecency:
    """Tests for recency computation."""

    def test_recent_items_score_higher(self, spark: SparkSession) -> None:
        """Test that recent interactions contribute more to score."""
        from datetime import datetime, timedelta

        now = datetime.now()
        data = [
            ("u1", "i1", 5.0, now - timedelta(days=1)),   # Recent
            ("u2", "i2", 5.0, now - timedelta(days=100)), # Old
        ]
        df = spark.createDataFrame(data, [USER_ID, ITEM_ID, VALUE, TS])

        result = _compute_recency(df, halflife_days=30, reference_time=now)
        scores = {row[ITEM_ID]: row["recency_score"] for row in result.collect()}

        # Recent item should have higher score
        assert scores["i1"] > scores["i2"]


class TestFeatureStore:
    """Tests for FeatureStore class."""

    @pytest.fixture
    def populated_store(
        self, spark: SparkSession, temp_feature_store: Path
    ) -> FeatureStore:
        """Create a populated feature store."""
        adapter = get_adapter("synthetic")
        build_features_batch(adapter, spark, temp_feature_store)
        return FeatureStore(path=temp_feature_store, spark=spark)

    def test_caching(self, populated_store: FeatureStore) -> None:
        """Test that DataFrames are cached."""
        # First access
        df1 = populated_store.interactions()
        # Second access should return same object
        df2 = populated_store.interactions()

        assert df1 is df2

    def test_clear_cache(self, populated_store: FeatureStore) -> None:
        """Test cache clearing."""
        df1 = populated_store.interactions()
        populated_store.clear_cache()
        df2 = populated_store.interactions()

        # After clear, should be different object
        assert df1 is not df2

    def test_missing_table_raises(
        self, spark: SparkSession, temp_feature_store: Path
    ) -> None:
        """Test that missing table raises FileNotFoundError."""
        store = FeatureStore(path=temp_feature_store, spark=spark)

        with pytest.raises(FileNotFoundError):
            store.interactions()
