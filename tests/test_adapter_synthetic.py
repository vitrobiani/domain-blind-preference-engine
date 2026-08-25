"""Tests for the synthetic adapter."""

import pytest
from pyspark.sql import SparkSession

from preference_engine.adapter.registry import get_adapter, list_adapters
from preference_engine.schema import (
    USER_ID,
    ITEM_ID,
    VALUE,
    TS,
    FeatureKind,
    Applies,
)


class TestAdapterRegistry:
    """Tests for adapter discovery and loading."""

    def test_list_adapters_includes_synthetic(self) -> None:
        adapters = list_adapters()
        assert "synthetic" in adapters

    def test_get_adapter_synthetic(self) -> None:
        adapter = get_adapter("synthetic")
        assert adapter.name == "synthetic"

    def test_get_adapter_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            get_adapter("nonexistent_adapter")


class TestSyntheticAdapterSchema:
    """Tests for synthetic adapter data schemas."""

    @pytest.fixture
    def adapter(self):
        return get_adapter("synthetic")

    def test_interactions_columns(self, spark: SparkSession, adapter) -> None:
        """Test that interactions has correct columns."""
        df = adapter.interactions(spark)
        assert USER_ID in df.columns
        assert ITEM_ID in df.columns
        assert VALUE in df.columns
        assert TS in df.columns

    def test_interactions_types(self, spark: SparkSession, adapter) -> None:
        """Test that interactions has correct types."""
        df = adapter.interactions(spark)
        schema = {f.name: f.dataType.simpleString() for f in df.schema.fields}
        assert schema[USER_ID] == "string"
        assert schema[ITEM_ID] == "string"
        assert schema[VALUE] == "double"
        assert schema[TS] == "timestamp"

    def test_user_features_columns(self, spark: SparkSession, adapter) -> None:
        """Test that user_features has user_id and feature columns."""
        df = adapter.user_features(spark)
        assert USER_ID in df.columns
        assert "age_group" in df.columns
        assert "activity_level" in df.columns

    def test_item_features_columns(self, spark: SparkSession, adapter) -> None:
        """Test that item_features has item_id and feature columns."""
        df = adapter.item_features(spark)
        assert ITEM_ID in df.columns
        assert "category" in df.columns
        assert "quality_score" in df.columns

    def test_feature_specs(self, adapter) -> None:
        """Test that feature_specs describes all features."""
        specs = adapter.feature_specs()
        names = {s.name for s in specs}
        assert names == {"age_group", "activity_level", "category", "quality_score"}

        # Check kinds
        spec_dict = {s.name: s for s in specs}
        assert spec_dict["age_group"].kind == FeatureKind.CATEGORICAL
        assert spec_dict["activity_level"].kind == FeatureKind.NUMERIC
        assert spec_dict["category"].kind == FeatureKind.CATEGORICAL
        assert spec_dict["quality_score"].kind == FeatureKind.NUMERIC

        # Check applies
        assert spec_dict["age_group"].applies == Applies.USER
        assert spec_dict["category"].applies == Applies.ITEM


class TestSyntheticAdapterData:
    """Tests for synthetic adapter data content."""

    @pytest.fixture
    def adapter(self):
        return get_adapter("synthetic")

    def test_interactions_count(self, spark: SparkSession, adapter) -> None:
        """Test default interaction count."""
        df = adapter.interactions(spark)
        assert df.count() == adapter.n_interactions

    def test_user_features_count(self, spark: SparkSession, adapter) -> None:
        """Test user count matches config."""
        df = adapter.user_features(spark)
        assert df.count() == adapter.n_users

    def test_item_features_count(self, spark: SparkSession, adapter) -> None:
        """Test item count matches config."""
        df = adapter.item_features(spark)
        assert df.count() == adapter.n_items

    def test_reproducibility(self, spark: SparkSession) -> None:
        """Test that same seed produces same data."""
        from adapters.synthetic.adapter import SyntheticAdapter

        # Use smaller dataset for faster test
        adapter1 = SyntheticAdapter(n_users=20, n_items=10, n_interactions=50, seed=123)
        adapter2 = SyntheticAdapter(n_users=20, n_items=10, n_interactions=50, seed=123)

        # Compare first 10 rows to verify determinism
        df1 = adapter1.interactions(spark).limit(10).collect()
        df2 = adapter2.interactions(spark).limit(10).collect()

        assert df1 == df2
