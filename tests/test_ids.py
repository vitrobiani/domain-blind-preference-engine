"""Tests for ID mapping utilities."""

import pytest
from pyspark.sql import SparkSession

from preference_engine.ids import IDMapper
from preference_engine.schema import USER_ID, ITEM_ID, USER_ID_INT, ITEM_ID_INT


class TestIDMapper:
    """Tests for the IDMapper class."""

    @pytest.fixture
    def sample_interactions(self, spark: SparkSession):
        """Create sample interaction data."""
        data = [
            ("u1", "i1", 4.0),
            ("u2", "i1", 3.0),
            ("u1", "i2", 5.0),
            ("u3", "i3", 2.0),
        ]
        return spark.createDataFrame(data, [USER_ID, ITEM_ID, "value"])

    def test_fit_and_transform(self, spark: SparkSession, sample_interactions) -> None:
        """Test fitting and transforming adds integer columns."""
        mapper = IDMapper().fit(sample_interactions)
        result = mapper.transform(sample_interactions)

        assert USER_ID_INT in result.columns
        assert ITEM_ID_INT in result.columns

    def test_transform_produces_integers(self, spark: SparkSession, sample_interactions) -> None:
        """Test that transformed IDs are integers."""
        mapper = IDMapper().fit(sample_interactions)
        result = mapper.transform(sample_interactions)

        schema = {f.name: f.dataType.simpleString() for f in result.schema.fields}
        assert schema[USER_ID_INT] == "double"  # StringIndexer produces doubles
        assert schema[ITEM_ID_INT] == "double"

    def test_user_id_to_int(self, spark: SparkSession, sample_interactions) -> None:
        """Test single user ID conversion."""
        mapper = IDMapper().fit(sample_interactions)

        idx = mapper.user_id_to_int("u1")
        assert idx is not None
        assert isinstance(idx, int)

    def test_item_id_to_int(self, spark: SparkSession, sample_interactions) -> None:
        """Test single item ID conversion."""
        mapper = IDMapper().fit(sample_interactions)

        idx = mapper.item_id_to_int("i1")
        assert idx is not None
        assert isinstance(idx, int)

    def test_unknown_id_returns_none(self, spark: SparkSession, sample_interactions) -> None:
        """Test that unknown IDs return None."""
        mapper = IDMapper().fit(sample_interactions)

        assert mapper.user_id_to_int("unknown") is None
        assert mapper.item_id_to_int("unknown") is None

    def test_roundtrip_user(self, spark: SparkSession, sample_interactions) -> None:
        """Test user ID round-trip: string -> int -> string."""
        mapper = IDMapper().fit(sample_interactions)

        original = "u2"
        idx = mapper.user_id_to_int(original)
        assert idx is not None
        recovered = mapper.int_to_user_id(idx)
        assert recovered == original

    def test_roundtrip_item(self, spark: SparkSession, sample_interactions) -> None:
        """Test item ID round-trip: string -> int -> string."""
        mapper = IDMapper().fit(sample_interactions)

        original = "i3"
        idx = mapper.item_id_to_int(original)
        assert idx is not None
        recovered = mapper.int_to_item_id(idx)
        assert recovered == original

    def test_inverse_transform_users(self, spark: SparkSession, sample_interactions) -> None:
        """Test inverse transform for users."""
        mapper = IDMapper().fit(sample_interactions)
        transformed = mapper.transform(sample_interactions)

        # Select just user_id_int and inverse transform
        user_ints = transformed.select(USER_ID_INT).distinct()
        result = mapper.inverse_transform_users(user_ints)

        assert USER_ID in result.columns
        user_ids = {row[USER_ID] for row in result.collect()}
        assert user_ids == {"u1", "u2", "u3"}

    def test_transform_without_fit_raises(self, spark: SparkSession, sample_interactions) -> None:
        """Test that transform without fit raises error."""
        mapper = IDMapper()
        with pytest.raises(RuntimeError, match="must be fit"):
            mapper.transform(sample_interactions)

    def test_save_load_roundtrip(self, spark: SparkSession, sample_interactions, tmp_path) -> None:
        """Test that a saved mapper reloads with identical mappings."""
        original = IDMapper().fit(sample_interactions)
        original.save(tmp_path / "mapper")

        loaded = IDMapper.load(tmp_path / "mapper", spark)

        # Every known user/item maps to the same integer as before
        for uid in ["u1", "u2", "u3"]:
            assert loaded.user_id_to_int(uid) == original.user_id_to_int(uid)
        for iid in ["i1", "i2", "i3"]:
            assert loaded.item_id_to_int(iid) == original.item_id_to_int(iid)

        # Unknown IDs still return None on the loaded mapper
        assert loaded.user_id_to_int("nobody") is None
        assert loaded.item_id_to_int("nothing") is None
