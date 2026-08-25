"""Tests for schema definitions."""

import pytest

from preference_engine.schema import (
    InteractionType,
    FeatureKind,
    Applies,
    FeatureSpec,
    USER_ID,
    ITEM_ID,
    VALUE,
    TS,
    USER_ID_INT,
    ITEM_ID_INT,
)


class TestInteractionType:
    """Tests for InteractionType enum."""

    def test_explicit_value(self) -> None:
        assert InteractionType.EXPLICIT.value == "explicit"

    def test_implicit_value(self) -> None:
        assert InteractionType.IMPLICIT.value == "implicit"

    def test_is_string_enum(self) -> None:
        assert isinstance(InteractionType.EXPLICIT, str)


class TestFeatureKind:
    """Tests for FeatureKind enum."""

    def test_all_kinds_exist(self) -> None:
        kinds = [FeatureKind.CATEGORICAL, FeatureKind.NUMERIC, FeatureKind.ORDINAL, FeatureKind.TEXT]
        assert len(kinds) == 4

    def test_values(self) -> None:
        assert FeatureKind.CATEGORICAL.value == "categorical"
        assert FeatureKind.NUMERIC.value == "numeric"


class TestApplies:
    """Tests for Applies enum."""

    def test_user_and_item(self) -> None:
        assert Applies.USER.value == "user"
        assert Applies.ITEM.value == "item"


class TestFeatureSpec:
    """Tests for FeatureSpec dataclass."""

    def test_creation(self) -> None:
        spec = FeatureSpec(name="test", kind=FeatureKind.NUMERIC, applies=Applies.USER)
        assert spec.name == "test"
        assert spec.kind == FeatureKind.NUMERIC
        assert spec.applies == Applies.USER

    def test_frozen(self) -> None:
        spec = FeatureSpec(name="test", kind=FeatureKind.NUMERIC, applies=Applies.USER)
        with pytest.raises(AttributeError):
            spec.name = "other"  # type: ignore


class TestColumnNames:
    """Tests for column name constants."""

    def test_canonical_columns(self) -> None:
        assert USER_ID == "user_id"
        assert ITEM_ID == "item_id"
        assert VALUE == "value"
        assert TS == "ts"

    def test_integer_id_columns(self) -> None:
        assert USER_ID_INT == "user_id_int"
        assert ITEM_ID_INT == "item_id_int"
