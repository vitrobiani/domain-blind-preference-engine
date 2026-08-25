"""
Canonical schema definitions for the preference engine.

This module defines the fixed column names and types that all adapters must
produce and all signals must consume. Domain-agnostic by design.
"""

from dataclasses import dataclass
from enum import Enum


class InteractionType(str, Enum):
    """Type of user-item interaction in the dataset."""

    EXPLICIT = "explicit"  # Value is a stated rating/score (e.g., 1-5 stars)
    IMPLICIT = "implicit"  # Value is a confidence/count (e.g., watch time, purchases)


class FeatureKind(str, Enum):
    """Kind of feature for vectorization purposes."""

    CATEGORICAL = "categorical"  # Finite set of discrete values
    NUMERIC = "numeric"  # Continuous numeric values
    ORDINAL = "ordinal"  # Ordered discrete values
    TEXT = "text"  # Free-form text (treated as categorical/hashed)


class Applies(str, Enum):
    """Whether a feature applies to users or items."""

    USER = "user"
    ITEM = "item"


@dataclass(frozen=True)
class FeatureSpec:
    """
    Specification for a single feature column.

    Used by generic code to properly vectorize features without
    domain knowledge.

    Attributes:
        name: Column name in the DataFrame.
        kind: Type of feature for encoding strategy.
        applies: Whether this feature is on users or items.
    """

    name: str
    kind: FeatureKind
    applies: Applies


# Fixed column names - import these everywhere; never hardcode strings
USER_ID = "user_id"
ITEM_ID = "item_id"
VALUE = "value"
TS = "ts"

# Integer ID columns (for ALS and other algorithms requiring int IDs)
USER_ID_INT = "user_id_int"
ITEM_ID_INT = "item_id_int"
