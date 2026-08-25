"""
Base adapter interface for domain-specific data loading.

Each domain (movies, products, etc.) implements this interface to
transform raw data into the canonical schema.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from preference_engine.schema import InteractionType, FeatureSpec

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


class DomainAdapter(ABC):
    """
    Abstract base class for domain-specific data adapters.

    An adapter is the only place that understands the raw dataset format.
    It transforms domain data into the canonical schema used by all signals.

    Subclasses must set:
        name: Unique identifier for this adapter.
        dataset_name: Short slug identifying the underlying dataset. Used
            to derive per-dataset storage paths (e.g. .data/<dataset_name>/…),
            so builds against different datasets never clobber each other.
        interaction_type: Whether interactions are explicit or implicit.

    Subclasses must implement:
        interactions(): Return canonical interactions DataFrame.
        user_features(): Return user features DataFrame.
        item_features(): Return item features DataFrame.
        feature_specs(): Return list of feature specifications.
    """

    name: str
    dataset_name: str
    interaction_type: InteractionType

    @abstractmethod
    def interactions(self, spark: "SparkSession") -> "DataFrame":
        """
        Load and transform interactions into canonical format.

        Args:
            spark: Active SparkSession.

        Returns:
            DataFrame with columns: user_id (string), item_id (string),
            value (double, nullable), ts (timestamp).
        """
        raise NotImplementedError

    @abstractmethod
    def user_features(self, spark: "SparkSession") -> "DataFrame":
        """
        Load and transform user features into canonical format.

        Args:
            spark: Active SparkSession.

        Returns:
            DataFrame with user_id (string) plus feature columns.
            May return just user_id if no features exist.
        """
        raise NotImplementedError

    @abstractmethod
    def item_features(self, spark: "SparkSession") -> "DataFrame":
        """
        Load and transform item features into canonical format.

        Args:
            spark: Active SparkSession.

        Returns:
            DataFrame with item_id (string) plus feature columns.
            May return just item_id if no features exist.
        """
        raise NotImplementedError

    @abstractmethod
    def feature_specs(self) -> list[FeatureSpec]:
        """
        Describe all feature columns for generic vectorization.

        Returns:
            List of FeatureSpec objects describing each feature column
            in user_features and item_features DataFrames.
        """
        raise NotImplementedError
