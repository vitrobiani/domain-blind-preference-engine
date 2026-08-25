"""
Synthetic data adapter for testing and development.

Generates reproducible fake data with configurable size.
"""

import random
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    TimestampType,
)

from preference_engine.adapter.base import DomainAdapter
from preference_engine.schema import (
    InteractionType,
    FeatureSpec,
    FeatureKind,
    Applies,
    USER_ID,
    ITEM_ID,
    VALUE,
    TS,
)

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


class SyntheticAdapter(DomainAdapter):
    """
    Adapter generating synthetic test data.

    Creates reproducible fake users, items, and interactions.
    Used for testing the pipeline without external data dependencies.

    Configuration:
        n_users: Number of users to generate (default 200).
        n_items: Number of items to generate (default 100).
        n_interactions: Number of interactions to generate (default 2000).
        seed: Random seed for reproducibility (default 42).
    """

    name = "synthetic"
    dataset_name = "synthetic"
    interaction_type = InteractionType.EXPLICIT

    # Constants for synthetic data
    AGE_GROUPS = ["young", "adult", "senior"]
    CATEGORIES = ["A", "B", "C", "D"]

    def __init__(
        self,
        n_users: int = 200,
        n_items: int = 100,
        n_interactions: int = 2000,
        seed: int = 42,
    ) -> None:
        self.n_users = n_users
        self.n_items = n_items
        self.n_interactions = n_interactions
        self.seed = seed

    def interactions(self, spark: "SparkSession") -> "DataFrame":
        """
        Generate synthetic interaction data.

        Creates random user-item interactions with ratings (1-5)
        and timestamps spread over the past year.

        Returns:
            DataFrame[user_id, item_id, value, ts]
        """
        rng = random.Random(self.seed)

        # Fixed base timestamp for reproducibility (June 1, 2025)
        base_time = datetime(2025, 6, 1, 0, 0, 0)

        # Generate interactions
        data = []
        for _ in range(self.n_interactions):
            user_idx = rng.randint(0, self.n_users - 1)
            item_idx = rng.randint(0, self.n_items - 1)
            rating = float(rng.randint(1, 5))
            # Random timestamp within the past year
            ts = base_time + timedelta(seconds=rng.randint(0, 365 * 24 * 3600))

            data.append((f"u{user_idx}", f"i{item_idx}", rating, ts))

        schema = StructType([
            StructField(USER_ID, StringType(), False),
            StructField(ITEM_ID, StringType(), False),
            StructField(VALUE, DoubleType(), True),
            StructField(TS, TimestampType(), False),
        ])

        return spark.createDataFrame(data, schema)

    def user_features(self, spark: "SparkSession") -> "DataFrame":
        """
        Generate synthetic user features.

        Creates users with:
        - age_group: categorical (young, adult, senior)
        - activity_level: numeric (0-100)

        Returns:
            DataFrame[user_id, age_group, activity_level]
        """
        rng = random.Random(self.seed + 1)  # Different seed for user features

        data = []
        for i in range(self.n_users):
            age_group = rng.choice(self.AGE_GROUPS)
            activity_level = round(rng.uniform(0, 100), 2)
            data.append((f"u{i}", age_group, activity_level))

        schema = StructType([
            StructField(USER_ID, StringType(), False),
            StructField("age_group", StringType(), False),
            StructField("activity_level", DoubleType(), False),
        ])

        return spark.createDataFrame(data, schema)

    def item_features(self, spark: "SparkSession") -> "DataFrame":
        """
        Generate synthetic item features.

        Creates items with:
        - category: categorical (A, B, C, D)
        - quality_score: numeric (0-10)

        Returns:
            DataFrame[item_id, category, quality_score]
        """
        rng = random.Random(self.seed + 2)  # Different seed for item features

        data = []
        for i in range(self.n_items):
            category = rng.choice(self.CATEGORIES)
            quality_score = round(rng.uniform(0, 10), 2)
            data.append((f"i{i}", category, quality_score))

        schema = StructType([
            StructField(ITEM_ID, StringType(), False),
            StructField("category", StringType(), False),
            StructField("quality_score", DoubleType(), False),
        ])

        return spark.createDataFrame(data, schema)

    def feature_specs(self) -> list[FeatureSpec]:
        """
        Return feature specifications for synthetic data.

        Returns:
            List of FeatureSpec for age_group, activity_level,
            category, and quality_score.
        """
        return [
            FeatureSpec(name="age_group", kind=FeatureKind.CATEGORICAL, applies=Applies.USER),
            FeatureSpec(name="activity_level", kind=FeatureKind.NUMERIC, applies=Applies.USER),
            FeatureSpec(name="category", kind=FeatureKind.CATEGORICAL, applies=Applies.ITEM),
            FeatureSpec(name="quality_score", kind=FeatureKind.NUMERIC, applies=Applies.ITEM),
        ]
