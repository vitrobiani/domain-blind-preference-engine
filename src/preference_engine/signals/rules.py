"""
Association rules signal using FP-Growth.

Finds item co-occurrence patterns and scores candidates by association
strength to items the user has already interacted with.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from pyspark.sql import functions as F
from pyspark.ml.fpm import FPGrowth

from preference_engine.signals.base import Scorer, RAW_SCORE
from preference_engine.schema import ITEM_ID, USER_ID

if TYPE_CHECKING:
    from pyspark.sql import DataFrame
    from preference_engine.streaming.feature_store import FeatureStore


class RulesScorer(Scorer):
    """
    Scorer using FP-Growth association rules.

    Treats each user's interaction history as a transaction.
    Mines association rules between items.
    Scores candidate items by association strength to user's history.
    """

    name = "rules"

    def __init__(self) -> None:
        self._rules: "DataFrame | None" = None
        self._user_items: "DataFrame | None" = None
        self._spark = None

    def fit(self, ctx: "FeatureStore", params: dict) -> None:
        """
        Mine association rules from interaction patterns.

        Args:
            ctx: Feature store with interactions.
            params: FP-Growth parameters (min_support, min_confidence).
        """
        min_support = params.get("min_support", 0.01)
        min_confidence = params.get("min_confidence", 0.2)

        # Load interactions
        interactions = ctx.interactions()
        self._spark = interactions.sparkSession

        # Create transactions: list of items per user
        transactions = (
            interactions
            .groupBy(USER_ID)
            .agg(F.collect_set(ITEM_ID).alias("items"))
        )

        # Store user items for later lookup
        self._user_items = (
            interactions
            .select(USER_ID, ITEM_ID)
            .distinct()
        )

        # Fit FP-Growth
        fp = FPGrowth(
            itemsCol="items",
            minSupport=min_support,
            minConfidence=min_confidence,
        )

        model = fp.fit(transactions)

        # Get association rules
        # Rules have: antecedent, consequent, confidence, lift, support
        self._rules = model.associationRules

    def raw_score(self, user_id: str, candidates: "DataFrame") -> "DataFrame":
        """
        Score candidates by association to user's interaction history.

        Args:
            user_id: User whose history to match against.
            candidates: Items to score.

        Returns:
            DataFrame[item_id, raw_score] with association confidence.
            Returns empty DataFrame if no relevant rules found.
        """
        if self._rules is None or self._user_items is None:
            raise RuntimeError("RulesScorer must be fit before scoring")

        # Get user's interaction history
        user_history = (
            self._user_items
            .filter(F.col(USER_ID) == user_id)
            .select(ITEM_ID)
            .collect()
        )

        if not user_history:
            # No history - return empty
            return candidates.select(ITEM_ID).withColumn(RAW_SCORE, F.lit(None)).limit(0)

        user_items_set = {row[ITEM_ID] for row in user_history}

        # Find rules where antecedent is subset of user's history
        # and consequent is in candidates
        @F.udf("boolean")
        def antecedent_in_history(antecedent):
            if antecedent is None:
                return False
            return set(antecedent).issubset(user_items_set)

        # Filter rules to those applicable to this user
        applicable_rules = (
            self._rules
            .filter(antecedent_in_history(F.col("antecedent")))
        )

        # Explode consequent and aggregate confidence per item
        item_scores = (
            applicable_rules
            .select(
                F.explode("consequent").alias(ITEM_ID),
                "confidence"
            )
            .groupBy(ITEM_ID)
            .agg(F.max("confidence").alias("max_confidence"))
        )

        # Join with candidates
        return (
            candidates
            .select(ITEM_ID)
            .join(item_scores, on=ITEM_ID, how="left")
            .withColumn(
                RAW_SCORE,
                F.coalesce(F.col("max_confidence"), F.lit(0.0))
            )
            .select(ITEM_ID, RAW_SCORE)
        )

    def save(self, path: Path) -> None:
        if self._rules is None or self._user_items is None:
            raise RuntimeError("RulesScorer must be fit before save")

        super().save(path)
        path = Path(path)
        self._rules.write.mode("overwrite").parquet(str(path / "rules"))
        self._user_items.write.mode("overwrite").parquet(str(path / "user_items"))

    @classmethod
    def load(cls, path: Path, ctx: "FeatureStore") -> "RulesScorer":
        path = Path(path)
        instance = cls()
        instance._rules = ctx.spark.read.parquet(str(path / "rules"))
        instance._user_items = ctx.spark.read.parquet(str(path / "user_items"))
        instance._spark = ctx.spark
        return instance
