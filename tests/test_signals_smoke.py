"""Smoke tests for signals."""

import pytest
from pathlib import Path
from pyspark.sql import SparkSession

from preference_engine.signals.base import Scorer
from preference_engine.signals.popularity import PopularityScorer
from preference_engine.signals.trends import TrendsScorer
from preference_engine.signals.segments import SegmentsScorer
from preference_engine.signals.content import ContentScorer
from preference_engine.signals.als_cf import ALSScorer
from preference_engine.signals.rules import RulesScorer
from preference_engine.streaming.feature_store import FeatureStore, build_features_batch
from preference_engine.adapter.registry import get_adapter
from preference_engine.schema import ITEM_ID


class TestScorerInterface:
    """Tests for the Scorer abstract interface."""

    def test_scorer_is_abstract(self) -> None:
        """Test that Scorer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Scorer()  # type: ignore

    def test_scorer_subclass_requires_methods(self) -> None:
        """Test that subclasses must implement abstract methods."""

        class IncompleteScorer(Scorer):
            name = "incomplete"

        with pytest.raises(TypeError):
            IncompleteScorer()


@pytest.fixture(scope="module")
def feature_store(spark: SparkSession, tmp_path_factory) -> FeatureStore:
    """Create a populated feature store for signal tests."""
    store_path = tmp_path_factory.mktemp("feature_store")
    adapter = get_adapter("synthetic")
    build_features_batch(adapter, spark, store_path)
    return FeatureStore(path=store_path, spark=spark)


@pytest.fixture
def candidates(spark: SparkSession) -> "DataFrame":
    """Create candidate items DataFrame."""
    # Use a subset of items as candidates
    items = [(f"i{i}",) for i in range(20)]
    return spark.createDataFrame(items, [ITEM_ID])


class TestPopularityScorer:
    """Tests for popularity signal."""

    def test_fit_and_score(
        self, spark: SparkSession, feature_store: FeatureStore, candidates
    ) -> None:
        """Test popularity scorer fit and score."""
        scorer = PopularityScorer()
        scorer.fit(feature_store, {})

        result = scorer.raw_score("u1", candidates)

        assert ITEM_ID in result.columns
        assert "raw_score" in result.columns
        assert result.count() == candidates.count()

    def test_normalized_score(
        self, spark: SparkSession, feature_store: FeatureStore, candidates
    ) -> None:
        """Test that normalized scores are in [0, 1]."""
        from pyspark.sql import functions as F

        scorer = PopularityScorer()
        scorer.fit(feature_store, {})

        result = scorer.score("u1", candidates)

        # Check scores are normalized
        stats = result.agg(F.min("score"), F.max("score")).collect()[0]
        min_score = stats[0]
        max_score = stats[1]

        assert min_score >= 0.0
        assert max_score <= 1.0


class TestTrendsScorer:
    """Tests for trends signal."""

    def test_fit_and_score(
        self, spark: SparkSession, feature_store: FeatureStore, candidates
    ) -> None:
        """Test trends scorer fit and score."""
        scorer = TrendsScorer()
        scorer.fit(feature_store, {"halflife_days": 30})

        result = scorer.raw_score("u1", candidates)

        assert ITEM_ID in result.columns
        assert "raw_score" in result.columns
        assert result.count() == candidates.count()


class TestSegmentsScorer:
    """Tests for segments signal."""

    def test_fit_and_score(
        self, spark: SparkSession, feature_store: FeatureStore, candidates
    ) -> None:
        """Test segments scorer fit and score."""
        scorer = SegmentsScorer()
        scorer.fit(feature_store, {"k": 4})

        result = scorer.raw_score("u1", candidates)

        assert ITEM_ID in result.columns
        assert "raw_score" in result.columns

    def test_cold_user_returns_empty(
        self, spark: SparkSession, feature_store: FeatureStore, candidates
    ) -> None:
        """Test that unknown user returns empty DataFrame."""
        scorer = SegmentsScorer()
        scorer.fit(feature_store, {"k": 4})

        result = scorer.raw_score("unknown_user", candidates)

        assert result.count() == 0


class TestContentScorer:
    """Tests for content signal."""

    def test_fit_and_score(
        self, spark: SparkSession, feature_store: FeatureStore, candidates
    ) -> None:
        """Test content scorer fit and score."""
        scorer = ContentScorer()
        scorer.fit(feature_store, {})

        result = scorer.raw_score("u1", candidates)

        assert ITEM_ID in result.columns
        assert "raw_score" in result.columns

    def test_cold_user_returns_empty(
        self, spark: SparkSession, feature_store: FeatureStore, candidates
    ) -> None:
        """Test that user with no history returns empty DataFrame."""
        scorer = ContentScorer()
        scorer.fit(feature_store, {})

        result = scorer.raw_score("unknown_user", candidates)

        assert result.count() == 0


class TestALSScorer:
    """Tests for ALS signal."""

    def test_fit_and_score_explicit(
        self, spark: SparkSession, feature_store: FeatureStore, candidates
    ) -> None:
        """Test ALS scorer in explicit mode."""
        scorer = ALSScorer(implicit=False)
        scorer.fit(feature_store, {"rank": 10, "max_iter": 5})

        result = scorer.raw_score("u1", candidates)

        assert ITEM_ID in result.columns
        assert "raw_score" in result.columns

    def test_fit_and_score_implicit(
        self, spark: SparkSession, feature_store: FeatureStore, candidates
    ) -> None:
        """Test ALS scorer in implicit mode."""
        scorer = ALSScorer(implicit=True)
        scorer.fit(feature_store, {"rank": 10, "max_iter": 5})

        result = scorer.raw_score("u1", candidates)

        assert ITEM_ID in result.columns
        assert "raw_score" in result.columns

    def test_cold_user_returns_empty(
        self, spark: SparkSession, feature_store: FeatureStore, candidates
    ) -> None:
        """Test that cold user returns empty DataFrame."""
        scorer = ALSScorer(implicit=False)
        scorer.fit(feature_store, {"rank": 10, "max_iter": 5})

        result = scorer.raw_score("unknown_user", candidates)

        assert result.count() == 0


class TestRulesScorer:
    """Tests for rules signal."""

    def test_fit_and_score(
        self, spark: SparkSession, feature_store: FeatureStore, candidates
    ) -> None:
        """Test rules scorer fit and score."""
        scorer = RulesScorer()
        # Use low support threshold for small dataset
        scorer.fit(feature_store, {"min_support": 0.001, "min_confidence": 0.1})

        result = scorer.raw_score("u1", candidates)

        assert ITEM_ID in result.columns
        assert "raw_score" in result.columns

    def test_cold_user_returns_empty(
        self, spark: SparkSession, feature_store: FeatureStore, candidates
    ) -> None:
        """Test that user with no history returns empty DataFrame."""
        scorer = RulesScorer()
        scorer.fit(feature_store, {"min_support": 0.001, "min_confidence": 0.1})

        result = scorer.raw_score("unknown_user", candidates)

        assert result.count() == 0
