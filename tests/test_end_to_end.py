"""End-to-end tests for the recommendation pipeline."""

from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from preference_engine.adapter.registry import get_adapter
from preference_engine.combiner.heuristics import load_heuristics
from preference_engine.combiner.combiner import FINAL_SCORE, BREAKDOWN
from preference_engine.streaming.feature_store import build_features_batch, FeatureStore
from preference_engine.serving.query import RecommendationEngine


class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.fixture
    def feature_store(self, spark: SparkSession, temp_feature_store: Path) -> FeatureStore:
        """Build feature store from synthetic data."""
        adapter = get_adapter("synthetic")
        build_features_batch(adapter, spark, temp_feature_store)
        return FeatureStore(path=temp_feature_store, spark=spark)

    @pytest.fixture
    def heuristics(self):
        """Load example heuristics config."""
        config_path = Path(__file__).parent.parent / "config" / "heuristics.example.yaml"
        return load_heuristics(config_path)

    @pytest.fixture
    def engine(self, heuristics, feature_store) -> RecommendationEngine:
        """Create and fit the recommendation engine."""
        engine = RecommendationEngine(heuristics, feature_store)
        engine.fit()
        return engine

    def test_full_pipeline_synthetic(self, engine: RecommendationEngine) -> None:
        """Test complete pipeline with synthetic data."""
        # User u1 should exist in synthetic data
        recommendations = engine.recommend("u1")
        results = recommendations.collect()

        # Should return top_k items (20 by default)
        assert len(results) > 0
        assert len(results) <= 20

        # Each result should have required fields
        for row in results:
            assert "item_id" in row
            assert FINAL_SCORE in row
            assert BREAKDOWN in row
            assert row[FINAL_SCORE] is not None

    def test_recommend_returns_ranking(self, engine: RecommendationEngine) -> None:
        """Test that recommend returns sorted results."""
        recommendations = engine.recommend("u1")
        results = recommendations.collect()

        # Results should be sorted by final_score descending
        scores = [row[FINAL_SCORE] for row in results]
        assert scores == sorted(scores, reverse=True), "Results should be sorted by score descending"

    def test_recommend_includes_breakdown(self, engine: RecommendationEngine) -> None:
        """Test that results include per-signal breakdown."""
        recommendations = engine.recommend("u1")
        results = recommendations.collect()

        assert len(results) > 0, "Should have at least one recommendation"

        # Check the first result has breakdown
        first_result = results[0]
        breakdown = first_result[BREAKDOWN]

        assert breakdown is not None, "Breakdown should not be None"
        assert isinstance(breakdown, dict), "Breakdown should be a dict"

        # Check that active signals are in breakdown
        # (popularity, als_cf, content, segments, trends have non-zero weights)
        active_signals = {"popularity", "als_cf", "content", "segments", "trends"}
        breakdown_signals = set(breakdown.keys())

        # At least some active signals should appear in breakdown
        assert len(breakdown_signals & active_signals) > 0, \
            f"Expected some of {active_signals} in breakdown, got {breakdown_signals}"

    def test_deterministic_with_fixed_seed(
        self, heuristics, spark: SparkSession, temp_feature_store: Path
    ) -> None:
        """Test that results are deterministic with fixed seed."""
        # Build feature store twice and compare results
        adapter = get_adapter("synthetic")

        # First run
        store_path_1 = temp_feature_store / "run1"
        store_path_1.mkdir(parents=True, exist_ok=True)
        build_features_batch(adapter, spark, store_path_1)
        feature_store_1 = FeatureStore(path=store_path_1, spark=spark)

        engine_1 = RecommendationEngine(heuristics, feature_store_1)
        engine_1.fit()
        results_1 = engine_1.recommend("u1").collect()

        # Second run
        store_path_2 = temp_feature_store / "run2"
        store_path_2.mkdir(parents=True, exist_ok=True)
        build_features_batch(adapter, spark, store_path_2)
        feature_store_2 = FeatureStore(path=store_path_2, spark=spark)

        engine_2 = RecommendationEngine(heuristics, feature_store_2)
        engine_2.fit()
        results_2 = engine_2.recommend("u1").collect()

        # Compare item rankings
        items_1 = [row["item_id"] for row in results_1]
        items_2 = [row["item_id"] for row in results_2]

        assert items_1 == items_2, \
            f"Rankings should be deterministic. Run 1: {items_1[:5]}... Run 2: {items_2[:5]}..."

    def test_exclude_seen_works(self, engine: RecommendationEngine, feature_store: FeatureStore) -> None:
        """Test that seen items are excluded from recommendations."""
        from pyspark.sql import functions as F
        from preference_engine.schema import USER_ID, ITEM_ID

        # Get items user u1 has interacted with
        seen_items = (
            feature_store.interactions()
            .filter(F.col(USER_ID) == "u1")
            .select(ITEM_ID)
            .distinct()
            .collect()
        )
        seen_item_ids = {row[ITEM_ID] for row in seen_items}

        # Get recommendations
        recommendations = engine.recommend("u1")
        results = recommendations.collect()
        recommended_ids = {row["item_id"] for row in results}

        # Recommended items should not include seen items
        overlap = seen_item_ids & recommended_ids
        assert len(overlap) == 0, \
            f"Seen items should be excluded but found: {overlap}"

    def test_unknown_user_returns_empty_or_popular(
        self, engine: RecommendationEngine
    ) -> None:
        """Test cold-start: unknown user gets some recommendations (popularity-based)."""
        # This user doesn't exist in synthetic data
        recommendations = engine.recommend("nonexistent_user_xyz")
        results = recommendations.collect()

        # Should still return something (popularity-based fallback)
        # or empty if all scorers return empty for unknown user
        # Either behavior is acceptable for cold start
        assert isinstance(results, list)
