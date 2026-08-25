"""Round-trip tests for scorer save/load."""

from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from preference_engine.adapter.registry import get_adapter
from preference_engine.streaming.feature_store import (
    FeatureStore,
    build_features_batch,
)
from preference_engine.signals import save_scorers, load_scorers, get_scorer
from preference_engine.signals.base import SCORE


@pytest.fixture(scope="module")
def feature_store(spark: SparkSession, tmp_path_factory) -> FeatureStore:
    """Build a synthetic feature store once and reuse it across tests."""
    store_path = tmp_path_factory.mktemp("fs")
    adapter = get_adapter("synthetic")
    build_features_batch(adapter, spark, store_path)
    return FeatureStore(path=store_path, spark=spark)


def _scores_by_item(df) -> dict[str, float]:
    """Materialize a score DataFrame into {item_id: score}."""
    return {row["item_id"]: row[SCORE] for row in df.collect()}


class TestPopularityRoundtrip:
    def test_scores_match_after_reload(
        self, spark: SparkSession, feature_store: FeatureStore, tmp_path: Path
    ) -> None:
        scorer = get_scorer("popularity")
        scorer.fit(feature_store, {})

        candidates = feature_store.item_features().select("item_id").distinct()
        before = _scores_by_item(scorer.score("u0", candidates))

        save_scorers({"popularity": scorer}, tmp_path / "models")
        loaded = load_scorers(tmp_path / "models", feature_store)

        assert "popularity" in loaded
        after = _scores_by_item(loaded["popularity"].score("u0", candidates))

        assert before.keys() == after.keys()
        for item_id in before:
            assert before[item_id] == pytest.approx(after[item_id])


class TestALSRoundtrip:
    def test_scores_match_after_reload(
        self, spark: SparkSession, feature_store: FeatureStore, tmp_path: Path
    ) -> None:
        scorer = get_scorer("als_cf", implicit=False)
        scorer.fit(feature_store, {"rank": 8, "reg": 0.1, "max_iter": 5})

        candidates = feature_store.item_features().select("item_id").distinct()
        before = _scores_by_item(scorer.score("u0", candidates))

        save_scorers({"als_cf": scorer}, tmp_path / "models")
        loaded = load_scorers(tmp_path / "models", feature_store)

        assert "als_cf" in loaded
        after = _scores_by_item(loaded["als_cf"].score("u0", candidates))

        assert before.keys() == after.keys()
        for item_id in before:
            # ALS predictions are deterministic given the same fitted factors
            assert before[item_id] == pytest.approx(after[item_id], rel=1e-4)


class TestSegmentsRoundtrip:
    def test_scores_match_after_reload(
        self, spark: SparkSession, feature_store: FeatureStore, tmp_path: Path
    ) -> None:
        scorer = get_scorer("segments")
        scorer.fit(feature_store, {"k": 4})

        candidates = feature_store.item_features().select("item_id").distinct()
        before = _scores_by_item(scorer.score("u0", candidates))

        save_scorers({"segments": scorer}, tmp_path / "models")
        loaded = load_scorers(tmp_path / "models", feature_store)

        assert "segments" in loaded
        after = _scores_by_item(loaded["segments"].score("u0", candidates))

        assert before.keys() == after.keys()
        for item_id in before:
            assert before[item_id] == pytest.approx(after[item_id])


class TestAtomicSwap:
    def test_target_dir_exists_after_overwrite(
        self, spark: SparkSession, feature_store: FeatureStore, tmp_path: Path
    ) -> None:
        """A second save over the same path should leave the target dir intact."""
        scorer = get_scorer("popularity")
        scorer.fit(feature_store, {})

        save_scorers({"popularity": scorer}, tmp_path / "models")
        save_scorers({"popularity": scorer}, tmp_path / "models")  # overwrite

        target = tmp_path / "models" / "popularity"
        assert target.exists()
        assert (target / "manifest.json").exists()

        # No .tmp cruft should remain
        assert not (tmp_path / "models" / ".popularity.tmp").exists()
