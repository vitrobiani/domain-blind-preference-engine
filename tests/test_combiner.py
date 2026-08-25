"""Tests for combiner and heuristics (Phase 5 placeholder)."""

import pytest
from pathlib import Path

from preference_engine.combiner.heuristics import (
    HeuristicsConfig,
    SignalConfig,
    FiltersConfig,
    load_heuristics,
)


class TestHeuristicsConfig:
    """Tests for heuristics configuration loading."""

    def test_signal_config_defaults(self) -> None:
        """Test SignalConfig default values."""
        config = SignalConfig(weight=0.5)
        assert config.weight == 0.5
        assert config.params == {}
        assert config.feature_weights == {}

    def test_filters_config_defaults(self) -> None:
        """Test FiltersConfig default values."""
        config = FiltersConfig()
        assert config.exclude_seen is True
        assert config.exclude_items == []
        assert config.require_features == {}

    def test_heuristics_from_dict(self) -> None:
        """Test HeuristicsConfig construction from dict."""
        raw = {
            "domain": "synthetic",
            "top_k": 10,
            "normalization": "minmax",
            "signals": {
                "popularity": {"weight": 0.3},
                "als_cf": {"weight": 0.7, "params": {"rank": 16}},
            },
            "filters": {"exclude_seen": False},
        }
        config = HeuristicsConfig(**raw)

        assert config.domain == "synthetic"
        assert config.top_k == 10
        assert config.signals["popularity"].weight == 0.3
        assert config.signals["als_cf"].params["rank"] == 16
        assert config.filters.exclude_seen is False

    def test_get_weights(self) -> None:
        """Test getting signal weights."""
        config = HeuristicsConfig(
            domain="test",
            signals={
                "a": SignalConfig(weight=0.3),
                "b": SignalConfig(weight=0.7),
            },
        )
        weights = config.get_weights()
        assert weights == {"a": 0.3, "b": 0.7}

    def test_get_active_signals(self) -> None:
        """Test getting active (non-zero weight) signals."""
        config = HeuristicsConfig(
            domain="test",
            signals={
                "a": SignalConfig(weight=0.3),
                "b": SignalConfig(weight=0.0),
                "c": SignalConfig(weight=0.5),
            },
        )
        active = config.get_active_signals()
        assert set(active) == {"a", "c"}


class TestLoadHeuristics:
    """Tests for loading heuristics from YAML."""

    def test_load_example_config(self) -> None:
        """Test loading the example heuristics file."""
        path = Path("config/heuristics.example.yaml")
        if not path.exists():
            pytest.skip("Example config not found")

        config = load_heuristics(path)
        assert config.domain == "synthetic"
        assert config.top_k == 20

    def test_load_nonexistent_raises(self) -> None:
        """Test that loading nonexistent file raises."""
        with pytest.raises(FileNotFoundError):
            load_heuristics("nonexistent.yaml")


# Placeholder tests for combiner (Phase 5)


class TestCombiner:
    """Tests for score combination."""

    @pytest.mark.skip(reason="Phase 5 - not yet implemented")
    def test_combine_weighted_sum(self) -> None:
        pass

    @pytest.mark.skip(reason="Phase 5 - not yet implemented")
    def test_combine_missing_signal(self) -> None:
        pass

    @pytest.mark.skip(reason="Phase 5 - not yet implemented")
    def test_apply_filters_exclude_seen(self) -> None:
        pass
