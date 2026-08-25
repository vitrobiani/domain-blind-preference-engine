"""
Signal registry for discovering and instantiating scorers.

Provides factory functions for creating scorer instances by name.
"""

from typing import TYPE_CHECKING

from preference_engine.signals.popularity import PopularityScorer
from preference_engine.signals.trends import TrendsScorer
from preference_engine.signals.segments import SegmentsScorer
from preference_engine.signals.content import ContentScorer
from preference_engine.signals.als_cf import ALSScorer
from preference_engine.signals.rules import RulesScorer

if TYPE_CHECKING:
    from preference_engine.signals.base import Scorer


# Registry of all available scorers
SCORERS: dict[str, type["Scorer"]] = {
    "popularity": PopularityScorer,
    "trends": TrendsScorer,
    "segments": SegmentsScorer,
    "content": ContentScorer,
    "als_cf": ALSScorer,
    "rules": RulesScorer,
}


def list_signals() -> list[str]:
    """
    List all available signal names.

    Returns:
        List of signal names.
    """
    return list(SCORERS.keys())


def get_scorer(name: str, **kwargs) -> "Scorer":
    """
    Create a scorer instance by name.

    Args:
        name: Signal name (e.g., "popularity", "als_cf").
        **kwargs: Additional arguments passed to scorer constructor.

    Returns:
        Instantiated Scorer subclass.

    Raises:
        ValueError: If signal name not found.
    """
    if name not in SCORERS:
        available = list_signals()
        raise ValueError(f"Signal '{name}' not found. Available: {available}")

    return SCORERS[name](**kwargs)
