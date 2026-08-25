"""
Heuristics configuration loading and validation.

Parses and validates the user's heuristics YAML file using pydantic.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class SignalConfig(BaseModel):
    """Configuration for a single signal."""

    weight: float = Field(ge=0.0, le=1.0, description="Signal weight in [0, 1]")
    params: dict[str, Any] = Field(default_factory=dict)
    feature_weights: dict[str, float] = Field(default_factory=dict)


class FiltersConfig(BaseModel):
    """Filtering configuration."""

    exclude_seen: bool = True
    exclude_items: list[str] = Field(default_factory=list)
    require_features: dict[str, Any] = Field(default_factory=dict)


class HeuristicsConfig(BaseModel):
    """
    Complete heuristics configuration.

    Loaded from YAML and validated by pydantic.
    """

    domain: str = Field(description="Adapter name to load")
    top_k: int = Field(default=20, ge=1, description="Number of items to return")
    normalization: str = Field(default="minmax", pattern="^(minmax|zscore)$")
    signals: dict[str, SignalConfig] = Field(default_factory=dict)
    filters: FiltersConfig = Field(default_factory=FiltersConfig)

    @field_validator("signals", mode="before")
    @classmethod
    def parse_signals(cls, v: dict[str, Any]) -> dict[str, SignalConfig]:
        """Convert raw signal dicts to SignalConfig objects."""
        if not isinstance(v, dict):
            return v
        result = {}
        for name, cfg in v.items():
            if isinstance(cfg, SignalConfig):
                result[name] = cfg
            else:
                result[name] = SignalConfig(**cfg)
        return result

    def get_weights(self) -> dict[str, float]:
        """Get signal name to weight mapping."""
        return {name: cfg.weight for name, cfg in self.signals.items()}

    def get_active_signals(self) -> list[str]:
        """Get names of signals with weight > 0."""
        return [name for name, cfg in self.signals.items() if cfg.weight > 0]


def load_heuristics(path: Path | str) -> HeuristicsConfig:
    """
    Load and validate heuristics from a YAML file.

    Args:
        path: Path to the heuristics YAML file.

    Returns:
        Validated HeuristicsConfig.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        pydantic.ValidationError: If the config is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Heuristics file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    return HeuristicsConfig(**raw)
