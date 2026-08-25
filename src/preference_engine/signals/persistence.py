"""
Top-level persistence helpers for scorers.

Provides save/load of the full active-scorer set with atomic swap
so a concurrent reader never sees a half-written model directory.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from preference_engine.signals.registry import SCORERS

if TYPE_CHECKING:
    from preference_engine.signals.base import Scorer
    from preference_engine.streaming.feature_store import FeatureStore


def save_scorers(scorers: dict[str, "Scorer"], root: Path) -> None:
    """
    Persist all fitted scorers under `root/<name>/`.

    Writes each scorer to a `.<name>.tmp/` sibling first, then swaps
    it in with an atomic rename. This keeps `root/<name>/` valid at
    all times for any concurrent reader (e.g. a serving process).
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    for name, scorer in scorers.items():
        target = root / name
        tmp = root / f".{name}.tmp"

        if tmp.exists():
            shutil.rmtree(tmp)

        scorer.save(tmp)

        if target.exists():
            shutil.rmtree(target)
        tmp.rename(target)


def load_scorers(
    root: Path,
    ctx: "FeatureStore",
) -> dict[str, "Scorer"]:
    """
    Load every scorer directory found under `root/`.

    Reads each subdirectory's manifest.json to determine which
    Scorer subclass to instantiate, then delegates to its load().
    Skips hidden dirs (e.g. in-progress `.name.tmp` writes) and
    dirs without a manifest.
    """
    root = Path(root)
    if not root.exists():
        return {}

    scorers: dict[str, "Scorer"] = {}
    for signal_dir in sorted(root.iterdir()):
        if not signal_dir.is_dir() or signal_dir.name.startswith("."):
            continue

        manifest_path = signal_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        manifest = json.loads(manifest_path.read_text())
        signal_name = manifest["signal"]
        cls = SCORERS[signal_name]
        scorers[signal_name] = cls.load(signal_dir, ctx)

    return scorers
