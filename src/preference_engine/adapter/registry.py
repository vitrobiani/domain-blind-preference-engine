"""
Adapter registry for discovering and loading domain adapters.

Adapters are discovered from the adapters/ directory by convention.
"""

import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from preference_engine.adapter.base import DomainAdapter


def _get_adapters_dir() -> Path:
    """Get the adapters directory path."""
    # Adapters live in project_root/adapters/
    # This file is at project_root/src/preference_engine/adapter/registry.py
    # __file__ -> adapter/ -> preference_engine/ -> src/ -> project_root/
    return Path(__file__).parent.parent.parent.parent / "adapters"


def list_adapters() -> list[str]:
    """
    List all available adapter names.

    Returns:
        List of adapter directory names that contain an adapter.py file.
    """
    adapters_dir = _get_adapters_dir()
    if not adapters_dir.exists():
        return []

    return [
        d.name
        for d in adapters_dir.iterdir()
        if d.is_dir() and (d / "adapter.py").exists()
    ]


def get_adapter(name: str) -> "DomainAdapter":
    """
    Load and instantiate an adapter by name.

    Args:
        name: Adapter name (directory name under adapters/).

    Returns:
        Instantiated DomainAdapter subclass.

    Raises:
        ValueError: If adapter not found or invalid.
    """
    adapters_dir = _get_adapters_dir()
    adapter_path = adapters_dir / name / "adapter.py"

    if not adapter_path.exists():
        available = list_adapters()
        raise ValueError(
            f"Adapter '{name}' not found. Available: {available}"
        )

    # Add adapters directory to path for import
    adapters_str = str(adapters_dir)
    if adapters_str not in sys.path:
        sys.path.insert(0, adapters_str)

    # Import the adapter module
    module = importlib.import_module(f"{name}.adapter")

    # Find the DomainAdapter subclass
    from preference_engine.adapter.base import DomainAdapter

    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, DomainAdapter)
            and attr is not DomainAdapter
        ):
            return attr()

    raise ValueError(
        f"No DomainAdapter subclass found in adapters/{name}/adapter.py"
    )
