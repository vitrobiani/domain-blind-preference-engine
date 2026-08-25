"""Adapter package for domain-specific data loading."""

from preference_engine.adapter.base import DomainAdapter
from preference_engine.adapter.registry import get_adapter, list_adapters

__all__ = ["DomainAdapter", "get_adapter", "list_adapters"]
