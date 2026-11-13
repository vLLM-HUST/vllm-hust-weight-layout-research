from __future__ import annotations

from dataclasses import dataclass

from ._version import __version__
from .registry import get_registry, register_builtins


@dataclass(frozen=True)
class OperatorOptimizationBundle:
    """Introspection carrier referenced by the Extension Bundle manifest."""

    extension_id: str = "org.vllm-hust.operator-optimizations"
    extension_version: str = __version__

    @staticmethod
    def available_optimizations() -> tuple[str, ...]:
        register_builtins()
        return get_registry().ids()
