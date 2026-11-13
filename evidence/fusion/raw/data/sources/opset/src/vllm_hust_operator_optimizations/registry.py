from __future__ import annotations

from collections.abc import Iterable

from .api import Optimization


class OptimizationRegistry:
    def __init__(self) -> None:
        self._items: dict[str, Optimization] = {}

    def register(self, optimization: Optimization) -> None:
        key = optimization.optimization_id
        if key in self._items:
            raise ValueError(f"duplicate optimization id: {key}")
        self._items[key] = optimization

    def select(self, enabled: Iterable[str]) -> tuple[Optimization, ...]:
        requested = tuple(dict.fromkeys(enabled))
        unknown = sorted(set(requested) - self._items.keys())
        if unknown:
            raise ValueError(f"unknown operator optimization(s): {', '.join(unknown)}")
        return tuple(self._items[key] for key in requested)

    def ids(self) -> tuple[str, ...]:
        return tuple(self._items)


_REGISTRY = OptimizationRegistry()
_BUILTINS_REGISTERED = False


def register(optimization: Optimization) -> None:
    _REGISTRY.register(optimization)


def get_registry() -> OptimizationRegistry:
    return _REGISTRY


def register_builtins() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    from .optimizations import LinearSwiGLUGraph, PersistentMatmulEmpty

    register(PersistentMatmulEmpty())
    register(LinearSwiGLUGraph())
    _BUILTINS_REGISTERED = True
