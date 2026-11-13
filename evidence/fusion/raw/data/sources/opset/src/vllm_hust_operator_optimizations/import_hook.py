from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import logging
import sys
from types import ModuleType

from .api import Optimization

logger = logging.getLogger(__name__)


class _OptimizingLoader(importlib.abc.Loader):
    def __init__(self, loader: importlib.abc.Loader, optimizations: tuple[Optimization, ...]) -> None:
        self._loader = loader
        self._optimizations = optimizations

    def create_module(self, spec):
        create = getattr(self._loader, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module: ModuleType) -> None:
        self._loader.exec_module(module)
        for optimization in self._optimizations:
            result = optimization.apply(module)
            logger.info(
                "Operator optimization %s: %s (%s)",
                result.optimization_id,
                result.state,
                result.detail,
            )


class OptimizationImportHook(importlib.abc.MetaPathFinder):
    def __init__(self, optimizations: tuple[Optimization, ...]) -> None:
        self._by_module: dict[str, list[Optimization]] = {}
        for optimization in optimizations:
            for target_module in optimization.target_modules:
                self._by_module.setdefault(target_module, []).append(optimization)

    def find_spec(self, fullname: str, path, target=None):
        selected = self._by_module.get(fullname)
        if not selected:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _OptimizingLoader(spec.loader, tuple(selected))
        return spec


def install(optimizations: tuple[Optimization, ...]) -> None:
    pending: list[Optimization] = []
    for optimization in optimizations:
        for target_module in optimization.target_modules:
            loaded = sys.modules.get(target_module)
            if loaded is None:
                continue
            result = optimization.apply(loaded)
            logger.info(
                "Operator optimization %s on %s: %s (%s)",
                result.optimization_id,
                target_module,
                result.state,
                result.detail,
            )
        if any(name not in sys.modules for name in optimization.target_modules):
            pending.append(optimization)
    if pending:
        sys.meta_path.insert(0, OptimizationImportHook(tuple(pending)))
