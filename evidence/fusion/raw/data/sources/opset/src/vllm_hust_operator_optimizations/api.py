from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import ModuleType


@dataclass(frozen=True)
class OptimizationResult:
    optimization_id: str
    state: str
    detail: str


class Optimization(ABC):
    """One independently gated operator optimization."""

    optimization_id: str
    target_module: str

    @property
    def target_modules(self) -> tuple[str, ...]:
        """Modules that must each receive this optimization."""
        return (self.target_module,)

    @abstractmethod
    def apply(self, module: ModuleType) -> OptimizationResult:
        """Apply the optimization or reject an incompatible target."""
