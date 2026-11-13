from __future__ import annotations

import ast
import inspect
import importlib.metadata as metadata
import warnings
from types import ModuleType

from ..api import Optimization, OptimizationResult
from ..rewrite import replace_single_call


class PersistentMatmulEmpty(Optimization):
    optimization_id = "persistent-matmul-empty"
    target_module = "vllm_ascend.ops.triton.batch_invariant.matmul"
    function_name = "linear_persistent"

    @staticmethod
    def _validate(function: ast.FunctionDef) -> None:
        stores = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Subscript)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "linear_persistent_kernel"
        ]
        if len(stores) != 1:
            raise RuntimeError("linear_persistent must launch exactly one persistent kernel")

    def apply(self, module: ModuleType) -> OptimizationResult:
        try:
            failed_pair = (
                metadata.version("triton-ascend") == "3.2.2"
                and metadata.version("torch-npu") == "2.10.0.post4"
            )
        except metadata.PackageNotFoundError:
            failed_pair = False
        if failed_pair:
            detail = (
                "disabled: this tested backend combination failed poisoned-output "
                "and model correctness gates; retaining original torch.zeros"
            )
            warnings.warn(detail, RuntimeWarning, stacklevel=2)
            return OptimizationResult(self.optimization_id, "blocked", detail)
        function = getattr(module, self.function_name, None)
        if function is None:
            raise RuntimeError(f"missing {module.__name__}.{self.function_name}")
        source = inspect.getsource(function)
        if "torch.empty((M, N)" in source and "torch.zeros((M, N)" not in source:
            return OptimizationResult(
                self.optimization_id,
                "already-applied",
                "target source already allocates the fully-written output with torch.empty",
            )

        replace_single_call(
            module,
            self.function_name,
            old_owner="torch",
            old_name="zeros",
            new_name="empty",
            validator=self._validate,
        )
        return OptimizationResult(
            self.optimization_id,
            "applied",
            "replaced the redundant output zero initialization in linear_persistent",
        )
