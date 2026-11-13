from __future__ import annotations

import inspect
from types import ModuleType

from ..api import Optimization, OptimizationResult


class LinearSwiGLUGraph(Optimization):
    """Install the gated unquantized GEMM + SwiGLU Graph fusion."""

    optimization_id = "linear-swiglu-graph"
    target_module = "vllm_ascend.ops.linear"
    _manager_module = "vllm_ascend.compilation.graph_fusion_pass_manager"
    _moe_module = "vllm_ascend.ops.fused_moe.fused_moe_0_23_0"

    @property
    def target_modules(self) -> tuple[str, ...]:
        return (self.target_module, self._manager_module, self._moe_module)

    @staticmethod
    def _install_custom_op(module: ModuleType) -> OptimizationResult:
        if getattr(module, "_vllm_hust_linear_swiglu_installed", False):
            return OptimizationResult(
                "linear-swiglu-graph", "already-applied", "custom op already registered"
            )
        if not hasattr(module, "unquantized_gemm"):
            raise RuntimeError("missing vllm_ascend.ops.linear.unquantized_gemm")
        if hasattr(module, "unquantized_linear_swiglu"):
            raise RuntimeError("upstream already defines unquantized_linear_swiglu")

        import torch
        from vllm.utils.torch_utils import direct_register_custom_op

        def unquantized_linear_swiglu(
            x: torch.Tensor,
            gate_up_weight: torch.Tensor,
        ) -> torch.Tensor:
            input_shape = x.shape
            flat_x = x.reshape(-1, input_shape[-1])
            if flat_x.shape[0] <= 128:
                from .linear_swiglu_kernel import linear_swiglu

                output = linear_swiglu(flat_x, gate_up_weight)
            else:
                import torch_npu

                output = torch_npu.npu_swiglu(
                    torch.nn.functional.linear(flat_x, gate_up_weight)
                )
            return output.reshape(*input_shape[:-1], gate_up_weight.shape[0] // 2)

        def unquantized_linear_swiglu_fake(
            x: torch.Tensor,
            gate_up_weight: torch.Tensor,
        ) -> torch.Tensor:
            output_shape = (*x.shape[:-1], gate_up_weight.shape[0] // 2)
            return torch.empty(output_shape, dtype=x.dtype, device=x.device)

        # ``from __future__ import annotations`` keeps package import light,
        # while torch's schema inference requires concrete runtime types.
        concrete_annotations = {
            "x": torch.Tensor,
            "gate_up_weight": torch.Tensor,
            "return": torch.Tensor,
        }
        unquantized_linear_swiglu.__annotations__ = concrete_annotations
        unquantized_linear_swiglu_fake.__annotations__ = concrete_annotations

        direct_register_custom_op(
            op_name="unquantized_linear_swiglu",
            op_func=unquantized_linear_swiglu,
            fake_impl=unquantized_linear_swiglu_fake,
            mutates_args=[],
            dispatch_key="PrivateUse1",
        )
        module.unquantized_linear_swiglu = unquantized_linear_swiglu
        module.unquantized_linear_swiglu_fake = unquantized_linear_swiglu_fake
        module._vllm_hust_linear_swiglu_installed = True
        return OptimizationResult(
            "linear-swiglu-graph", "applied", "registered gated fused linear-SwiGLU op"
        )

    @staticmethod
    def _install_graph_pass(module: ModuleType) -> OptimizationResult:
        manager = getattr(module, "GraphFusionPassManager", None)
        if manager is None:
            raise RuntimeError("missing GraphFusionPassManager")
        if getattr(manager, "_vllm_hust_linear_swiglu_installed", False):
            return OptimizationResult(
                "linear-swiglu-graph", "already-applied", "fusion pass already installed"
            )
        source = inspect.getsource(manager.configure)
        if "AddRMSNormQuantFusionPass" not in source or "MulsAddFusionPass" not in source:
            raise RuntimeError("unsupported GraphFusionPassManager.configure structure")
        if "LinearSwiGLUFusionPass" in source:
            raise RuntimeError("upstream already configures LinearSwiGLUFusionPass")

        original = manager.configure

        def configure(self, config):
            original(self, config)
            from vllm_ascend.utils import is_310p

            if not is_310p():
                from .linear_swiglu_fusion_pass import LinearSwiGLUFusionPass

                self.passes.append(LinearSwiGLUFusionPass(config))

        configure.__name__ = original.__name__
        configure.__qualname__ = original.__qualname__
        configure.__doc__ = original.__doc__
        configure.__wrapped__ = original
        manager.configure = configure
        manager._vllm_hust_linear_swiglu_installed = True
        return OptimizationResult(
            "linear-swiglu-graph", "applied", "attached linear-SwiGLU Graph fusion pass"
        )

    def apply(self, module: ModuleType) -> OptimizationResult:
        if module.__name__ == self._moe_module:
            from .shared_expert_swiglu import install_shared_expert_fusion
            return install_shared_expert_fusion(module)
        if module.__name__ == self.target_module:
            return self._install_custom_op(module)
        if module.__name__ == self._manager_module:
            return self._install_graph_pass(module)
        raise RuntimeError(f"unexpected target module: {module.__name__}")
