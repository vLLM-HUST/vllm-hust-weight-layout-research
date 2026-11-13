import pytest

from vllm_hust_operator_optimizations.api import Optimization
from vllm_hust_operator_optimizations.optimizations import LinearSwiGLUGraph
from vllm_hust_operator_optimizations.registry import OptimizationRegistry


class FakeOptimization(Optimization):
    optimization_id = "fake"
    target_module = "fake.module"

    def apply(self, module):
        raise NotImplementedError


def test_registry_selects_known_optimizations():
    registry = OptimizationRegistry()
    optimization = FakeOptimization()
    registry.register(optimization)

    assert registry.select(["fake"]) == (optimization,)


def test_registry_rejects_unknown_optimization():
    registry = OptimizationRegistry()

    with pytest.raises(ValueError, match="unknown operator optimization"):
        registry.select(["missing"])


def test_linear_swiglu_graph_declares_both_runtime_seams():
    optimization = LinearSwiGLUGraph()

    assert optimization.optimization_id == "linear-swiglu-graph"
    assert optimization.target_modules == (
        "vllm_ascend.ops.linear",
        "vllm_ascend.compilation.graph_fusion_pass_manager",
        "vllm_ascend.ops.fused_moe.fused_moe_0_23_0",
    )
