from __future__ import annotations

import inspect
import ast
import textwrap
from types import ModuleType

from ..api import OptimizationResult


def install_shared_expert_fusion(module: ModuleType) -> OptimizationResult:
    """Fuse inside the opaque MoE seam without moving stream/event boundaries."""
    cls = getattr(module, "AscendFusedMoE", None)
    if cls is None:
        raise RuntimeError("missing AscendFusedMoE shared-expert seam")
    if getattr(cls, "_opset_shared_swiglu_installed", False):
        return OptimizationResult("linear-swiglu-graph", "already-applied", "shared seam installed")
    first = getattr(cls, "_shared_experts_part1", None)
    second = getattr(cls, "_shared_experts_part2", None)
    if not callable(first) or not callable(second):
        raise RuntimeError("missing shared-expert split methods")
    first_source, second_source = inspect.getsource(first), inspect.getsource(second)
    first_body = ast.parse(textwrap.dedent(first_source)).body[0].body
    second_body = ast.parse(textwrap.dedent(second_source)).body[0].body
    if (
        "self._shared_experts.gate_up_proj(hidden_states)" not in first_source
        or "self._shared_experts.act_fn(shared_gate_up)" not in second_source
        or "self._shared_experts.down_proj(shared_act)" not in second_source
        or len(first_body) != 2
        or len(second_body) != 5
    ):
        raise RuntimeError("unsupported shared-expert implementation")

    import torch
    import vllm.envs as envs
    from vllm_ascend.utils import get_weight_prefetch_method

    from .linear_swiglu_kernel import linear_swiglu

    def eligible(self, x):
        shared = self._shared_experts
        projection = shared.gate_up_proj
        weight = projection.weight
        config = getattr(self, "vllm_config", None)
        prefetch = get_weight_prefetch_method()
        prefetch_enabled = any(
            getattr(getattr(prefetch, field, None), "enable", False)
            for field in ("mlp", "moe", "attn")
        )
        return (
            envs.VLLM_BATCH_INVARIANT
            and config is not None
            and getattr(config, "lora_config", None) is None
            and not prefetch_enabled
            and x.device.type == "npu"
            and x.ndim == 2
            and 0 < x.shape[0] <= 128
            and x.dtype in (torch.float16, torch.bfloat16)
            and weight.ndim == 2
            and weight.dtype == x.dtype
            and weight.device == x.device
            and weight.shape[1] == x.shape[1]
            and weight.shape[0] % 2 == 0
            and getattr(projection, "bias", None) is None
            and type(projection.quant_method).__name__ == "AscendUnquantizedLinearMethod"
            and type(shared.act_fn).__name__ == "AscendSiluAndMul"
        )

    def part1(self, hidden_states):
        if not eligible(self, hidden_states):
            return first(self, hidden_states)
        if not getattr(self, "_opset_shared_swiglu_logged", False):
            from vllm.logger import logger
            logger.info("OPset shared-expert Linear-SwiGLU fusion exercised")
            self._opset_shared_swiglu_logged = True
        return linear_swiglu(hidden_states, self._shared_experts.gate_up_proj.weight)

    def part2(self, hidden_states, intermediate):
        if not eligible(self, hidden_states):
            return second(self, hidden_states, intermediate)
        expected = self._shared_experts.gate_up_proj.weight.shape[0]
        if intermediate.shape[-1] == expected:
            return second(self, hidden_states, intermediate)
        if intermediate.shape[-1] * 2 != expected:
            raise RuntimeError("unexpected shared-expert intermediate width")
        output = self._shared_experts.down_proj(intermediate)[0]
        gate = getattr(self._shared_experts, "expert_gate", None)
        if gate is not None:
            output = torch.sigmoid(gate(hidden_states)[0]) * output
        return output

    cls._shared_experts_part1, cls._shared_experts_part2 = part1, part2
    cls._opset_shared_swiglu_installed = True
    return OptimizationResult("linear-swiglu-graph", "applied", "registered guarded opaque shared-expert fusion")
