from __future__ import annotations

import torch
import torch_npu
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config.compilation import Range
from vllm.logger import logger
from vllm_ascend.compilation.passes.base_pattern import BasePattern


class _LinearSwiGLUPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        return [
            torch.randn(2, 64, device="npu", dtype=self.dtype),
            torch.randn(256, 64, device="npu", dtype=self.dtype),
        ]

    def get_pattern(self):
        def pattern(x, gate_up_weight):
            return torch_npu.npu_swiglu(
                torch.ops.vllm.unquantized_gemm(x, gate_up_weight, None)
            )

        return pattern

    def get_replacement(self):
        def replacement(x, gate_up_weight):
            return torch.ops.vllm.unquantized_linear_swiglu(x, gate_up_weight)

        return replacement


class LinearSwiGLUFusionPass(VllmInductorPass):
    def __init__(self, vllm_config):
        super().__init__(vllm_config)
        self.patterns = PatternMatcherPass(pass_name="linear_swiglu_fusion_pass")
        if vllm_config.model_config.dtype in (torch.float16, torch.bfloat16):
            _LinearSwiGLUPattern(vllm_config).register(self.patterns)

    def __call__(self, graph: torch.fx.Graph) -> None:  # type: ignore[override]
        self.begin()
        self.matched_count = self.patterns.apply(graph)
        logger.info("Fused %s unquantized linear + SwiGLU patterns", self.matched_count)
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return True
