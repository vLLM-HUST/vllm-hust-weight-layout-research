"""Experimental shared-expert seam; preserves caller stream/event boundaries."""
import inspect
import torch
from candidate_swiglu_kernel import linear_swiglu


def install(module):
    cls = module.AscendFusedMoE
    if getattr(cls, '_opset_shared_probe', False):
        return
    first, second = cls._shared_experts_part1, cls._shared_experts_part2
    if 'gate_up_proj' not in inspect.getsource(first) or 'act_fn' not in inspect.getsource(second):
        raise RuntimeError('unsupported shared-expert seam')

    def part1(self, hidden_states):
        projection = self._shared_experts.gate_up_proj
        weight = projection.weight
        if (hidden_states.ndim == 2 and 0 < hidden_states.shape[0] <= 128
                and hidden_states.dtype in (torch.float16, torch.bfloat16)
                and weight.ndim == 2 and weight.dtype == hidden_states.dtype
                and weight.shape[1] == hidden_states.shape[1]
                and weight.shape[0] % 2 == 0
                and getattr(projection, 'bias', None) is None
                and type(projection.quant_method).__name__ == 'AscendUnquantizedLinearMethod'):
            if not getattr(self, '_opset_shared_logged', False):
                print('OPSET_SHARED_FUSION_HIT', tuple(hidden_states.shape), tuple(weight.shape), flush=True)
                self._opset_shared_logged = True
            return linear_swiglu(hidden_states, weight)
        return first(self, hidden_states)

    def part2(self, hidden_states, intermediate):
        # Half-width denotes fused activation; original part1 returns full width.
        expected = self._shared_experts.gate_up_proj.weight.shape[0]
        if intermediate.shape[-1] == expected:
            return second(self, hidden_states, intermediate)
        if intermediate.shape[-1] * 2 != expected:
            raise RuntimeError('unexpected shared-expert intermediate width')
        shared_out = self._shared_experts.down_proj(intermediate)[0]
        gate = getattr(self._shared_experts, 'expert_gate', None)
        if gate is not None:
            shared_out = torch.sigmoid(gate(hidden_states)[0]) * shared_out
        return shared_out

    cls._shared_experts_part1, cls._shared_experts_part2 = part1, part2
    cls._opset_shared_probe = True
