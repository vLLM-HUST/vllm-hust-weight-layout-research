from types import ModuleType

import pytest

from vllm_hust_operator_optimizations.optimizations.shared_expert_swiglu import install_shared_expert_fusion


def test_shared_expert_rejects_missing_class():
    with pytest.raises(RuntimeError, match="missing AscendFusedMoE"):
        install_shared_expert_fusion(ModuleType("unknown"))


def test_shared_expert_rejects_missing_split_methods():
    module = ModuleType("unknown")
    module.AscendFusedMoE = type("UnknownMoe", (), {})
    with pytest.raises(RuntimeError, match="missing shared-expert"):
        install_shared_expert_fusion(module)
