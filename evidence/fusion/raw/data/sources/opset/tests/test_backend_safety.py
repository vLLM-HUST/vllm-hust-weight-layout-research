from types import ModuleType
from unittest.mock import patch

import pytest

from vllm_hust_operator_optimizations.optimizations.persistent_matmul_empty import PersistentMatmulEmpty


def test_known_failed_backend_preserves_original():
    module = ModuleType("test_matmul")
    module.linear_persistent = lambda x, y: None
    original = module.linear_persistent
    with patch(
        "vllm_hust_operator_optimizations.optimizations.persistent_matmul_empty.metadata.version",
        side_effect=lambda name: {"triton-ascend": "3.2.2", "torch-npu": "2.10.0.post4"}[name],
    ), pytest.warns(RuntimeWarning, match="retaining original"):
        result = PersistentMatmulEmpty().apply(module)
    assert result.state == "blocked"
    assert module.linear_persistent is original
