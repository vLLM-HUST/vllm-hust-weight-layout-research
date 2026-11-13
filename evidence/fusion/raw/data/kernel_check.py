"""Reuse OPset's shape/tail regression and check the standalone fused kernel."""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy


root = Path(__file__).resolve().parent
os.environ["VLLM_HUST_OPERATOR_OPTIMIZATIONS"] = "persistent-matmul-empty"
os.environ.pop("VLLM_BATCH_INVARIANT", None)
from vllm_hust_operator_optimizations.plugin import activate
activate()
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    runpy.run_path(str(root / "sources/opset/experiments/persistent_matmul_empty/correctness.py"), run_name="__main__")
raw = buffer.getvalue()
(root / "results/persistent-correctness-raw.log").write_text(raw)
persistent = next(json.loads(line) for line in reversed(raw.splitlines()) if line.startswith('{"timestamp":'))
import torch
import torch_npu
from vllm_hust_operator_optimizations.optimizations.linear_swiglu_kernel import linear_swiglu

fused = []
for dtype in [torch.float16, torch.bfloat16]:
    for m, n, k in [(1, 512, 2048), (32, 512, 2048), (7, 513, 257), (128, 513, 257)]:
        torch.manual_seed(42)
        x = (torch.randn((m, k)) * 0.2).to(device="npu", dtype=dtype)
        weight = (torch.randn((2 * n, k)) * 0.2).to(device="npu", dtype=dtype)
        reference = torch_npu.npu_swiglu(torch.nn.functional.linear(x, weight))
        values = [linear_swiglu(x, weight) for _ in range(3)]
        torch.npu.synchronize()
        for value in values:
            assert bool(torch.isfinite(value).all())
            torch.testing.assert_close(value.float().cpu(), reference.float().cpu(), rtol=0.03 if dtype == torch.bfloat16 else 0.005, atol=0.05 if dtype == torch.bfloat16 else 0.02)
        for value in values[1:]:
            torch.testing.assert_close(value, values[0], rtol=0, atol=0)
        fused.append({"dtype": str(dtype), "m": m, "n": n, "k": k, "launches": 3, "passed": True})
result = {"persistent": persistent, "fused": {"cases": len(fused), "launches": len(fused) * 3, "results": fused}, "evidence_level": "operator-correctness-only"}
(root / "results/kernel-correctness.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"persistent_cases": persistent["cases"], "persistent_launches": persistent["launches"], "fused_cases": len(fused), "fused_launches": len(fused) * 3}))
