"""Record current runtime provenance and source-guard results without starting a model."""
import hashlib
import importlib
import importlib.metadata as metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys


root = Path(__file__).resolve().parent
result = {"evidence_level": "source-and-device-probe", "python": sys.version}
result["packages"] = {n: metadata.version(n) for n in ["torch", "torch-npu", "triton-ascend", "vllm", "vllm-ascend", "vllm-hust-opset"]}
result["sources"] = {}
for name, path in {"vllm": str(root / "sources/runtime-vllm"), "ascend": str(root / "sources/runtime-ascend"), "opset": str(root / "sources/opset")}.items():
    result["sources"][name] = {
        "commit": subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip(),
        "origin": subprocess.check_output(["git", "-C", path, "remote", "get-url", "origin"], text=True).strip(),
    }
import torch
import torch_npu
result["npu"] = {"available": torch.npu.is_available(), "count": torch.npu.device_count()}
result["device_roundtrip"] = []
result["active_import_paths"] = {n: importlib.import_module(n).__file__ for n in ["vllm", "vllm_ascend", "vllm_hust_operator_optimizations"]}
for index in range(torch.npu.device_count()):
    value = torch.arange(8, device=f"npu:{index}")
    torch.npu.synchronize(index)
    result["device_roundtrip"].append(torch.equal(value.cpu(), torch.arange(8)))
os.environ["VLLM_HUST_OPERATOR_OPTIMIZATIONS"] = "persistent-matmul-empty,linear-swiglu-graph"
try:
    from vllm_hust_operator_optimizations.plugin import activate
    activate()
    mm = importlib.import_module("vllm_ascend.ops.triton.batch_invariant.matmul")
    linear = importlib.import_module("vllm_ascend.ops.linear")
    manager = importlib.import_module("vllm_ascend.compilation.graph_fusion_pass_manager")
    source = inspect.getsource(mm.linear_persistent)
    result["opset_source_guard"] = {
        "empty_allocation": "empty" in mm.linear_persistent.__code__.co_names and "zeros" not in mm.linear_persistent.__code__.co_names,
        "fused_op_registered": hasattr(linear, "unquantized_linear_swiglu"),
        "fusion_pass_installed": bool(getattr(manager.GraphFusionPassManager, "_vllm_hust_linear_swiglu_installed", False)),
        "persistent_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }
except Exception as error:
    result["opset_source_guard"] = {"error": f"{type(error).__name__}: {error}"}
(root / "results/environment-probe.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
