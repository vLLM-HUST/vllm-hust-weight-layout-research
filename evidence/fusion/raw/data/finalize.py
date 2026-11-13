"""Produce a deployment receipt and portable checksum inventory."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess


root = Path(__file__).resolve().parent
results = root / "results"
diagnosis = json.loads((results / "diagnosis.json").read_text())
kernel = json.loads((results / "kernel-correctness.json").read_text())
model = json.loads((results / "model-receipt.json").read_text())
npu = subprocess.check_output(["npu-smi", "info"], text=True)
(results / "npu-after.txt").write_text(npu)
processes = subprocess.check_output(["ps", "-eo", "pid,pgid,args"], text=True)
owned_services = [line for line in processes.splitlines() if "VLLM::" in line or "vllm.entrypoints.openai.api_server" in line]
summary = {
    "evidence_level": "development-environment-and-correctness",
    "formal_task_quality_runs": 0,
    "baseline_research_ready": True,
    "formal_evaluation_ready": False,
    "packages": {n: metadata.version(n) for n in ["torch", "torch-npu", "triton-ascend", "vllm", "vllm-ascend", "vllm-hust-opset", "vllm-hust-benchmark", "pyarrow"]},
    "hardware": {"type": "Ascend910B2", "physical_npus": [5, 6], "logical_npus": [0, 1], "tensor_parallel_size": 2},
    "runtime": {"cann": "9.1.0", "precision": "bfloat16", "source_profile": "server-version upstream revisions checked out in HUST forks; current HUST main is staged but inactive"},
    "safe_default": {"enabled_optimizations": [], "launcher": "serve_baseline.sh"},
    "standalone_correctness": {"persistent_cases": kernel["persistent"]["cases"], "persistent_launches": kernel["persistent"]["launches"], "fused_cases": kernel["fused"]["cases"], "fused_launches": kernel["fused"]["launches"]},
    "model_weights": {"expected_revision": model["expected_revision"], "shards_verified": len(model["files"]), "all_matched": model["all_matched"]},
    "graph_diagnosis": diagnosis,
    "optimization_gates": {"persistent-matmul-empty": "blocked: reproducible degenerate model Graph output despite standalone correctness passing", "linear-swiglu-graph": "not-exercised: 0 fusion matches; smoke output equals baseline"},
    "cleanup": {"services_remaining": owned_services, "npu_processes_absent": "No running processes found" in npu and not owned_services},
}
save = lambda p, obj: p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
save(results / "deployment-summary.json", summary)
matrix = json.loads((root / "artifacts/applicability-draft.json").read_text())
matrix["status"] = "blocked-for-formal-evaluation"
matrix["observations"] = summary["optimization_gates"]
matrix["common_blockers"][0] = "The persistent-matmul-empty treatment fails the Qwen3.5 TP2 Graph model correctness gate. Linear-SwiGLU has 0 observed fusion matches in the current text Graph."
matrix["common_blockers"][1] = "Model weight shards are verified; task/evaluator/run contracts still require freezing before matched quality evaluation."
save(results / "applicability.json", matrix)
lines = []
for path in sorted(results.rglob("*")):
    if path.is_file() and path.name != "SHA256SUMS":
        lines.append(hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.relative_to(results).as_posix())
(results / "SHA256SUMS").write_text("\n".join(lines) + "\n")
print(json.dumps({"baseline_research_ready": True, "formal_evaluation_ready": False, "cleanup": summary["cleanup"], "receipt": "results/deployment-summary.json"}))
