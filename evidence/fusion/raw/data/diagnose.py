"""Run each optimization independently after the persistent-runtime B0 finishes."""
import json
from pathlib import Path
import subprocess
import sys
import time


root = Path(__file__).resolve().parent
baseline = root / "results/20261012T200600Z-B0"
deadline = time.monotonic() + 1300
while not (baseline / "result.json").exists():
    if time.monotonic() > deadline:
        raise TimeoutError("baseline did not complete")
    time.sleep(5)
base_result = json.loads((baseline / "result.json").read_text())
if base_result["status"] != "completed-smoke":
    raise RuntimeError("baseline startup failed; isolate runtime before enabling optimizations")
base = json.loads((baseline / "response.json").read_text())["choices"][0]["message"]["content"]
outcome = {"baseline": str(baseline.relative_to(root)), "baseline_excerpt": base[:160], "variants": []}
(root / "results/diagnosis.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2) + "\n")
if len(set(base)) < 5:
    raise RuntimeError("baseline output is degenerate; candidate tests skipped")
for optimization in ["linear-swiglu-graph", "persistent-matmul-empty"]:
    command = [sys.executable, str(root / "smoke.py"), "--arm", "B1", "--optimization", optimization]
    print("starting", optimization, flush=True)
    subprocess.run(command, timeout=1300, check=False)
    status = json.loads((root / "status.json").read_text())
    run = root / status["run"]
    item = {"optimization": optimization, "run": status["run"], "status": status["status"]}
    if (run / "response.json").exists():
        text = json.loads((run / "response.json").read_text())["choices"][0]["message"]["content"]
        item.update(text_equal=text == base, excerpt=text[:160], degenerate=len(set(text)) < 5)
    outcome["variants"].append(item)
    (root / "results/diagnosis.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(outcome, ensure_ascii=False), flush=True)
