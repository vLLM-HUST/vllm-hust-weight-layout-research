"""Compare shared model files against the benchmark's pinned model receipt."""
import hashlib
import json
from pathlib import Path
import time


root = Path(__file__).resolve().parent
reference = root / "sources/benchmark/reports/pyramidkv-five-dataset-adaptation-20261009/reproduction/model-hashes.json"
model = Path("/data/shared_upload/models/vspec/Qwen3.5-35B-A3B")
expected = json.loads(reference.read_text())
result = {"reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(), "expected_revision": expected["revision"], "files": []}
started = time.monotonic()
for entry in expected["files"]:
    name = entry["file"]
    if Path(name).name != name:
        raise ValueError("reference file must be a basename")
    digest = hashlib.sha256()
    with (model / name).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    result["files"].append({"file": name, "sha256": actual, "expected": entry["sha256"], "matched": actual == entry["sha256"]})
    print(name, "matched" if actual == entry["sha256"] else "MISMATCH", flush=True)
result["all_matched"] = all(e["matched"] for e in result["files"])
result["elapsed_seconds"] = time.monotonic() - started
(root / "results/model-receipt.json").write_text(json.dumps(result, indent=2) + "\n")
raise SystemExit(0 if result["all_matched"] else 1)
