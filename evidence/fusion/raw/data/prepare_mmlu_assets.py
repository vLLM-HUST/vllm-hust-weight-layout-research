"""Fetch only the immutable MMLU-Pro source files and verify canonical hashes."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request


root = Path(__file__).resolve().parent
assets = root / "datasets/mmlu-pro"
assets.mkdir(parents=True, exist_ok=True)
from vllm_hust_benchmark.mmlu_pro_pyramidkv import DATASET_REVISION, DATA_HASHES
import pyarrow.parquet as pq

records = []
for split, expected in DATA_HASHES.items():
    name = f"{split}-00000-of-00001.parquet"
    path = assets / name
    data = None
    for host in ["huggingface.co", "hf-mirror.com"]:
        url = f"https://{host}/datasets/TIGER-Lab/MMLU-Pro/resolve/{DATASET_REVISION}/data/{name}"
        try:
            with urllib.request.urlopen(url, timeout=25) as response:
                candidate = response.read()
            if hashlib.sha256(candidate).hexdigest() != expected:
                raise ValueError("canonical SHA-256 mismatch")
            data = candidate
            break
        except Exception as error:
            print(host, type(error).__name__, flush=True)
    if data is None:
        raise RuntimeError(f"unable to fetch pinned {split} asset")
    path.write_bytes(data)
    records.append({"file": name, "sha256": expected, "bytes": len(data), "rows": pq.read_metadata(path).num_rows})
    print(split, records[-1], flush=True)
(assets / "receipt.json").write_text(json.dumps({"revision": DATASET_REVISION, "files": records, "evidence_level": "source-assets-only", "quality_runs": 0}, indent=2) + "\n")
