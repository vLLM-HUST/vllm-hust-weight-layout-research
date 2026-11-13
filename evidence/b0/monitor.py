"""Low-rate NPU HBM sampling, labelled with campaign stage/cell."""
import datetime
import json
from pathlib import Path
import subprocess
import time

ROOT=Path('/root/opset-research/b0-weight-layout-20261023')
with (ROOT/'npu-samples.jsonl').open('a') as f:
    while not (ROOT/'cleanup.complete.json').exists():
        started=time.monotonic()
        try:
            state=json.loads((ROOT/'status.json').read_text())
            r=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=20)
            record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'stage':state,'code':r.returncode,'output':r.stdout,
                    'sample_duration_seconds':time.monotonic()-started}
            f.write(json.dumps(record)+'\n');f.flush()
        except Exception as e:
            f.write(json.dumps({'error':str(e)})+'\n');f.flush()
        time.sleep(max(0,5-(time.monotonic()-started)))
