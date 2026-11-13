"""Five-second server counters/gauges; no model or NPU operations."""
import datetime
import fcntl
import json
from pathlib import Path
import time
import urllib.request

ROOT=Path('/root/opset-research/b0-weight-layout-20261023')
lock=(ROOT/'metrics-monitor.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
keys=('vllm:num_requests_running','vllm:num_requests_waiting','vllm:kv_cache_usage_perc',
      'vllm:num_preemptions','vllm:generation_tokens','vllm:prompt_tokens')
with (ROOT/'metrics-samples.jsonl').open('a') as f:
    while not (ROOT/'cleanup.complete.json').exists():
        start=time.monotonic()
        try:
            state=json.loads((ROOT/'status.json').read_text())
            with urllib.request.urlopen('http://127.0.0.1:18123/metrics',timeout=3) as response:
                lines=response.read().decode().splitlines()
            values={}
            for line in lines:
                if line.startswith(keys):
                    k,v=line.rsplit(' ',1)
                    try:values[k]=float(v)
                    except ValueError:pass
            f.write(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                'stage':state,'metrics':values})+'\n');f.flush()
        except Exception:pass
        time.sleep(max(0,5-(time.monotonic()-start)))
