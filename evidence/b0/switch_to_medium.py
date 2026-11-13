"""Finish the in-flight Large cell, then stop its owner and launch revised queue."""
import json, os, subprocess, time
from pathlib import Path

ROOT=Path('/root/opset-research/b0-weight-layout-20261023')
TARGET=ROOT/'cells/large-r1-P4-C64/summary.json'
OLD=57591
PY='/root/opset-research/.venv/bin/python'
def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False

while alive(OLD) and not TARGET.exists():time.sleep(5)
if alive(OLD):
    command=Path(f'/proc/{OLD}/cmdline').read_bytes()
    if b'run_b0.py' not in command:raise RuntimeError('old owner PID changed')
    os.kill(OLD,15)
for _ in range(100):
    if not alive(OLD) and not alive(56838):break
    time.sleep(2)
else:raise RuntimeError('old controller has not finished cleanup; no overlapping launch')
marker=ROOT/'cleanup.complete.json'
if marker.exists():marker.rename(ROOT/'large-controller-stop-cleanup.json')
with (ROOT/'medium-campaign.log').open('ab',buffering=0) as out:
    process=subprocess.Popen(['bash',str(ROOT/'run_medium_campaign.sh')],stdout=out,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT)
(ROOT/'medium-controller.json').write_text(json.dumps({'pid':process.pid,'switched_epoch':time.time(),'last_large_cell':str(TARGET)}))
