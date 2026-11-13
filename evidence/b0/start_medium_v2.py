"""One-off safe restart after the user's explicit 64-request revision."""
import json, os, subprocess, time
from pathlib import Path
ROOT=Path('/root/opset-research/b0-weight-layout-20261023')
PY='/root/opset-research/.venv/bin/python'
for pid in [949504,950848]:
    if Path(f'/proc/{pid}').exists():raise RuntimeError(f'previous controller still alive: {pid}')
archive=ROOT/'medium52-initialization-attempt';archive.mkdir(exist_ok=True)
for rel in ['servers/medium-r1-ctx4096-seq64-budget8192','medium-protocol.json','medium-reuse-map.json','cleanup.complete.json']:
    path=ROOT/rel
    if path.exists():path.rename(archive/path.name)
if list((ROOT/'cells').glob('medium-r*/summary.json')):
    raise RuntimeError('unexpected measured Medium52 cell; review before restarting')
env=os.environ.copy();env['MEDIUM_DATA_BASE']='/root/opset-research/datasets/mmlu-weight-layout-v1'
subprocess.run([PY,str(ROOT/'build_medium.py')],env=env,check=True)
with (ROOT/'medium-campaign.log').open('ab',buffering=0) as out:
    p=subprocess.Popen(['bash',str(ROOT/'run_medium_campaign.sh')],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
(ROOT/'medium-controller.json').write_text(json.dumps({'pid':p.pid,'started_epoch':time.time(),'revision':'medium-v2','performance_requests_per_bucket':64}))
print('Started Medium64 controller',p.pid,flush=True)
