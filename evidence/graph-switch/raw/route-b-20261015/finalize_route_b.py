"""Package immutable relative-path receipts for the task-group probe."""
import hashlib
import importlib.metadata as md
import json
import shutil
import subprocess
from pathlib import Path

root=Path(__file__).resolve().parent
out=root/'results/route-b-20261015'
for name in ['route_b_probe.py','candidate_swiglu_kernel.py','env.sh','finalize_route_b.py']:
    shutil.copy2(root/name,out/name)
def command(args):
    p=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    return dict(exit_code=p.returncode,output=p.stdout)
environment=dict(
    versions={n:md.version(n) for n in ['torch','torch-npu','triton-ascend','vllm','vllm-ascend']},
    runtime_commits={n:command(['git','-C',str(root/'sources'/n),'rev-parse','HEAD'])
                     for n in ['runtime-ascend','runtime-vllm']},
    device=command(['npu-smi','info']),
    processes=command(['ps','-eo','pid,ppid,args']),
    scope='single logical NPU 0, no service; no TP or model-level qualification')
(out/'environment.json').write_text(json.dumps(environment,indent=2))
summary={}
for name in ['control','kernel','topology','fusion','fusion-v2','triton']:
    p=out/(name+'.json')
    if p.exists():
        r=json.loads(p.read_text())
        summary[name]={k:r.get(k) for k in ['stage','all_pass','capture_count','shape','dtype']}
        summary[name]['successful_checks']=sum(c['finite'] and c['close'] and c['suffix'] for c in r['checks'])
(out/'summary.json').write_text(json.dumps(summary,indent=2))
files=[]
for p in sorted(out.iterdir()):
    if p.is_file() and p.name!='manifest.json':
        files.append(dict(path=p.name,size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(out/'manifest.json').write_text(json.dumps(dict(files=files),indent=2))
print(json.dumps(summary,indent=2))
