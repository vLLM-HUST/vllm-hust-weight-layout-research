"""Portable relative-path index; raw evidence remains immutable."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

root=Path(__file__).resolve().parent
suite=json.loads(root.joinpath('matched-suite.json').read_text())
summary=json.loads(root.joinpath('results/matched-summary.json').read_text())
agreements=[]
for iteration in range(1,4):
    pair={s['variant']:s for s in suite['runs'] if s['round']==iteration}
    b,c=[root/'results'/pair[v]['run_id'] for v in ('baseline','fusion')]
    qb,qc=[json.loads(p.joinpath('quality.json').read_text()) for p in (b,c)]
    agreements.append(dict(round=iteration,smoke_equal=pair['baseline']['smoke_text']==pair['fusion']['smoke_text'],predictions_equal=sum(x['predicted']==y['predicted'] for x,y in zip(qb,qc)),questions=len(qb),requests_equal=all(x['request']==y['request'] for x,y in zip(qb,qc))))
paired=[]
for workload in ('short','long'):
    values=[]
    for iteration in range(1,4):
        rows=[r for r in summary['lifecycle'] if r['round']==iteration and r['concurrency']==1 and r['workload']==workload and r['metric']=='tpot_ms']
        d={r['variant']:r['mean'] for r in rows}; values.append(d['baseline']-d['fusion'])
    # Exact Student-t(.975, df=2) quantile from its analytic CDF.
    critical=math.sqrt(2*.95**2/(1-.95**2))
    mean=statistics.mean(values); half=critical*statistics.stdev(values)/math.sqrt(3)
    paired.append(dict(workload=workload,paired_tpot_reduction_ms=values,mean_ms=mean,ci95_ms=[mean-half,mean+half],interpretation='three-lifecycle descriptive CI; not a broad certification'))
npu=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True).stdout
root.joinpath('results/final-npu-state.txt').write_text(npu)
processes=subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True,check=True).stdout
remaining=[line for line in processes.splitlines() if any(word in line for word in ('vllm.entrypoints.openai.api_server','EngineCore','Worker_TP'))]
record=dict(fusion_status='passed sampled correctness; installed in development plugin',empty_status='blocked; no safe candidate passed full gate',agreements=agreements,paired_tpot=paired,c2_status='exploratory; initial protocol did not independently prewarm C2; no reliable gain claim',processes_remaining=remaining,no_publication=True,tests='11 passed',evidence_scope='20 unique MMLU-Pro rows repeated across three pairs; not full five-dataset evaluation')
root.joinpath('results/repair-conclusion.json').write_text(json.dumps(record,indent=2))
index=[]
for path in sorted(root.joinpath('results').rglob('*')):
    if not path.is_file() or path.name in ('repair-manifest.json','finalization.log'): continue
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    index.append(dict(path=str(path.relative_to(root)),size=path.stat().st_size,sha256=h.hexdigest()))
root.joinpath('results/repair-manifest.json').write_text(json.dumps(dict(schema='opset-repair-evidence-v1',artifacts=index),indent=2))
print(json.dumps(record,indent=2))
