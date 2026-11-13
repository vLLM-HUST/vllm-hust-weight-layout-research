"""Aggregate raw requests by lifecycle; keep positive/zero/negative results."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

root=Path(__file__).resolve().parent
suite=json.loads(root.joinpath('matched-suite.json').read_text())
rows=[]; quality=[]; groups={}
for state in suite['runs']:
    run=root/'results'/state['run_id']
    if not run.joinpath('measurements.json').exists(): continue
    measurements=json.loads(run.joinpath('measurements.json').read_text())
    quality.append(dict(run_id=state['run_id'],variant=state['variant'],round=state['round'],accuracy=state['quality_accuracy'],samples=state['quality_samples']))
    for item in measurements:
        samples=item.get('requests',[item])
        for metric in ('ttft_ms','tpot_ms','output_tokens_s'):
            values=[x[metric] for x in samples if x.get(metric) is not None]
            if not values: continue
            row=dict(run_id=state['run_id'],variant=state['variant'],round=state['round'],workload=item['workload'],concurrency=item['concurrency'],sample=item['sample'],metric=metric,value=statistics.mean(values))
            rows.append(row)
            groups.setdefault((state['variant'],item['workload'],item['concurrency'],metric,state['round']),[]).extend(values)
        if 'aggregate_tokens_s' in item:
            row=dict(run_id=state['run_id'],variant=state['variant'],round=state['round'],workload=item['workload'],concurrency=2,sample=item['sample'],metric='aggregate_tokens_s',value=item['aggregate_tokens_s'])
            rows.append(row); groups.setdefault((state['variant'],item['workload'],2,'aggregate_tokens_s',state['round']),[]).append(row['value'])
lifecycle=[]
for (variant,workload,concurrency,metric,iteration),values in groups.items():
    lifecycle.append(dict(variant=variant,workload=workload,concurrency=concurrency,metric=metric,round=iteration,mean=statistics.mean(values),sd=statistics.stdev(values) if len(values)>1 else 0,variance=statistics.variance(values) if len(values)>1 else 0,samples=len(values)))
comparison=[]
for key in sorted({(r['workload'],r['concurrency'],r['metric']) for r in lifecycle}):
    row=dict(workload=key[0],concurrency=key[1],metric=key[2])
    for variant in ('baseline','fusion'):
        values=[r['mean'] for r in lifecycle if r['variant']==variant and (r['workload'],r['concurrency'],r['metric'])==key]
        row[variant]=dict(mean=statistics.mean(values),sd=statistics.stdev(values) if len(values)>1 else None,variance=statistics.variance(values) if len(values)>1 else None,lifecycles=len(values)) if values else None
    if all(row[v] for v in ('baseline','fusion')):
        ratio=row['fusion']['mean']/row['baseline']['mean']
        row['improvement_pct']=(ratio-1)*100 if key[2].endswith('tokens_s') else (1-ratio)*100
    comparison.append(row)
record=dict(suite_status=suite['status'],quality=quality,lifecycle=lifecycle,comparison=comparison,scope='sampled development evidence, not full MMLU-Pro or five-dataset results',startup_excluded=True,warmup_excluded=True,profiler_excluded=True)
root.joinpath('results/matched-summary.json').write_text(json.dumps(record,indent=2))
with root.joinpath('results/matched-raw-metrics.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]) if rows else ['run_id']); writer.writeheader(); writer.writerows(rows)
checksums=[]
for state in suite['runs']:
    for path in (root/'results'/state['run_id']).rglob('*'):
        if path.is_file():
            h=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
            checksums.append(h.hexdigest()+'  '+str(path.relative_to(root)))
root.joinpath('results/matched-SHA256SUMS').write_text('\n'.join(checksums)+'\n')
print(json.dumps(record,indent=2))
