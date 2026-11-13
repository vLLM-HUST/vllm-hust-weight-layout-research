"""Summarize CANN offline output; keep timing scopes explicit."""
import csv,json,re
from pathlib import Path
root=Path('/root/opset-research/b0-weight-layout-20261023/profiles')
rows=[]
for path in sorted(root.glob('*/*_ascend_pt/ASCEND_PROFILER_OUTPUT')):
    step=path/'step_trace_time.csv'
    if not step.exists():continue
    rank=int(re.search(r'_rank(\d+)_',path.parent.name)[1])
    data=list(csv.DictReader(step.open()))
    totals={k:sum(float(r[k]) for r in data if r.get(k)) for k in [
        'Computing','Communication(Not Overlapped)','Overlapped','Communication','Free','Stage','Preparing']}
    comm=json.loads((path/'communication.json').read_text())
    ops=[]
    for group in comm.values():
        for kind in ['p2p','collective']:
            ops.extend(v['Communication Time Info'] for k,v in group.get(kind,{}).items()
                       if 'total' not in k.lower() and 'Communication Time Info' in v)
    waits={k:sum(float(r.get(k,0)) for r in ops) for k in ['Elapse Time(ms)','Transit Time(ms)','Wait Time(ms)','Synchronization Time(ms)','Idle Time(ms)']}
    rows.append({'case':path.parent.parent.name,'rank':rank,'step_time_us':totals,
        'communication_operation_count':len(ops),'communication_operation_sums_ms':waits,
        'scope':'24-engine-iteration diagnostic capture, not full request or formal performance timing',
        'sync_wait_warning':'zero Wait/Sync fields do not prove absence of synchronization; free time is not assigned to synchronization',
        'source':str(path)})
assert len(rows)==8,f'expected8 rank traces, got{len(rows)}'
target=root.parent/'profile-analysis.json'
target.write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
