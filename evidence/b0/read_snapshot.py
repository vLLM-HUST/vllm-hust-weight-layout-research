"""Read-only evidence snapshot: no API calls, signals, or NPU workloads."""
import collections, datetime, json
from pathlib import Path

root=Path('/root/opset-research/b0-weight-layout-20261023')
cells={p.parent.name:json.loads(p.read_text()) for p in sorted((root/'cells').glob('*/summary.json'))}
medium={k:v for k,v in cells.items() if k.startswith('medium-r')}
snapshot={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'status':json.loads((root/'status.json').read_text()),
 'completed_medium_cells':len(medium),'planned_medium_cells':90,
 'medium_status_counts':dict(collections.Counter(v['status'] for v in medium.values())),
 'by_repeat':{str(r):sum(k.startswith(f'medium-r{r}-') for k in medium) for r in [1,2,3]},
 'extras_complete':(root/'extras.complete.json').exists(),
 'stability_complete':(root/'stability/P2/summary.json').exists(),
 'profile_receipts':[str(p.relative_to(root)) for p in (root/'profiles').glob('*/receipt.json')],
 'memory_receipts':[str(p.relative_to(root)) for p in (root/'memory').glob('*/receipt.json')],
 'cells':cells}
print(json.dumps(snapshot,ensure_ascii=False))
