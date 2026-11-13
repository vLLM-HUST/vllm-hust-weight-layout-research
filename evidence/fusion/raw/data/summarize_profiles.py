"""Kernel-duration shares, not wall-time shares (overlap is not additive)."""
import csv
import json
from pathlib import Path
from collections import defaultdict

root=Path(__file__).resolve().parent
suite=json.loads(root.joinpath('matched-suite.json').read_text())
summaries=[]
for state in suite['runs']:
    if state['round']!=3: continue
    for path in (root/'results'/state['run_id']/'profile').rglob('kernel_details.csv'):
        groups=defaultdict(lambda:dict(count=0,duration_us=0.,wait_us=0.))
        total=0.; instances=0
        with path.open(encoding='utf-8-sig',newline='') as stream:
            for row in csv.DictReader(stream):
                duration=float(row['Duration(us)']); total+=duration; instances+=1
                kind=row.get('Type') or row['Name']; name=row['Name']
                if 'linear_swiglu' in name: kind='shared_linear_swiglu'
                elif 'linear_persistent' in name: kind='linear_persistent'
                group=groups[kind]; group['count']+=1; group['duration_us']+=duration
                group['wait_us']+=float(row.get('Wait Time(us)') or 0)
        for value in groups.values():
            value['share_pct']=100*value['duration_us']/total
            value['mean_us']=value['duration_us']/value['count']
        summaries.append(dict(variant=state['variant'],run_id=state['run_id'],path=str(path.relative_to(root)),instances=instances,kernel_duration_sum_us=total,operators=dict(sorted(groups.items(),key=lambda x:-x[1]['duration_us']))))
root.joinpath('results/profile-summary.json').write_text(json.dumps(summaries,indent=2))
for row in summaries:
    print(row['variant'],row['path'],[(k,v['count'],round(v['share_pct'],3)) for k,v in row['operators'].items()][:12])
