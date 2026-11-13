"""Offline audit; only standard library; no model execution or network."""
from pathlib import Path
import collections, csv, gzip, hashlib, json, math, statistics

ROOT = Path(__file__).resolve().parents[1]
def read(p): return json.loads((ROOT / p).read_text(encoding='utf-8-sig'))
def lines(p):
    with (ROOT / p).open(encoding='utf-8-sig') as f: return [json.loads(s) for s in f if s.strip()]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

manifest=read('provenance.json')
missing=[]; checked=0
for row in manifest['files']:
    p=ROOT/row['path']
    if not p.exists():
        if row['path'].startswith('release-assets/'): missing.append(row['path']); continue
        raise AssertionError('Missing evidence: '+row['path'])
    assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'], row['path']
    checked+=1

fusion_dir=ROOT/'evidence/fusion/raw/data/results'
fusion={}; details=[]
for workload in ['short','long']:
    means={}
    for arm in ['baseline','fusion']:
        means[arm]=[]
        for r in [1,2,3]:
            dirs=list(fusion_dir.glob(f'20261014T*-{arm}-r{r}'))
            assert len(dirs)==1
            measurements=json.loads((dirs[0]/'measurements.json').read_text(encoding='utf-8-sig'))
            vals=[x['tpot_ms'] for x in measurements if x['workload']==workload and x['concurrency']==1]
            assert len(vals)==5
            means[arm].append(statistics.mean(vals))
            details.append({'workload':workload,'arm':arm,'round':r,'run_id':dirs[0].name,'samples':len(vals),'tpot_ms':statistics.mean(vals)})
    b=statistics.mean(means['baseline']); f=statistics.mean(means['fusion'])
    diffs=[x-y for x,y in zip(means['baseline'],means['fusion'])]
    mid=statistics.mean(diffs); half=4.302652729911275*statistics.stdev(diffs)/math.sqrt(3)
    fusion[workload]={'baseline_tpot_ms':b,'fusion_tpot_ms':f,'improvement_pct':(b-f)/b*100,
                      'paired_difference_ms':mid,'descriptive_95pct_t_interval_ms':[mid-half,mid+half]}
assert abs(fusion['short']['improvement_pct']-1.97405)<0.001
assert abs(fusion['long']['improvement_pct']-0.67331)<0.001

wrows=[]; ratios=[]
for bucket in ['P1','P2','P3','P4']:
    for c in [1,8,32]:
        cell=f'small-{bucket}-C{c}'
        b=read(f'evidence/b0/cells/{cell}/summary.json')
        w=read(f'evidence/w1/w1-small-r1/cells/{cell}/summary.json')
        for arm,path,s in [('B0',f'evidence/b0/cells/{cell}',b),('W1',f'evidence/w1/w1-small-r1/cells/{cell}',w)]:
            records=lines(path+'/requests.jsonl')
            assert len(records)==s['requests']==32 and all(x['success'] for x in records)
            assert sum(x['output_tokens'] for x in records)==s['output_tokens']
            assert math.isclose(s['output_tokens']/s['seconds'],s['dual_card_output_tokens_per_second'],rel_tol=1e-10)
        bt=b['dual_card_output_tokens_per_second']; wt=w['dual_card_output_tokens_per_second']
        ratios.append(wt/bt); wrows.append({'cell':cell,'B0_tokens_s':bt,'W1_tokens_s':wt,'delta_pct':(wt/bt-1)*100})
geomean=(math.exp(statistics.mean(math.log(x) for x in ratios))-1)*100
assert abs(geomean+1.8)<0.01

audit=read('evidence/b0/final-audit/final-evidence.json')
cells={k:v for k,v in audit['cells'].items() if k.startswith('medium-r')}
counts=dict(collections.Counter(v['status'] for v in cells.values()))
assert len(cells)==90 and counts=={'completed':87,'warmup_failed':3}
p2=statistics.mean(cells[f'medium-r{r}-P2-C64']['dual_card_output_tokens_per_second'] for r in [1,2,3])
assert round(p2,2)==944.92
formal=sum(v['requests'] for v in cells.values() if v['status']=='completed')
assert formal==5568
with gzip.open(ROOT/'evidence/dataset/medium/quality.answers.jsonl.gz','rt',encoding='utf-8') as f:
    answers={x['id']:x['answer'] for x in map(json.loads,f)}
original=lines('evidence/b0/cells/large-quality/requests.jsonl')
selected=[x for x in original if x['id'] in answers]
assert len(selected)==len({x['id'] for x in selected})==2406
correct=sum(x['success'] and x['text'].strip()==answers[x['id']] for x in selected)
assert correct==1414
stability=lines('evidence/b0/final-audit/stability/P2/requests.jsonl')
assert len(stability)==896 and all(x['success'] for x in stability)

result={'hash_verified_files':checked,'release_assets_not_present':missing,
 'fusion_C1_recomputed_from_requests':fusion,'W1_small':{'cells':12,'requests':384,'throughput_ratio_geomean_change_pct':geomean,'scope':'single-round historical comparison; different physical cards/time'},
 'B0_medium_from_collected_cell_summaries':{'statuses':counts,'formal_requests_in_successful_cells':formal,'P2_C64_mean_tokens_s':p2},
 'B0_medium_quality_from_large_raw_outputs':{'correct':correct,'total':len(selected),'accuracy_pct':correct/len(selected)*100},
 'B0_stability_raw_requests':{'count':len(stability),'failures':0},
 'limitations':['B0 Medium performance request-level files absent; aggregated evidence only.','Full fusion profiler is in original release archive.','No new NPU experiments executed.']}
out=ROOT/'verification';out.mkdir(exist_ok=True)
(out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
for name,rows in [('fusion-lifecycle-means.csv',details),('w1-small-comparison.csv',wrows)]:
    with (out/name).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps(result,ensure_ascii=False,indent=2))
