"""Read saved experiment artifacts into a small audit bundle."""
import collections,datetime,json,re,statistics,tarfile
from pathlib import Path
root=Path('/root/opset-research/b0-weight-layout-20261023')
out={'collected_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
names=['step5.complete.json','extras.complete.json','baseline-frozen.json','medium-protocol.json',
 'environment.json','runtime-effective-config.json','medium-reuse-map.json','status.json']
out['metadata']={n:json.loads((root/n).read_text()) for n in names if (root/n).exists()}
out['cells']={p.parent.name:json.loads(p.read_text()) for p in sorted((root/'cells').glob('*/summary.json'))}
out['stability']=json.loads((root/'stability/P2/summary.json').read_text())
out['memory']={str(p.relative_to(root)):json.loads(p.read_text()) for p in list((root/'profiles').glob('*/memory.*.json'))+list((root/'memory').glob('*/receipt.json'))}
out['profiles']={p.parent.name:json.loads(p.read_text()) for p in (root/'profiles').glob('*/receipt.json')}
out['servers']={}
def metrics(path):
    if not path.exists():return {}
    found=collections.defaultdict(float)
    for l in path.read_text().splitlines():
        if l.startswith('#'):continue
        k=l.split('{')[0].split(' ')[0]
        if 'preemptions' not in k:continue
        try:found[k]+=float(l.rsplit(' ',1)[1])
        except ValueError:pass
    return dict(found)
for p in sorted((root/'servers').glob('*/server.log')):
    lines=p.read_text(errors='replace').splitlines()
    modes=collections.Counter()
    for line in lines:
        m=re.search(r'\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(FULL|PIECEWISE|NONE)\s*\|\s*(\d+)',line)
        if m:modes[m[1]]+=int(m[2])
    out['servers'][p.parent.name]={
      'memory':[l for l in lines if re.search(r'Loading model weights took|Available KV cache|GiB for|Graph capturing finished|Maximum concurrency',l)],
      'compilation_or_fallback':[l for l in lines if re.search(r'recompil|fallback|falling back',l,re.I)],
      'graph_mode_observed_counts':dict(modes)}
out['preemptions']={p.name:{'before':metrics(p/'metrics.before.txt'),'after':metrics(p/'metrics.after.txt')} for p in list((root/'cells').glob('*'))+[root/'stability/P2']}
out['cleanup_text']=(root/'final-npu-state.txt').read_text()
(root/'final-evidence.json').write_text(json.dumps(out,ensure_ascii=False),encoding='utf-8')
with tarfile.open(root/'final-audit.tar.gz','w:gz') as tar:
    tar.add(root/'final-evidence.json',arcname='final-evidence.json')
    for p in [root/'npu-samples.jsonl',root/'metrics-samples.jsonl',root/'medium-campaign.log',root/'final-npu-state.txt',root/'stability/P2/requests.jsonl']:
        if p.exists():tar.add(p,arcname=str(p.relative_to(root)))
print('bundle bytes',(root/'final-audit.tar.gz').stat().st_size,flush=True)
