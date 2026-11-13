"""Deterministic nested Medium; never change original Small/Large files."""
import collections, gzip, hashlib, json, random
from pathlib import Path

SEED=20261023
import os
BASE=Path(os.environ.get('MEDIUM_DATA_BASE',str(Path(__file__).resolve().parent.parent/'mmlu-weight-layout-v1')))
def read(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return [json.loads(s) for s in f]
def write(path,rows):
    with path.open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as gz:
            for r in rows:gz.write((json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n').encode())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    dst=BASE/'medium';dst.mkdir(exist_ok=True)
    source=read(BASE/'large/performance.requests.jsonl.gz');perf=[]
    for bucket in ['P1','P2','P3','P4','P5']:
        rows=[r for r in source if r['bucket']==bucket]
        random.Random(f'{SEED}:large:performance:{bucket}').shuffle(rows)
        perf.extend(rows[:64])
    answers=read(BASE/'large/quality.answers.jsonl.gz')
    cats=collections.defaultdict(list)
    for a in answers:cats[a['category']].append(a)
    target=round(len(answers)/5)
    quota={c:len(v)*target//len(answers) for c,v in cats.items()}
    order=sorted(cats,key=lambda c:(-(len(cats[c])*target%len(answers)),c))
    for c in order[:target-sum(quota.values())]:quota[c]+=1
    selected=[]
    for c,rows in sorted(cats.items()):
        rows.sort(key=lambda a:hashlib.sha256(f'{SEED}:medium:quality:{a["id"]}'.encode()).digest())
        selected.extend(rows[:quota[c]])
    ids={a['id'] for a in selected}
    quality=[r for r in read(BASE/'large/quality.requests.jsonl.gz') if r['id'] in ids]
    assert len(quality)==len(ids)==target
    write(dst/'performance.requests.jsonl.gz',perf)
    write(dst/'quality.requests.jsonl.gz',quality)
    write(dst/'quality.answers.jsonl.gz',selected)
    files={p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(dst.glob('*.gz'))}
    manifest={'version':'medium-v2','seed':SEED,'parent_manifest_sha256':digest(BASE/'manifest.json'),
        'performance_count':len(perf),'performance_per_bucket':64,'quality_count':target,
        'quality_categories':quota,'files':files,'performance_ids':[r['id'] for r in perf],
        'quality_ids':sorted(ids),'selection':'performance: first64 in frozen Large per-bucket submission order; quality: Hamilton proportional subject quota, SHA256 ID ranking',
        'concurrency':[1,4,8,16,32,64],'max_num_seqs':64,
        'reuse':'original IDs and token arrays unchanged; quality may be scored from completed Large outputs; Large batch throughput never relabelled Medium'}
    (dst/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps({k:manifest[k] for k in ['performance_count','quality_count','quality_categories','files']},indent=2))

if __name__=='__main__':main()
