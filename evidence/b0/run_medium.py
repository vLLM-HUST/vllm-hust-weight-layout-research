"""Revised formal cohort: keep original B0 evidence immutable."""
import asyncio, collections, fcntl, gzip, json, signal
import run_b0 as b

CONC=[1,4,8,16,32,64]

def reuse():
    manifest=json.loads((b.DATA/'medium/manifest.json').read_text())
    assert manifest['parent_manifest_sha256']==b.sha(b.DATA/'manifest.json')
    for name,meta in manifest['files'].items():
        assert b.sha(b.DATA/'medium'/name)==meta['sha256'],name
    quality=b.records('medium','quality')
    source=b.ROOT/'cells/large-quality/requests.jsonl'
    if source.exists():
        raw={r['id']:r for r in map(json.loads,source.read_text().splitlines())}
        with gzip.open(b.DATA/'medium/quality.answers.jsonl.gz','rt') as f:
            answers={r['id']:r for r in map(json.loads,f)}
        if all(r['id'] in raw for r in quality):
            rows=[raw[r['id']] for r in quality]
            cats=collections.defaultdict(lambda:dict(total=0,correct=0,invalid=0,failed=0))
            for request in quality:
                r=raw[request['id']];a=answers[r['id']];c=cats[a['category']]
                text=r.get('text','').strip()
                valid=r['success'] and len(text)==1 and 'A'<=text<chr(65+request['num_options'])
                c['total']+=1;c['correct']+=int(valid and text==a['answer'])
                c['invalid']+=int(not valid);c['failed']+=int(not r['success'])
            for c in cats.values():c['accuracy']=c['correct']/c['total']
            target=b.ROOT/'cells/medium-quality';target.mkdir(exist_ok=True)
            (target/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            correct=sum(c['correct'] for c in cats.values())
            b.save(target/'summary.json',{'status':'completed','reused':True,'source':'large-quality',
                'requests':len(rows),'successes':sum(r['success'] for r in rows),
                'failures':sum(not r['success'] for r in rows),
                'quality':{'correct':correct,'total':len(rows),'accuracy':correct/len(rows),
                    'macro_accuracy':sum(c['accuracy'] for c in cats.values())/len(cats),
                    'invalid_including_failed':sum(c['invalid'] for c in cats.values()),'by_category':dict(cats)},
                'timing_scope':'quality outputs reused by ID; no Medium batch throughput inferred'})
    refs={}
    for folder in sorted((b.ROOT/'cells').glob('large-r*')):
        if not (folder/'summary.json').exists():continue
        summary=json.loads((folder/'summary.json').read_text())
        if summary.get('status')!='completed':continue
        bucket=folder.name.split('-')[2]
        ids={r['id'] for r in b.records('medium','performance',bucket)}
        rows=[r for r in map(json.loads,(folder/'requests.jsonl').read_text().splitlines()) if r['id'] in ids]
        refs[folder.name]={'source_summary':str(folder/'summary.json'),'matched_requests':len(rows),
            'matched_ttft_seconds':b.describe([r['ttft_seconds'] for r in rows if r['success']]),
            'matched_tpot_seconds':b.describe([r['tpot_seconds'] for r in rows if r['success'] and r['tpot_seconds'] is not None]),
            'scope':'latency observed within original Large workload; keep original full-cohort throughput separately; do not pool with Medium'}
    b.save(b.ROOT/'medium-reuse-map.json',{'utc':b.now(),'small':'all completed cells reused unchanged',
        'medium_manifest_sha256':b.sha(b.DATA/'medium/manifest.json'),'large_references':refs,
        'quality_source':'large-quality, same original IDs, exact outputs',
        'performance_policy':'three fresh starts per context for Medium; preserve Large evidence separately'})

async def main():
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM,asyncio.current_task().cancel)
    lock=(b.ROOT/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    reuse()
    cfg=json.loads((b.ROOT/'baseline-frozen.json').read_text());budget=cfg['budget']
    b.save(b.ROOT/'medium-protocol.json',{'utc':b.now(),'default_splits':['small','medium'],
        'performance_requests_per_bucket':64,'quality_requests':2406,'concurrency':CONC,
        'max_num_seqs':64,'repeats':3,'frozen_config':cfg,
        'high_concurrency_note':'64 requests support client concurrency64; Large64 evidence retained separately',
        'small_reused':True,'large_immutable':True,'stability_seconds':900})
    for repeat in range(1,4):
        groups=[([1,2],4096),([3],16384),([4],65536),([5],262144)]
        if repeat%2==0:groups.reverse()
        for buckets,context in groups:
            tasks=[]
            if repeat==1 and context==4096 and not (b.ROOT/'cells/medium-quality/summary.json').exists():
                tasks.append(('medium-quality','medium','quality',None,64))
            for bucket in buckets:
                conc=CONC if repeat%2 else CONC[::-1]
                tasks.extend((f'medium-r{repeat}-P{bucket}-C{c}','medium','performance',f'P{bucket}',c) for c in conc)
            await b.group(f'medium-r{repeat}-ctx{context}-seq64-budget{budget}',context,64,budget,tasks)
        b.summarize()
    results=b.summarize()
    selected={k:v for k,v in results.items() if k.startswith('medium-r')}
    b.save(b.ROOT/'step5.complete.json',{'utc':b.now(),'revision':'medium-v2',
        'performance_cells':len(selected),'statuses':dict(collections.Counter(r['status'] for r in selected.values())),
        'quality_result':results.get('medium-quality'),'budget':budget})
    b.status('all_five_steps_finished',revision='medium-v2')

if __name__=='__main__':asyncio.run(main())
