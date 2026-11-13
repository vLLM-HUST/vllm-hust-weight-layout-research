"""Resumable B0 campaign. Uses existing runtime; never edits donor sources."""
import asyncio
import collections
import datetime
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import time
import traceback

import aiohttp
import psutil

ROOT = Path('/root/opset-research/b0-weight-layout-20261023')
DATA = Path('/root/opset-research/datasets/mmlu-weight-layout-v1')
MODEL = Path('/data/shared_upload/models/vspec/Qwen3.5-35B-A3B')
PYTHON = '/root/opset-research/.venv/bin/python'
PORT = 18123
URL = f'http://127.0.0.1:{PORT}'
SEED = 20261023
CONTEXT = {'P1':4096,'P2':4096,'P3':16384,'P4':65536,'P5':262144}
CAPTURES = [1,2,4,8,16,32,64]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def log(message):
    print(now(), message, flush=True)


def status(stage, **kw):
    save(ROOT/'status.json', {'stage':stage,'updated_utc':now(),**kw})
    log(f'{stage}: {kw}')


def run(args, timeout=120):
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    return {'argv':args,'code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()


def records(split, kind, bucket=None):
    with gzip.open(DATA/split/f'{kind}.requests.jsonl.gz','rt') as f:
        rows=[json.loads(l) for l in f]
    if bucket: rows=[r for r in rows if r['bucket']==bucket]
    random.Random(f'{SEED}:{split}:{kind}:{bucket}').shuffle(rows)
    return rows


def quantile(values, q):
    if not values: return None
    a=sorted(values); x=(len(a)-1)*q; i=int(x)
    return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)


def describe(values):
    return {'mean':statistics.mean(values) if values else None,
            'p50':quantile(values,.5),'p95':quantile(values,.95),'p99':quantile(values,.99)}


async def text_get(session, path):
    async with session.get(URL+path, timeout=aiohttp.ClientTimeout(total=30)) as r:
        return await r.text()


class Server:
    def __init__(self, name, context, sequences, budget):
        self.name=name; self.folder=ROOT/'servers'/name
        self.folder.mkdir(parents=True,exist_ok=True)
        self.context=context;self.sequences=sequences;self.budget=budget
        self.proc=None;self.logfile=None

    async def start(self):
        graph={'cudagraph_mode':'FULL_AND_PIECEWISE',
               'cudagraph_capture_sizes':[x for x in CAPTURES if x<=self.sequences],
               'cudagraph_num_of_warmups':1}
        args=[PYTHON,'-m','vllm.entrypoints.openai.api_server',
              '--model',str(MODEL),'--served-model-name','b0-qwen35',
              '--host','127.0.0.1','--port',str(PORT),
              '--dtype','bfloat16','--tensor-parallel-size','2',
              '--pipeline-parallel-size','1','--data-parallel-size','1',
              '--no-enable-expert-parallel','--no-enforce-eager',
              '--no-async-scheduling','--enable-chunked-prefill',
              '--no-enable-prefix-caching','--max-model-len',str(self.context),
              '--max-num-seqs',str(self.sequences),'--max-num-batched-tokens',str(self.budget),
              '--gpu-memory-utilization','0.85','--kv-cache-dtype','auto',
              '--block-size','128','--mamba-cache-mode','none',
              '--seed',str(SEED),'--limit-mm-per-prompt','{"image":0,"video":0}',
              '--compilation-config',json.dumps(graph),
              '--additional-config','{"enable_cpu_binding":false}',
              '--cudagraph-metrics','--no-enable-log-requests']
        if self.name.startswith('profile-'):
            args += ['--profiler-config', json.dumps({
                'profiler':'torch','torch_profiler_dir':str(ROOT/'profiles'/self.name),
                'torch_profiler_with_stack':False,'torch_profiler_with_memory':True,
                'max_iterations':24,'ignore_frontend':True})]
        env=os.environ.copy()
        for k in list(env):
            if k.startswith(('VLLM_HUST','BETTERSCALE','VLLM_BATCH_INVARIANT')):
                env.pop(k,None)
        env.update({'VLLM_PLUGINS':'ascend','TORCH_DEVICE_BACKEND_AUTOLOAD':'0',
                    'ASCEND_RT_VISIBLE_DEVICES':os.environ.get('B0_DEVICES','0,1'),'OMP_NUM_THREADS':'1',
                    'TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1',
                    'VLLM_CACHE_ROOT':str(ROOT/'compile-cache'),
                    'VLLM_WORKER_MULTIPROC_METHOD':'spawn'})
        save(self.folder/'launch.json',{'argv':args,'environment':{k:v for k,v in env.items()
            if k.startswith(('VLLM','ASCEND','OMP','TORCH_DEVICE','TOKENIZERS'))},
            'started_utc':now(),'context':self.context,'sequences':self.sequences,'budget':self.budget})
        self.logfile=(self.folder/'server.log').open('a')
        args=['numactl','--interleave=2,4']+args
        save(self.folder/'binding-policy.json',{'physical_npuid':[6,7],
            'worker_cpu_affinity':{'TP0':'96-119','TP1':'48-71'},
            'api_cpus':'24-31','engine_cpus':'32-39','memory_policy':'interleave NUMA2,4',
            'binding_point':'after initialization, before all request warmups and timings'})
        self.proc=subprocess.Popen(args,stdout=self.logfile,stderr=subprocess.STDOUT,
                                   env=env,start_new_session=True)
        save(self.folder/'pid.json',{'pid':self.proc.pid,'pgid':self.proc.pid})
        start=time.monotonic()
        async with aiohttp.ClientSession() as session:
            while time.monotonic()-start<1200:
                if self.proc.poll() is not None:
                    save(self.folder/'startup.json',{'ready':False,'exit_code':self.proc.returncode,
                        'seconds':time.monotonic()-start})
                    raise RuntimeError(f'server exited {self.proc.returncode}: {self.folder}')
                try:
                    async with session.get(URL+'/health',timeout=aiohttp.ClientTimeout(total=2)) as r:
                        if r.status==200: break
                except Exception: pass
                await asyncio.sleep(3)
            else: raise TimeoutError(f'server startup exceeded 1200s: {self.name}')
            self.bind_explicit()
            save(self.folder/'startup.json',{'ready':True,'seconds':time.monotonic()-start})
            (self.folder/'metrics.ready.txt').write_text(await text_get(session,'/metrics'))
        save(self.folder/'npu.ready.json',run(['npu-smi','info']))
        self.snapshot_affinity()
        log(f'SERVER READY {self.name} in {time.monotonic()-start:.1f}s')

    def bind_explicit(self):
        processes=[psutil.Process(self.proc.pid)]+psutil.Process(self.proc.pid).children(recursive=True)
        seen=set();receipt=[]
        for p in processes:
            name=p.name()
            if 'Worker_TP0' in name:cpus=set(range(96,120));seen.add(0)
            elif 'Worker_TP1' in name:cpus=set(range(48,72));seen.add(1)
            elif 'EngineCore' in name:cpus=set(range(32,40))
            else:cpus=set(range(24,32))
            os.sched_setaffinity(p.pid,cpus)
            exceptions=[]
            for thread in p.threads():
                try:os.sched_setaffinity(thread.id,cpus)
                except ProcessLookupError:pass
                except OSError as e:
                    exceptions.append({'tid':thread.id,'error':str(e),
                        'name':Path(f'/proc/{p.pid}/task/{thread.id}/comm').read_text().strip(),
                        'affinity':sorted(os.sched_getaffinity(thread.id))})
            assert set(p.cpu_affinity())==cpus,(p.pid,name,p.cpu_affinity())
            receipt.append({'pid':p.pid,'name':name,'cpus':sorted(cpus),'threads':len(p.threads()),
                            'unmodifiable_vendor_threads':exceptions})
        if seen!={0,1}:raise RuntimeError(f'cannot identify both TP workers for binding: {seen}')
        save(self.folder/'explicit-binding.json',receipt)

    def snapshot_affinity(self):
        a=[]
        try:
            for p in [psutil.Process(self.proc.pid)]+psutil.Process(self.proc.pid).children(recursive=True):
                try:
                    st=Path(f'/proc/{p.pid}/status').read_text().splitlines()
                    numa=Path(f'/proc/{p.pid}/numa_maps').read_text()
                    nodes=collections.Counter()
                    import re
                    for node,count in re.findall(r'\bN(\d+)=(\d+)',numa):nodes[node]+=int(count)
                    a.append({'pid':p.pid,'name':p.name(),'cpu_affinity':p.cpu_affinity(),
                              'allowed':[l for l in st if l.startswith(('Cpus_allowed_list','Mems_allowed_list'))],
                              'numa_resident_pages':dict(nodes)})
                except (psutil.Error,FileNotFoundError):pass
        except psutil.Error:pass
        save(self.folder/'affinity.json',a)

    async def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                async with aiohttp.ClientSession() as s:
                    (self.folder/'metrics.final.txt').write_text(await text_get(s,'/metrics'))
            except Exception:pass
            os.killpg(self.proc.pid,signal.SIGTERM)
            for _ in range(60):
                if self.proc.poll() is not None:break
                await asyncio.sleep(1)
            try:os.killpg(self.proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            self.proc.wait(timeout=30)
        if self.logfile:self.logfile.close()
        await asyncio.sleep(4)
        save(self.folder/'npu.after.json',run(['npu-smi','info']))


async def request(session, row, quality):
    start=time.perf_counter(); first=None;last_token=None;ids=[];parts=[];usage={};chunks=0;reason=None
    payload={'model':'b0-qwen35','prompt':row['prompt_token_ids'],
             'max_tokens':row['output_tokens'],'temperature':0,'seed':SEED,
             'ignore_eos':not quality,'stream':True,'return_token_ids':True,
             'stream_options':{'include_usage':True},'add_special_tokens':False}
    result={'id':row['id'],'input_tokens':row['input_tokens'],
            'expected_output_tokens':row['output_tokens'],'started_utc':now()}
    try:
        async with session.post(URL+'/v1/completions',json=payload,
                                timeout=aiohttp.ClientTimeout(total=3600,sock_read=600)) as resp:
            if resp.status!=200:raise RuntimeError(f'HTTP {resp.status}: {(await resp.text())[:2000]}')
            while True:
                line=await resp.content.readline()
                if not line:break
                if not line.startswith(b'data: '):continue
                raw=line[6:].strip()
                if raw==b'[DONE]':break
                event=json.loads(raw)
                if 'error' in event:raise RuntimeError(str(event['error']))
                if event.get('usage'):usage=event['usage']
                for choice in event.get('choices',[]):
                    delta=choice.get('token_ids') or []
                    content=choice.get('text') or ''
                    if delta or content:
                        tick=time.perf_counter()
                        if first is None:first=tick
                        last_token=tick;chunks+=1
                        ids.extend(delta);parts.append(content)
                    if choice.get('finish_reason'):reason=choice['finish_reason']
        end=time.perf_counter()
        n=usage.get('completion_tokens',len(ids))
        if usage.get('prompt_tokens')!=row['input_tokens']:
            raise ValueError(f'prompt token mismatch: {usage.get("prompt_tokens")} vs {row["input_tokens"]}')
        if n!=len(ids):raise ValueError(f'output IDs/usage mismatch: {len(ids)} vs {n}')
        if not quality and n!=row['output_tokens']:raise ValueError(f'short output: {n}')
        if first is None:raise ValueError('empty generation')
        result.update({'success':True,'output_tokens':n,'output_token_ids':ids,
                       'text':''.join(parts),'finish_reason':reason,'usage':usage,
                       'e2e_seconds':end-start,'ttft_seconds':first-start,
                       'tpot_seconds':((last_token-first)/(n-1) if n>1 else None),
                       'stream_chunks':chunks})
    except Exception as e:
        result.update({'success':False,'error':f'{type(e).__name__}: {e}',
                       'e2e_seconds':time.perf_counter()-start,'output_token_ids':ids,'text':''.join(parts)})
    return result


async def batch(name, rows, concurrency, quality=False, warmup=True):
    folder=ROOT/'cells'/name; summary=folder/'summary.json'
    if summary.exists():
        log(f'RESUME completed cell {name}')
        return json.loads(summary.read_text())
    folder.mkdir(parents=True,exist_ok=True)
    save(folder/'spec.json',{'name':name,'requests':len(rows),'concurrency':concurrency,
        'quality':quality,'request_ids':[r['id'] for r in rows],
        'seed':SEED,'warmup_requests':min(concurrency,len(rows)) if warmup else 0})
    results=[]
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=concurrency+4),
                                    read_bufsize=4<<20) as session:
        if warmup:
            # Full representative requests, excluded from timing and retained separately.
            warm=await asyncio.gather(*(request(session,r,quality) for r in rows[:min(concurrency,len(rows))]))
            save(folder/'warmup.json',warm)
            if any(not r['success'] for r in warm):
                failure={'status':'warmup_failed','successes':0,'failures':len(rows),
                         'warmup_errors':[r.get('error') for r in warm if not r['success']]}
                save(summary,failure);return failure
        (folder/'metrics.before.txt').write_text(await text_get(session,'/metrics'))
        queue=asyncio.Queue()
        for row in rows:queue.put_nowait(row)
        out=(folder/'requests.jsonl').open('w',encoding='utf-8')
        async def worker():
            while not queue.empty():
                row=queue.get_nowait(); res=await request(session,row,quality)
                results.append(res);out.write(json.dumps(res,ensure_ascii=False)+'\n');out.flush()
                if len(results)%32==0:log(f'CELL {name}: {len(results)}/{len(rows)}')
                queue.task_done()
        start=time.perf_counter()
        try:await asyncio.gather(*(worker() for _ in range(concurrency)))
        finally:out.close()
        duration=time.perf_counter()-start
        (folder/'metrics.after.txt').write_text(await text_get(session,'/metrics'))
    good=[r for r in results if r['success']]
    result={'status':'completed' if len(good)==len(rows) else 'request_failures',
            'requests':len(rows),'successes':len(good),'failures':len(rows)-len(good),
            'concurrency':concurrency,'seconds':duration,
            'output_tokens':sum(r['output_tokens'] for r in good),
            'input_tokens':sum(r['input_tokens'] for r in good),
            'input_tokens_per_second':sum(r['input_tokens'] for r in good)/duration,
            'dual_card_output_tokens_per_second':sum(r['output_tokens'] for r in good)/duration,
            'requests_per_second':len(good)/duration,
            'ttft_seconds':describe([r['ttft_seconds'] for r in good]),
            'tpot_seconds':describe([r['tpot_seconds'] for r in good if r['tpot_seconds'] is not None]),
            'e2e_seconds':describe([r['e2e_seconds'] for r in good])}
    if quality:
        split='small' if 'small' in name else 'large'
        with gzip.open(DATA/split/'quality.answers.jsonl.gz','rt') as f:
            answers={r['id']:r for r in map(json.loads,f)}
        categories=collections.defaultdict(lambda:dict(total=0,correct=0,invalid=0,failed=0))
        correct=invalid=0
        by_id={r['id']:r for r in results}
        for row in rows:
            res=by_id[row['id']];a=answers[row['id']];cat=categories[a['category']];cat['total']+=1
            text=res.get('text','').strip()
            valid=res['success'] and len(text)==1 and 'A'<=text<chr(65+row['num_options'])
            hit=valid and text==a['answer']
            correct+=int(hit);invalid+=int(not valid)
            cat['correct']+=int(hit);cat['invalid']+=int(not valid);cat['failed']+=int(not res['success'])
        for cat in categories.values():cat['accuracy']=cat['correct']/cat['total']
        result['quality']={'correct':correct,'total':len(rows),'accuracy':correct/len(rows),
                           'invalid_including_failed':invalid,'by_category':dict(categories),
                           'macro_accuracy':statistics.mean(c['accuracy'] for c in categories.values())}
    save(summary,result)
    save(folder/'npu.after.json',run(['npu-smi','info']))
    log(f'CELL DONE {name}: {result["status"]}, {result["dual_card_output_tokens_per_second"]:.2f} tok/s')
    return result


def freeze():
    target=ROOT/'environment.json'
    if target.exists():return
    status('step1_freeze')
    packages={n:importlib.metadata.version(n) for n in ['torch','torch-npu','vllm','vllm-ascend','transformers','aiohttp']}
    repos={}
    for name in ['runtime-vllm','runtime-ascend']:
        p=Path('/root/opset-research/sources')/name
        repos[name]={'head':run(['git','-C',str(p),'rev-parse','HEAD']),
                     'status':run(['git','-C',str(p),'status','--short']),
                     'diff_sha256':hashlib.sha256(run(['git','-C',str(p),'diff','HEAD'])['stdout'].encode()).hexdigest()}
    models={}
    for p in sorted(MODEL.iterdir()):
        if p.is_file():
            log(f'HASH model {p.name}')
            models[p.name]={'bytes':p.stat().st_size,'sha256':sha(p)}
    manifest=json.loads((DATA/'manifest.json').read_text())
    for rel,meta in manifest['files'].items():
        if sha(DATA/rel)!=meta['sha256']:raise ValueError(f'dataset hash mismatch {rel}')
    for name,h in manifest['tokenizer_files'].items():
        if sha(MODEL/name)!=h:raise ValueError(f'tokenizer mismatch {name}')
    save(target,{'frozen_utc':now(),'packages':packages,'repos':repos,'model_files':models,
        'dataset_manifest_sha256':sha(DATA/'manifest.json'),
        'npu':run(['npu-smi','info']),'topology':run(['npu-smi','info','-t','topo']),
        'lscpu':run(['lscpu']),'numa':run(['numactl','--hardware']),
        'driver':Path('/usr/local/Ascend/driver/version.info').read_text(),
        'cann':Path('/usr/local/Ascend/cann-9.1.0/compiler/version.info').read_text(),
        'python':sys.version,'runner_sha256':sha(Path(__file__))})


async def group(name, context, sequences, budget, tasks):
    if all((ROOT/'cells'/t[0]/'summary.json').exists() for t in tasks):return
    server=Server(name,context,sequences,budget)
    try:
        status('running_group',group=name,context=context,sequences=sequences,budget=budget)
        await server.start()
        for cell,split,kind,bucket,c in tasks:
            status('running_cell',group=name,cell=cell)
            await batch(cell,records(split,kind,bucket),c,kind=='quality')
    except Exception as e:
        save(server.folder/'failure.json',{'error':str(e),'traceback':traceback.format_exc(),'utc':now()})
        log(f'GROUP FAILED {name}: {e}')
        # Every missing cell is explicitly accounted for, not silently skipped.
        for cell,split,kind,bucket,c in tasks:
            p=ROOT/'cells'/cell/'summary.json'
            if not p.exists():save(p,{'status':'server_failed','error':str(e),'group':name})
    finally:await server.stop()


def summarize():
    cells={str(p.parent.relative_to(ROOT/'cells')):json.loads(p.read_text())
           for p in sorted((ROOT/'cells').glob('*/summary.json'))}
    save(ROOT/'all-results.json',cells)
    return cells


async def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM,asyncio.current_task().cancel)
    # One campaign owner; never kill unrelated listeners or jobs.
    import fcntl
    lock=(ROOT/'campaign.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    freeze()
    if not (ROOT/'step1.complete.json').exists():save(ROOT/'step1.complete.json',{'completed_utc':now()})
    await group('small-ctx4096-seq32-budget4096',4096,32,4096,
        [('small-quality','small','quality',None,8)]+
        [(f'small-P{b}-C{c}','small','performance',f'P{b}',c) for b in [1,2] for c in [1,8,32]])
    save(ROOT/'step2.complete.json',{'result':json.loads((ROOT/'cells/small-quality/summary.json').read_text()),'utc':now()})
    for b in [3,4]:
        await group(f'small-P{b}-seq32-budget4096',CONTEXT[f'P{b}'],32,4096,
            [(f'small-P{b}-C{c}','small','performance',f'P{b}',c) for c in [1,8,32]])
    save(ROOT/'step3.complete.json',{'utc':now(),'cells':[f'small-P{b}-C{c}' for b in range(1,5) for c in [1,8,32]]})
    # Bounded budget tuning on complete Small buckets; never tune using Large quality data.
    for b in [2,3]:
        await group(f'tuning-P{b}-seq32-budget8192',CONTEXT[f'P{b}'],32,8192,
            [(f'tuning-budget8192-P{b}-C32','small','performance',f'P{b}',32)])
    frozen=ROOT/'baseline-frozen.json'
    if not frozen.exists():
        results=summarize();ratios=[]
        for b in [2,3]:
            a=results[f'small-P{b}-C32'];z=results[f'tuning-budget8192-P{b}-C32']
            if a.get('status')=='completed' and z.get('status')=='completed':
                ratios.append(z['dual_card_output_tokens_per_second']/a['dual_card_output_tokens_per_second'])
        ratio=math.prod(ratios)**(1/len(ratios)) if len(ratios)==2 else None
        budget=8192 if ratio is not None and ratio>=1.05 else 4096
        save(frozen,{'utc':now(),'budget':budget,'small_sequences':32,'large_sequences':64,
            'budget8192_vs4096_geomean':ratio,'selection_rule':'8192 only if both tuning cells complete and geomean >=1.05; otherwise retain4096',
            'contexts':CONTEXT,'graph_mode':'FULL_AND_PIECEWISE','captures':CAPTURES,
            'tp':2,'ep':False,'pp':1,'dp':1,'async':False,'apc':False,'mtp':False,
            'gpu_memory_utilization':.85,'block_size':128,'mamba_cache_mode':'none'})
    cfg=json.loads(frozen.read_text());budget=cfg['budget']
    save(ROOT/'step4.complete.json',cfg)
    for repeat in range(1,4):
        # Reverse traversal on even repeats to expose ordering/thermal effects.
        groups=[([1,2],4096),([3],16384),([4],65536),([5],262144)]
        if repeat%2==0:groups.reverse()
        for buckets,context in groups:
            tasks=[]
            if repeat==1 and context==4096:tasks.append(('large-quality','large','quality',None,64))
            for b in buckets:
                conc=[1,4,8,16,32,64]
                if repeat%2==0:conc.reverse()
                tasks += [(f'large-r{repeat}-P{b}-C{c}','large','performance',f'P{b}',c) for c in conc]
            await group(f'large-r{repeat}-ctx{context}-seq64-budget{budget}',context,64,budget,tasks)
        summarize()
    results=summarize()
    counts=dict(collections.Counter(r['status'] for r in results.values()))
    save(ROOT/'step5.complete.json',{'utc':now(),'statuses':counts,'performance_cells':90,
        'quality_result':results.get('large-quality'),'budget':budget})
    status('all_five_steps_finished',statuses=counts)


if __name__=='__main__':
    try:asyncio.run(main())
    except Exception:
        ROOT.mkdir(parents=True,exist_ok=True)
        save(ROOT/'fatal.json',{'utc':now(),'traceback':traceback.format_exc()})
        raise
