"""Owned, bounded TP2 Graph lifecycle. Warm client timing and raw evidence."""
import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent

def save(path, obj): path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')

def request(base, prompt, budget=64, streaming=False):
    payload=dict(model='research-model',messages=[dict(role='user',content=prompt)],temperature=0,seed=0,max_tokens=budget,chat_template_kwargs=dict(enable_thinking=False),stream=streaming)
    if streaming: payload['stream_options']=dict(include_usage=True)
    req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    start=time.perf_counter(); first=None; chunks=[]; usage=None
    with urllib.request.urlopen(req,timeout=180) as response:
        if not streaming:
            obj=json.load(response)
            return dict(request=payload,response=obj,elapsed_s=time.perf_counter()-start,text=obj['choices'][0]['message']['content'])
        for line in response:
            if not line.startswith(b'data: '): continue
            body=line[6:].strip()
            if body==b'[DONE]': break
            item=json.loads(body)
            chunks.append(item)
            if item.get('usage'): usage=item['usage']
            if any(c.get('delta',{}).get('content') for c in item.get('choices',[])) and first is None:
                first=time.perf_counter()
    stop=time.perf_counter()
    text=''.join(c.get('delta',{}).get('content') or '' for item in chunks for c in item.get('choices',[]))
    count=(usage or {}).get('completion_tokens',0)
    return dict(request=payload,chunks=chunks,usage=usage,text=text,elapsed_s=stop-start,ttft_ms=(first-start)*1000 if first else None,tpot_ms=(stop-first)*1000/(count-1) if first and count>1 else None,output_tokens_s=count/(stop-start),completion_tokens=count)

def post(base, path):
    with urllib.request.urlopen(urllib.request.Request(base+path,data=b'',method='POST'),timeout=120) as response:
        return response.status

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=['baseline','fusion','empty-source','empty-plugin','empty-eager'],required=True)
    parser.add_argument('--round',type=int,default=0)
    parser.add_argument('--diagnostic',action='store_true')
    parser.add_argument('--profile',action='store_true')
    args=parser.parse_args()
    run_id=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+args.variant+'-r'+str(args.round)
    run=ROOT/'results'/run_id; run.mkdir()
    if shutil.disk_usage(ROOT).free < 10*1024**3: raise RuntimeError('10GiB storage gate')
    env=os.environ.copy()
    env.update(VLLM_PLUGINS='ascend',VLLM_BATCH_INVARIANT='1',TORCH_DEVICE_BACKEND_AUTOLOAD='0',VLLM_CACHE_ROOT=str(ROOT/'cache'/'diagnostic-suite'/args.variant),OMP_NUM_THREADS='1',OPSET_EXPERIMENT=args.variant)
    env.pop('VLLM_HUST_OPERATOR_OPTIMIZATIONS',None)
    env['PYTHONPATH']=str(ROOT/'experimental-treatment')+os.pathsep+env.get('PYTHONPATH','')
    if args.variant in ('empty-plugin','empty-eager'):
        env['VLLM_PLUGINS']+=',operator_optimizations'
        env['VLLM_HUST_OPERATOR_OPTIMIZATIONS']='persistent-matmul-empty'
    argv=[sys.executable,'-m','vllm.entrypoints.openai.api_server','--model','/data/shared_upload/models/vspec/Qwen3.5-35B-A3B','--served-model-name','research-model','--host','127.0.0.1','--port','18080','--dtype','bfloat16','--tensor-parallel-size','2','--max-model-len','4096','--max-num-seqs','4','--max-num-batched-tokens','4096','--gpu-memory-utilization','0.85','--no-enable-prefix-caching','--limit-mm-per-prompt','{"image":0,"video":0}','--compilation-config','{"cudagraph_capture_sizes":[1,2,4]}']
    if args.variant=='empty-eager': argv+=['--enforce-eager']
    if args.profile:
        argv+=['--profiler-config',json.dumps(dict(profiler='torch',torch_profiler_dir=str(run/'profile'),torch_profiler_record_shapes=True,torch_profiler_with_stack=False,ignore_frontend=True))]
    state=dict(run_id=run_id,variant=args.variant,round=args.round,status='starting',evidence_level='development-matched',argv=argv)
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'experimental-treatment').glob('*.py')}
    state['candidate_hashes']=hashes
    state['packages']={name:importlib.metadata.version(name) for name in ('torch','torch-npu','vllm','vllm-ascend','triton-ascend','vllm-hust-opset')}
    save(run/'manifest.json',state)
    save(ROOT/'suite-current.json',state)
    service=None; base='http://127.0.0.1:18080'; started=time.monotonic()
    try:
        with (run/'server.log').open('w') as log:
            service=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(run/'owned-process.json',dict(pid=service.pid,pgid=service.pid))
            deadline=time.monotonic()+900
            while time.monotonic()<deadline:
                if service.poll() is not None: raise RuntimeError('server exited '+str(service.returncode))
                tail=(run/'server.log').read_text(errors='replace')[-12000:]
                if 'WorkerProc hit an exception' in tail or 'Engine core initialization failed' in tail: raise RuntimeError('worker startup failure')
                try:
                    with urllib.request.urlopen(base+'/health',timeout=2) as response:
                        if response.status==200: break
                except Exception: time.sleep(3)
            else: raise TimeoutError('startup deadline')
            state['startup_s']=time.monotonic()-started
            smoke=request(base,'请简要介绍 Transformer 的核心原理。',128)
            save(run/'smoke.json',smoke)
            text=smoke['text']; state['smoke_text']=text
            if not text or text.count('!')>len(text)*.5: raise RuntimeError('degenerate smoke output')
            if not args.diagnostic:
                table=__import__('pyarrow.parquet',fromlist=['read_table']).read_table(ROOT/'datasets/mmlu-pro/test-00000-of-00001.parquet').to_pylist()[:20]
                quality=[]
                for row in table:
                    prompt='请回答以下选择题，仅输出答案字母。\n'+row['question']+'\n'+'\n'.join(chr(65+i)+'. '+option for i,option in enumerate(row['options']))
                    item=request(base,prompt,16)
                    match=re.search(r'\b([A-J])\b',item['text'])
                    item.update(question_id=row.get('question_id'),expected=row['answer'],predicted=match.group(1) if match else None)
                    quality.append(item)
                save(run/'quality.json',quality)
                state['quality_accuracy']=sum(q['predicted']==q['expected'] for q in quality)/len(quality)
                state['quality_samples']=len(quality)
                workloads={'short':'请简要介绍 Transformer 的核心原理。','long':('背景材料：Transformer 利用自注意力处理序列，位置编码表示顺序，多头注意力提取不同关系。\n'*40)+'请概括 Transformer 的核心原理。'}
                measurements=[]
                for name,prompt in workloads.items():
                    for _ in range(3): request(base,prompt,64,True)
                    for sample in range(5):
                        item=request(base,prompt,64,True); item.update(workload=name,sample=sample,concurrency=1); measurements.append(item)
                    # A separate concurrency-2 sample group; do not mix with C1.
                    for sample in range(3):
                        start=time.perf_counter()
                        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                            pair=list(executor.map(lambda _: request(base,prompt,64,True),range(2)))
                        elapsed=time.perf_counter()-start
                        measurements.append(dict(workload=name,sample=sample,concurrency=2,requests=pair,elapsed_s=elapsed,aggregate_tokens_s=sum(x['completion_tokens'] for x in pair)/elapsed))
                save(run/'measurements.json',measurements)
                if args.profile:
                    state['profile_start']=post(base,'/start_profile')
                    save(run/'profile-request.json',request(base,workloads['short'],32))
                    state['profile_stop']=post(base,'/stop_profile')
                    time.sleep(5)
            state['status']='completed'
    except Exception as error:
        state.update(status='failed',error=type(error).__name__+': '+str(error))
    finally:
        if service:
            try: os.killpg(service.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: service.wait(timeout=25)
            except subprocess.TimeoutExpired:
                os.killpg(service.pid,signal.SIGKILL); service.wait(timeout=10)
            state['server_exit_code']=service.returncode
        state['elapsed_s']=time.monotonic()-started
        save(run/'result.json',state); save(ROOT/'suite-current.json',state)
        print(json.dumps(state,ensure_ascii=False),flush=True)
    return 0 if state['status']=='completed' else 1

if __name__=='__main__': raise SystemExit(main())
