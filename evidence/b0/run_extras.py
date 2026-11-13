"""Separate diagnostic profiling and sustained load; never mix with formal timings."""
import asyncio
import collections
import json
from pathlib import Path
import random
import signal
import time
import traceback

import aiohttp

import diagnostic_runtime as b


async def stability(server, concurrency, duration=900):
    folder=b.ROOT/'stability'/'P2'
    if (folder/'summary.json').exists():return
    folder.mkdir(parents=True,exist_ok=True)
    rows=b.records('large','performance','P2')
    complete=[]
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=concurrency+4),read_bufsize=4<<20) as session:
        warm=await asyncio.gather(*(b.request(session,r,False) for r in rows[:concurrency]))
        b.save(folder/'warmup.json',warm)
        if any(not r['success'] for r in warm):raise RuntimeError('stability warmup failed')
        (folder/'metrics.before.txt').write_text(await b.text_get(session,'/metrics'))
        start=time.perf_counter();counter=0
        out=(folder/'requests.jsonl').open('w')
        async def worker():
            nonlocal counter
            while time.perf_counter()-start<duration:
                k=counter;counter+=1
                row=dict(rows[k%len(rows)]);row['id']=f'stability-{k:07d}-{row["id"]}'
                submitted=time.perf_counter()-start
                arrivals=[0]*15
                def observe(tick,count):
                    window=int((tick-start)//60)
                    if 0<=window<15:arrivals[window]+=count
                r=await b.request(session,row,False,token_observer=observe)
                r.update(submitted_seconds=submitted,completed_seconds=time.perf_counter()-start)
                r['stream_output_tokens_by_window']=arrivals
                complete.append(r);out.write(json.dumps(r)+'\n');out.flush()
                if len(complete)%64==0:
                    b.status('stability_running',elapsed_seconds=time.perf_counter()-start,
                             requests=len(complete),concurrency=concurrency)
        try:await asyncio.gather(*(worker() for _ in range(concurrency)))
        finally:out.close()
        elapsed=time.perf_counter()-start
        (folder/'metrics.after.txt').write_text(await b.text_get(session,'/metrics'))
    good=[r for r in complete if r['success']]
    # Complete one-minute windows; boundary/drain completions are disclosed separately.
    windows=[]
    for i in range(15):
        rs=[r for r in good if i*60<=r['completed_seconds']<(i+1)*60]
        windows.append({'start_seconds':i*60,'end_seconds':(i+1)*60,
                        'requests':len(rs),'output_tokens_per_second':sum(r['output_tokens'] for r in rs)/60,
                        'stream_arrival_output_tokens_per_second':sum(r['stream_output_tokens_by_window'][i] for r in good)/60,
                        'all_received_stream_tokens_per_second':sum(r['stream_output_tokens_by_window'][i] for r in complete)/60,
                        'ttft_seconds':b.describe([r['ttft_seconds'] for r in rs]),
                        'tpot_seconds':b.describe([r['tpot_seconds'] for r in rs if r['tpot_seconds'] is not None])})
    b.save(folder/'summary.json',{'status':'completed','load_submission_seconds':duration,
        'elapsed_including_drain_seconds':elapsed,'concurrency':concurrency,
        'requests':len(complete),'failures':len(complete)-len(good),
        'failure_rate':(len(complete)-len(good))/len(complete),
        'output_tokens_per_second_including_drain':sum(r['output_tokens'] for r in good)/elapsed,
        'ttft_seconds':b.describe([r['ttft_seconds'] for r in good]),
        'tpot_seconds':b.describe([r['tpot_seconds'] for r in good if r['tpot_seconds'] is not None]),
        'windows':windows})


async def profiles(budget, high):
    for bucket,context in [('P2',4096),('P3',16384)]:
        for c in [1,high]:
            name=f'profile-{bucket}-C{c}'
            if (b.ROOT/'profiles'/name/'receipt.json').exists():continue
            server=b.Server(name,context,64,budget)
            try:
                b.status('profile_running',case=name)
                await server.start()
                rows=b.records('large','performance',bucket)[:c]
                async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=c+4),read_bufsize=4<<20) as session:
                    warm=await asyncio.gather(*(b.request(session,r,False) for r in rows))
                    if any(not r['success'] for r in warm):raise RuntimeError('profile warmup failed')
                    async with session.post(b.URL+'/collective_rpc',json={
                        'method':'b0_memory_snapshot','kwargs':{'reset_peak':'true'}},
                        timeout=aiohttp.ClientTimeout(total=120)) as r:
                        b.save(b.ROOT/'profiles'/name/'memory.before.json',{'status':r.status,'body':await r.json()})
                    async with session.post(b.URL+'/start_profile',timeout=aiohttp.ClientTimeout(total=120)) as r:
                        if r.status!=200:raise RuntimeError(await r.text())
                    measured=await asyncio.gather(*(b.request(session,r,False) for r in rows))
                    async with session.post(b.URL+'/stop_profile',timeout=aiohttp.ClientTimeout(total=300)) as r:
                        stop={'status':r.status,'body':await r.text()}
                    async with session.post(b.URL+'/collective_rpc',json={'method':'b0_memory_snapshot'},
                        timeout=aiohttp.ClientTimeout(total=120)) as r:
                        b.save(b.ROOT/'profiles'/name/'memory.after.json',{'status':r.status,'body':await r.json()})
                    b.save(b.ROOT/'profiles'/name/'receipt.json',{
                        'case':name,'profiled_request_count':len(rows),'profile_limit_iterations':24,
                        'stop':stop,'requests':measured,'not_formal_timing':True})
            except Exception as e:
                b.save(b.ROOT/'profiles'/name/'failure.json',{'error':str(e),'traceback':traceback.format_exc()})
            finally:await server.stop()


async def memory_contexts(budget):
    for bucket,context in [('P4',65536),('P5',262144)]:
        target=b.ROOT/'memory'/bucket/'receipt.json'
        if target.exists():continue
        server=b.Server(f'profile-memory-{bucket}',context,64,budget)
        try:
            b.status('memory_diagnostic_running',bucket=bucket)
            await server.start()
            async with aiohttp.ClientSession(read_bufsize=4<<20) as session:
                warm=await b.request(session,b.records('large','performance',bucket)[0],False)
                if not warm['success']:raise RuntimeError(f'memory warmup failed: {warm}')
                async with session.post(b.URL+'/collective_rpc',json={'method':'b0_memory_snapshot'},
                    timeout=aiohttp.ClientTimeout(total=120)) as r:
                    if r.status!=200:raise RuntimeError(await r.text())
                    b.save(target,{'bucket':bucket,'context':context,'snapshot':await r.json(),
                                   'scope':'memory-only diagnostic; no performance score'})
        except Exception as e:
            b.save(target.parent/'failure.json',{'error':str(e),'traceback':traceback.format_exc()})
        finally:await server.stop()


async def main():
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM,asyncio.current_task().cancel)
    if not (b.ROOT/'step5.complete.json').exists():raise RuntimeError('formal B0 campaign not finished')
    cfg=json.loads((b.ROOT/'baseline-frozen.json').read_text())
    results=b.summarize()
    eligible=[c for c in [1,4,8,16,32,64] if all(
        results.get(f'large-r{r}-P2-C{c}',{}).get('status')=='completed' for r in [1,2,3])]
    if not eligible:raise RuntimeError('no successful P2 concurrency for sustained load')
    # Highest successful tested concurrency, same frozen seq/budget/context contract.
    high=max(eligible)
    if not (b.ROOT/'stability/P2/summary.json').exists():
        server=b.Server('stability-P2',4096,64,cfg['budget'])
        try:
            b.status('stability_starting',concurrency=high)
            await server.start()
            await stability(server,high)
        finally:await server.stop()
    await profiles(cfg['budget'],high)
    await memory_contexts(cfg['budget'])
    required=[b.ROOT/'stability/P2/summary.json']+[
        b.ROOT/'profiles'/f'profile-{bucket}-C{c}'/'receipt.json'
        for bucket in ['P2','P3'] for c in [1,high]]+[
        b.ROOT/'memory'/bucket/'receipt.json' for bucket in ['P4','P5']]
    missing=[str(p) for p in required if not p.exists()]
    if missing:raise RuntimeError(f'extra measurements incomplete: {missing}')
    b.save(b.ROOT/'extras.complete.json',{'completed_utc':b.now(),'high_concurrency':high})
    b.save(b.ROOT/'final-npu-state.json',b.run(['npu-smi','info']))
    b.status('formal_and_extra_tests_finished')


if __name__=='__main__':asyncio.run(main())
