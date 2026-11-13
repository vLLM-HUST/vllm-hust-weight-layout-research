"""Isolated task-group update probe; never starts a model service."""
import argparse
import importlib.metadata as md
import json
import time
import traceback
from pathlib import Path

import torch
import torch_npu
import triton
from candidate_swiglu_kernel import _linear_swiglu_kernel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['control', 'kernel', 'topology', 'fusion', 'triton'], required=True)
    p.add_argument('--m', type=int, default=7)
    p.add_argument('--n', type=int, default=256)
    p.add_argument('--k', type=int, default=2048)
    p.add_argument('--dtype', choices=['bf16', 'fp16'], default='bf16')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    record = dict(mode=args.mode, shape=[args.m,args.n,args.k], dtype=args.dtype,
                  capture_count=0, checks=[], updates=[], evidence='isolated-operator-task-group')
    def save():
        Path(args.output).write_text(json.dumps(record, indent=2))
    def stage(name):
        record['stage'] = name
        save()
        print(name, flush=True)
    try:
        stage('imported')
        record['versions'] = {n: md.version(n) for n in ['torch','torch-npu','triton-ascend']}
        torch.npu.set_device(0)
        torch.manual_seed(20261015)
        dt = torch.bfloat16 if args.dtype == 'bf16' else torch.float16
        m,n,k = args.m,args.n,args.k
        if args.mode not in ('fusion','triton'):
            x = torch.randn(64, device='npu')
            mid, y, z = torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)
            def run(which):
                if args.mode == 'control':
                    torch.add(x, 2 if which == 'baseline' else 3, out=y)
                elif args.mode == 'kernel':
                    if which == 'baseline': torch.add(x,2,out=y)
                    else: torch.mul(x,3,out=y)
                elif which == 'baseline':
                    torch.add(x, 2, out=mid)
                    torch.mul(mid, 3, out=y)
                else:
                    torch.add(x, 2, alpha=3, out=y)
            def expected(which):
                if args.mode == 'control': return x + (2 if which == 'baseline' else 3)
                if args.mode == 'kernel': return x+2 if which == 'baseline' else x*3
                return (x+2)*3 if which == 'baseline' else x+6
        else:
            x = (torch.randn(m,k)*.2).to('npu',dt)
            w = (torch.randn(2*n,k)*.2).to('npu',dt)
            w2 = -w
            y = torch.empty(m,n,device='npu',dtype=dt)
            z = torch.empty_like(y)
            # Keep every native allocation alive through the graph lifetime.
            keepalive=[]
            from vllm_ascend.ops.triton.batch_invariant.matmul import linear_persistent
            def run(which):
                if which == 'baseline' and args.mode == 'fusion':
                    mm=linear_persistent(x,w)
                    act=torch_npu.npu_swiglu(mm)
                    # DMA memcpy is not packable in this backend's task group;
                    # use an AICore identity to adapt native allocating outputs.
                    torch.add(act,0,out=y)
                    keepalive.extend([mm,act])
                else:
                    bm=min(triton.next_power_of_2(m),128)
                    bn=64 if args.mode=='triton' and which=='candidate' else 128
                    selected_w=w2 if args.mode=='triton' and which=='candidate' else w
                    _linear_swiglu_kernel[(triton.cdiv(m,bm)*triton.cdiv(n,bn),)](
                        x,selected_w,y,m,n,k,*x.stride(),*selected_w.stride(),*y.stride(),
                        BLOCK_M=bm,BLOCK_N=bn,BLOCK_K=256,NUM_BLOCKS_N=triton.cdiv(n,bn))
            def expected(which):
                # Independent materialized reference, not the updated graph.
                selected_w=w2 if args.mode=='triton' and which=='candidate' else w
                return torch_npu.npu_swiglu(torch.nn.functional.linear(x,selected_w))
        stage('warmup')
        for which in ['baseline','candidate']:
            for _ in range(3): run(which)
        torch.npu.synchronize()
        if args.mode in ('fusion','triton'):
            for which in ['baseline','candidate']:
                ref=expected(which)
                run(which); torch.npu.synchronize()
                assert torch.allclose(y.float(),ref.float(),rtol=.03 if dt==torch.bfloat16 else .005,
                                      atol=.05 if dt==torch.bfloat16 else .02), 'pre-capture numeric gate failed'
        graph=torch.npu.NPUGraph()
        event=torch.npu.ExternalEvent()
        update_stream=torch.npu.Stream()
        stage('capture')
        with torch.npu.graph(graph):
            capture_stream=torch.npu.current_stream()
            event.wait(capture_stream)
            event.reset(capture_stream)
            torch.npu.graph_task_group_begin(capture_stream)
            run('baseline')
            handle=torch.npu.graph_task_group_end(capture_stream)
            # Public suffix outside the mutable task group.
            torch.add(y,1,out=z)
        record['capture_count']=1
        record['addresses']={'input':x.data_ptr(),'output':y.data_ptr(),'suffix':z.data_ptr()}
        for index,which in enumerate(['baseline','candidate','baseline','candidate','baseline']):
            stage('update-'+str(index)+'-'+which)
            torch.npu.synchronize()
            start=time.perf_counter()
            with torch.npu.stream(update_stream):
                torch.npu.graph_task_update_begin(update_stream,handle)
                run(which)
                torch.npu.graph_task_update_end(update_stream)
                event.record(update_stream)
            update_stream.synchronize()
            record['updates'].append({'variant':which,'host_update_sync_ms':(time.perf_counter()-start)*1000})
            for repeat in range(3):
                if repeat:
                    x.copy_(torch.randn_like(x)*.2)
                ref=expected(which)
                # Poison the fixed output to reject stale/eager-only writes.
                y.fill_(float('nan')); z.fill_(float('nan'))
                torch.npu.synchronize()
                if repeat:
                    with torch.npu.stream(update_stream): event.record(update_stream)
                graph.replay()
                torch.npu.synchronize()
                a,b=y.float().cpu(),ref.float().cpu()
                close=bool(torch.allclose(a,b,rtol=.03 if args.dtype=='bf16' else .005,atol=.05 if args.dtype=='bf16' else .02))
                suffix=bool(torch.equal(z.cpu(),(y+1).cpu()))
                row=dict(index=index,variant=which,repeat=repeat,finite=bool(a.isfinite().all()),
                         close=close,suffix=suffix,max_abs=float((a-b).abs().nan_to_num(posinf=1e30).max()))
                record['checks'].append(row); save()
                assert row['finite'] and close and suffix, 'updated replay correctness failed: '+str(row)
        record['all_pass']=True
        stage('finished')
    except Exception:
        record['all_pass']=False
        record['error']=traceback.format_exc()
        save()
        print(record['error'],flush=True)
        raise


if __name__=='__main__':
    with torch.inference_mode(): main()
