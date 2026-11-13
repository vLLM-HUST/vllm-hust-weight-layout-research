"""Graph replay correctness and paired warm operator timing; no service."""
import json
import statistics
import time
from pathlib import Path
import torch
import torch_npu
from candidate_swiglu_kernel import linear_swiglu
from vllm_ascend.ops.triton.batch_invariant.matmul import linear_persistent

root = Path(__file__).resolve().parent
results = []
timing = []
torch.manual_seed(20261014)
for dtype in (torch.float16, torch.bfloat16):
    for m,n,k in [(1,256,2048),(2,256,2048),(4,256,2048),(7,256,2048),(32,256,2048),(128,256,2048),(7,513,257)]:
        x = (torch.randn(m,k) * .2).to('npu', dtype)
        w = (torch.randn(2*n,k) * .2).to('npu', dtype)
        baseline = lambda: torch_npu.npu_swiglu(linear_persistent(x,w))
        candidate = lambda: linear_swiglu(x,w)
        for function in (baseline,candidate):
            for _ in range(5): function()
        torch.npu.synchronize()
        gb, gc = torch.npu.NPUGraph(), torch.npu.NPUGraph()
        with torch.npu.graph(gb): reference = baseline()
        with torch.npu.graph(gc): output = candidate()
        for repeat in range(3):
            x.copy_((torch.randn(m,k) * .2).to('npu',dtype))
            gb.replay(); gc.replay(); torch.npu.synchronize()
            a,b = output.float().cpu(), reference.float().cpu()
            abs_error = (a-b).abs()
            close = bool(torch.allclose(a,b,rtol=.03 if dtype==torch.bfloat16 else .005,atol=.05 if dtype==torch.bfloat16 else .02))
            results.append(dict(dtype=str(dtype),shape=[m,n,k],repeat=repeat,finite=bool(a.isfinite().all()),close=close,max_abs=float(abs_error.max()),mean_abs=float(abs_error.mean()),exact_fraction=float((a==b).float().mean())))
        for repeat in range(30):
            order = [('baseline',gb),('candidate',gc)]
            if repeat % 2: order.reverse()
            for name,graph in order:
                torch.npu.synchronize(); start=time.perf_counter()
                for _ in range(20): graph.replay()
                torch.npu.synchronize()
                timing.append(dict(dtype=str(dtype),shape=[m,n,k],variant=name,repeat=repeat,latency_ms=(time.perf_counter()-start)*1000/20))
        del gb,gc
        print(json.dumps(results[-1]),flush=True)
summary=[]
for dtype in (str(torch.float16),str(torch.bfloat16)):
    for shape in sorted({tuple(x['shape']) for x in timing}):
        row=dict(dtype=dtype,shape=shape)
        for variant in ('baseline','candidate'):
            values=[x['latency_ms'] for x in timing if x['dtype']==dtype and tuple(x['shape'])==shape and x['variant']==variant]
            row[variant]=dict(mean_ms=statistics.mean(values),sd_ms=statistics.stdev(values),variance_ms2=statistics.variance(values),samples=len(values))
        row['latency_reduction_pct']=(1-row['candidate']['mean_ms']/row['baseline']['mean_ms'])*100
        summary.append(row)
record=dict(correctness=results,timing=timing,summary=summary,all_pass=all(x['finite'] and x['close'] for x in results),evidence_level='operator-graph-only')
root.joinpath('results/shared-rounded-validation.json').write_text(json.dumps(record,indent=2))
print(json.dumps(dict(all_pass=record['all_pass'],comparisons=len(results))),flush=True)
raise SystemExit(0 if record['all_pass'] else 1)
