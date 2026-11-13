"""Final fused-SwiGLU full-write and same-input determinism oracle."""
import inspect
import json
from pathlib import Path
import torch
import torch_npu
from vllm_hust_operator_optimizations.optimizations import linear_swiglu_kernel as module

rows=[]
torch.manual_seed(42)
for dtype in (torch.float16,torch.bfloat16):
    for m,n,k in [(1,256,2048),(20,256,2048),(128,256,2048),(7,513,257)]:
        x=(torch.randn(m,k)*.2).to('npu',dtype)
        w=(torch.randn(2*n,k)*.2).to('npu',dtype)
        reference=torch_npu.npu_swiglu(torch.nn.functional.linear(x,w))
        torch.npu.synchronize()
        for sentinel in (7.,float('nan')):
            namespace=dict(module.__dict__,SENTINEL=sentinel)
            source=inspect.getsource(module.linear_swiglu).replace('torch.empty((m, n),','torch.full((m, n), SENTINEL,')
            exec(compile(source,'<fusion-poison>','exec'),namespace)
            values=[namespace['linear_swiglu'](x,w) for _ in range(3)]
            torch.npu.synchronize()
            for value in values:
                assert bool(value.isfinite().all())
                torch.testing.assert_close(value.float().cpu(),reference.float().cpu(),rtol=.03 if dtype==torch.bfloat16 else .005,atol=.05 if dtype==torch.bfloat16 else .02)
            assert all(torch.equal(values[0],value) for value in values[1:])
            row=dict(dtype=str(dtype),shape=[m,n,k],sentinel=str(sentinel),launches=3,passed=True)
            rows.append(row); print(json.dumps(row),flush=True)
Path(__file__).resolve().parent.joinpath('results/fusion-poison-probe.json').write_text(json.dumps(dict(all_pass=True,cases=len(rows),launches=len(rows)*3,rows=rows),indent=2))
