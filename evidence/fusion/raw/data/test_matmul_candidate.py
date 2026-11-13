"""Test explicit dot accumulator candidate against poisoned-output oracles."""
import inspect
import json
from pathlib import Path
import torch
import torch_npu
import candidate_matmul
from vllm_ascend.ops.triton.batch_invariant.matmul import linear_persistent

rows=[]
for m in (1,20,32):
    torch.manual_seed(42)
    x=torch.randn(m,2048,device='npu',dtype=torch.bfloat16)
    w=torch.randn(6144,2048,device='npu',dtype=torch.bfloat16)*.1
    reference=linear_persistent(x,w)
    torch.npu.synchronize()
    for sentinel in (None,7.,float('nan')):
        namespace=dict(candidate_matmul.__dict__,SENTINEL=sentinel)
        source=inspect.getsource(candidate_matmul.linear_persistent)
        if sentinel is not None: source=source.replace('torch.empty((M, N),','torch.full((M, N), SENTINEL,')
        exec(compile(source,'<explicit-accumulator>','exec'),namespace)
        value=namespace['linear_persistent'](x,w)
        torch.npu.synchronize()
        row=dict(m=m,sentinel=str(sentinel),finite=bool(value.isfinite().all()),equal=bool(torch.equal(value,reference)),nan_count=int(value.isnan().sum()))
        rows.append(row); print(json.dumps(row),flush=True)
Path(__file__).resolve().parent.joinpath('results/explicit-accumulator-probe.json').write_text(json.dumps(rows,indent=2))
