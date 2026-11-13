"""Test whether the persistent kernel overwrites prefilled output storage."""
import inspect
import argparse
import json
from pathlib import Path
import torch
import torch_npu
from vllm_ascend.ops.triton.batch_invariant import matmul

rows=[]
parser=argparse.ArgumentParser(); parser.add_argument('--ordered',action='store_true'); args=parser.parse_args()
if args.ordered:
    from triton.backends.ascend import backend_register
    def stream(device):
        return torch_npu._C._npu_getCurrentRawStream(torch.npu.current_device() if device is None else device)
    backend_register.backend_strategy_registry._get_instance().strategies['torch_npu']['get_current_stream']=stream
torch.manual_seed(42)
for m in (1,20,32):
    x=torch.randn(m,2048,device='npu',dtype=torch.bfloat16)
    weight=torch.randn(6144,2048,device='npu',dtype=torch.bfloat16)*.1
    reference=matmul.linear_persistent(x,weight)
    torch.npu.synchronize()
    for sentinel in (7.,float('nan')):
        namespace=dict(matmul.__dict__,OUTPUT_SENTINEL=sentinel)
        source=inspect.getsource(matmul.linear_persistent).replace('torch.zeros((M, N),','torch.full((M, N), OUTPUT_SENTINEL,')
        exec(compile(source,'<poison-output>','exec'),namespace)
        output=namespace['linear_persistent'](x,weight)
        torch.npu.synchronize()
        a,b=output.float().cpu(),reference.float().cpu()
        row=dict(m=m,sentinel=str(sentinel),finite=bool(a.isfinite().all()),nan_count=int(a.isnan().sum()),max_abs=float((a-b).abs().max()),equal=bool(torch.equal(a,b)))
        rows.append(row); print(json.dumps(row),flush=True)
Path(__file__).resolve().parent.joinpath('results/output-poison-probe'+('-ordered' if args.ordered else '')+'.json').write_text(json.dumps(rows,indent=2))
