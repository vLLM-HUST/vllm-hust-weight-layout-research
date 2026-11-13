"""Compare mixed Torch/Triton producer ordering with known output values."""
import argparse
import inspect
import json
from pathlib import Path
import torch
import torch_npu
from vllm_ascend.ops.triton.batch_invariant import matmul

parser=argparse.ArgumentParser(); parser.add_argument('--ordered',action='store_true'); args=parser.parse_args()
if args.ordered:
    from triton.backends.ascend import backend_register
    def stream(device):
        if device is None: device=torch.npu.current_device()
        return torch_npu._C._npu_getCurrentRawStream(device)
    backend_register.backend_strategy_registry._get_instance().strategies['torch_npu']['get_current_stream']=stream
namespace=dict(matmul.__dict__)
exec(compile(inspect.getsource(matmul.linear_persistent).replace('torch.zeros(', 'torch.empty('),'<queue-empty>','exec'),namespace)
empty=namespace['linear_persistent']
torch.use_deterministic_algorithms(True,warn_only=True)
rows=[]
for device in range(torch.npu.device_count()):
    torch.npu.set_device(device)
    for m,n,k in [(1,32,2048),(20,6144,2048),(20,256,2048)]:
        for name,function in [('empty',empty),('zeros',matmul.linear_persistent)]:
            for repeat in range(10):
                # No intervening synchronization; producer and allocation are queued.
                x=torch.zeros(m,k,device='npu',dtype=torch.bfloat16); x.add_(1)
                weight=torch.ones(n,k,device='npu',dtype=torch.bfloat16)
                output=function(x,weight)
                torch.npu.synchronize()
                actual=output.float().cpu()
                row=dict(device=device,shape=[m,n,k],variant=name,repeat=repeat,finite=bool(actual.isfinite().all()),correct=bool((actual==k).all()),max_abs=float((actual-k).abs().max()))
                rows.append(row)
    print(json.dumps(dict(device=device,ordered=args.ordered,failures=sum(not r['correct'] for r in rows if r['device']==device))),flush=True)
path=Path(__file__).resolve().parent/'results'/('queue-order-'+('ordered' if args.ordered else 'default')+'.json')
path.write_text(json.dumps(rows,indent=2))
