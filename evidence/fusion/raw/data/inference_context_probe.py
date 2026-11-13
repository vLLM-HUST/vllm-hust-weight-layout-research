"""Isolate inference-mode and Tensor-subclass differences from random probes."""
import inspect
import argparse
import json
from pathlib import Path
import torch
import torch_npu
from vllm.model_executor.parameter import ModelWeightParameter
from vllm_ascend.ops.triton.batch_invariant import matmul

root=Path(__file__).resolve().parent
namespace=dict(matmul.__dict__)
parser=argparse.ArgumentParser(); parser.add_argument('--aligned',action='store_true'); args=parser.parse_args()
source=inspect.getsource(matmul.linear_persistent).replace('torch.zeros(', 'torch.empty(')
if args.aligned:
    source=source.replace('    # Calculate number of blocks per dimension (ceil division)', '    BLOCK_M = triton.next_power_of_2(BLOCK_M)\n    BLOCK_N = triton.next_power_of_2(BLOCK_N)\n\n    # Calculate number of blocks per dimension (ceil division)')
exec(compile(source,'<context-empty>','exec'),namespace)
empty=namespace['linear_persistent']
torch.use_deterministic_algorithms(True,warn_only=True)
rows=[]
for inference in (False,True):
    with torch.inference_mode(inference):
        for subclass in (False,True):
            x=torch.randn(20,2048,device='npu',dtype=torch.bfloat16)
            weight=torch.randn(6144,2048,device='npu',dtype=torch.bfloat16)*.1
            if subclass: weight=weight.as_subclass(ModelWeightParameter)
            torch.npu.synchronize()
            for repeat in range(3):
                output=empty(x,weight); torch.npu.synchronize()
                reference=matmul.linear_persistent(x,weight); torch.npu.synchronize()
                row=dict(inference=inference,subclass=subclass,repeat=repeat,finite=bool(output.isfinite().all()),reference_finite=bool(reference.isfinite().all()),equal=bool(torch.equal(output,reference)),nan_count=int(output.isnan().sum()))
                rows.append(row); print(json.dumps(row),flush=True)
root.joinpath('results/inference-context-probe'+('-aligned' if args.aligned else '')+'.json').write_text(json.dumps(rows,indent=2))
