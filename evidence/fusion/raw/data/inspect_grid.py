"""Compare requested grid with compiler metadata and visible device properties."""
import json
from pathlib import Path
import torch
import torch_npu
from vllm.triton_utils import triton
from vllm_ascend.ops.triton.batch_invariant.matmul import linear_persistent_kernel

x=torch.ones(20,2048,device='npu',dtype=torch.bfloat16)
w=torch.ones(6144,2048,device='npu',dtype=torch.bfloat16)
out=torch.zeros(20,6144,device='npu',dtype=torch.bfloat16)
props=triton.runtime.driver.active.utils.get_device_properties(torch.npu.current_device())
kernel=linear_persistent_kernel[(24,)](a_ptr=x,b_ptr=w,c_ptr=out,M=20,N=6144,K=2048,stride_am=2048,stride_ak=1,stride_bn=2048,stride_bk=1,stride_cm=6144,stride_cn=1,BLOCK_M=20,BLOCK_N=256,BLOCK_K=256,NUM_BLOCKS_M=1,NUM_BLOCKS_N=24,GRID_SIZE=24)
torch.npu.synchronize()
record=dict(device_properties=props,metadata=str(kernel.metadata),missing_columns=(out[0]==0).nonzero().flatten().cpu().tolist())
Path(__file__).resolve().parent.joinpath('results/grid-inspection.json').write_text(json.dumps(record,indent=2,default=str))
print(json.dumps(record,default=str),flush=True)
