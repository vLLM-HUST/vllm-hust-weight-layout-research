"""Health and fail-closed checks, no model service or device resets."""
import importlib
import json
from pathlib import Path
import torch
import torch_npu
from vllm_hust_operator_optimizations.optimizations.persistent_matmul_empty import PersistentMatmulEmpty

rows=[]
for device in range(torch.npu.device_count()):
    torch.npu.set_device(device)
    value=torch.ones(4,device='npu')+1
    torch.npu.synchronize()
    assert torch.equal(value.cpu(),torch.full((4,),2.))
    rows.append(dict(device=device,passed=True))
module=importlib.import_module('vllm_ascend.ops.triton.batch_invariant.matmul')
original=module.linear_persistent
result=PersistentMatmulEmpty().apply(module)
assert result.state=='blocked'
assert module.linear_persistent is original
record=dict(devices=rows,empty_state=result.state,original_preserved=True)
from vllm_hust_operator_optimizations.optimizations.shared_expert_swiglu import install_shared_expert_fusion
shared=importlib.import_module('vllm_ascend.ops.fused_moe.fused_moe_0_23_0')
shared_result=install_shared_expert_fusion(shared)
assert shared_result.state=='applied'
record['shared_source_guard']='passed'
Path(__file__).resolve().parent.joinpath('results/repaired-plugin-health.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
