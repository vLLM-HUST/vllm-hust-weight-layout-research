"""Real chat-prefill length, both TP devices, production widths and strides."""
import inspect
import json
from pathlib import Path
import torch
import torch_npu
from transformers import AutoTokenizer
from vllm_ascend.ops.triton.batch_invariant import matmul

root=Path(__file__).resolve().parent
tokenizer=AutoTokenizer.from_pretrained('/data/shared_upload/models/vspec/Qwen3.5-35B-A3B')
tokens=tokenizer.apply_chat_template([dict(role='user',content='请简要介绍 Transformer 的核心原理。')],tokenize=True,add_generation_prompt=True,enable_thinking=False)
if hasattr(tokens, 'keys'): tokens = tokens['input_ids']
if tokens and isinstance(tokens[0], list): tokens = tokens[0]
m=len(tokens)
namespace=dict(matmul.__dict__)
exec(compile(inspect.getsource(matmul.linear_persistent).replace('torch.zeros(', 'torch.empty('),'<shape-empty>','exec'),namespace)
candidate=namespace['linear_persistent']
torch.use_deterministic_algorithms(True,warn_only=True)
rows=[]
for device in range(torch.npu.device_count()):
    torch.npu.set_device(device)
    for dtype in (torch.bfloat16,torch.float32):
        for count,n,k in [(m,6144,2048),(m,32,2048),(m,256,2048),(m,2048,2048),(1,32,2048),(1,256,2048)]:
            torch.manual_seed(20261014)
            x=torch.randn(count,k,device='npu',dtype=dtype)
            w=torch.randn(n,k,device='npu',dtype=dtype)*.1
            a=matmul.linear_persistent(x,w)
            b=candidate(x,w)
            ref=torch.nn.functional.linear(x,w)
            torch.npu.synchronize()
            ac,bc,rc=a.float().cpu(),b.float().cpu(),ref.float().cpu()
            row=dict(device=device,dtype=str(dtype),shape=[count,n,k],zeros_finite=bool(ac.isfinite().all()),empty_finite=bool(bc.isfinite().all()),equal=bool(torch.equal(ac,bc)),zeros_vs_native=float((ac-rc).abs().max()),empty_vs_zeros=float((bc-ac).abs().max()))
            rows.append(row); print(json.dumps(row),flush=True)
            del x,w,a,b,ref
root.joinpath('results/production-shape-probe-m'+str(m)+'.json').write_text(json.dumps(dict(prompt_tokens=m,rows=rows),indent=2))
