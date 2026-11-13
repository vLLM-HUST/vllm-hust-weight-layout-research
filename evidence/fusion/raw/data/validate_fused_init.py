"""Reuse the full correctness matrix against the fused-init candidate."""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import torch
import torch_npu
import candidate_matmul
from vllm_ascend.ops.triton.batch_invariant import matmul

root=Path(__file__).resolve().parent
os.environ.pop('VLLM_BATCH_INVARIANT',None)
os.environ.pop('VLLM_HUST_OPERATOR_OPTIMIZATIONS',None)
matmul.linear_persistent=candidate_matmul.linear_persistent
torch.use_deterministic_algorithms(True,warn_only=True)
buffer=io.StringIO()
with contextlib.redirect_stdout(buffer):
    runpy.run_path(str(root/'sources/opset/experiments/persistent_matmul_empty/correctness.py'),run_name='__main__')
raw=buffer.getvalue()
root.joinpath('results/fused-init-correctness.log').write_text(raw)
summary=next(json.loads(line) for line in reversed(raw.splitlines()) if line.startswith('{"timestamp":'))
graph_rows=[]
for device in range(torch.npu.device_count()):
    torch.npu.set_device(device)
    for dtype in (torch.float16,torch.bfloat16,torch.float32):
        x=(torch.randn(20,2048)*.2).to('npu',dtype)
        w=(torch.randn(6144,2048)*.2).to('npu',dtype)
        for _ in range(3): candidate_matmul.linear_persistent(x,w)
        torch.npu.synchronize()
        graph=torch.npu.NPUGraph()
        with torch.npu.graph(graph): out=candidate_matmul.linear_persistent(x,w)
        for repeat in range(3):
            x.copy_((torch.randn(20,2048)*.2).to('npu',dtype))
            graph.replay(); torch.npu.synchronize()
            reference=torch.nn.functional.linear(x,w)
            torch.testing.assert_close(out.float().cpu(),reference.float().cpu(),atol=.05 if dtype==torch.bfloat16 else .02,rtol=.03 if dtype==torch.bfloat16 else .005)
            assert bool(out.isfinite().all())
            graph_rows.append(dict(device=device,dtype=str(dtype),repeat=repeat,passed=True))
        del graph
root.joinpath('results/fused-init-correctness.json').write_text(json.dumps(dict(matrix=summary,graph=graph_rows,all_pass=True),indent=2))
print(json.dumps(dict(all_pass=True,matrix_cases=summary['cases'],matrix_launches=summary['launches'],graph_comparisons=len(graph_rows))),flush=True)
