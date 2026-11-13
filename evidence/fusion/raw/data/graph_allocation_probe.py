"""Bounded direct NPU Graph replay test; no model service or global patch."""
import json
import argparse
from pathlib import Path
import torch
import torch_npu
from vllm_ascend.ops.triton.batch_invariant import matmul

original = matmul.linear_persistent
parser = argparse.ArgumentParser()
parser.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
parser.add_argument('--deterministic', action='store_true')
args = parser.parse_args()
dtype = getattr(torch, args.dtype)
if args.deterministic:
    torch.use_deterministic_algorithms(True, warn_only=True)
source = __import__('inspect').getsource(original).replace('torch.zeros(', 'torch.empty(')
namespace = dict(matmul.__dict__)
exec(compile(source, '<empty-probe>', 'exec'), namespace)
candidate = namespace['linear_persistent']
rows = []
torch.manual_seed(42)
for m, n, k in [(1, 1, 2048), (1, 32, 2048), (1, 256, 2048), (1, 2048, 2048), (4, 6144, 2048), (32, 2048, 2048), (1, 124416, 2048)]:
    x = torch.randn(m, k, device='npu', dtype=dtype) * .1
    w = torch.randn(n, k, device='npu', dtype=dtype) * .1
    reference = torch.nn.functional.linear(x, w).float().cpu()
    for name, function in [('zeros', original), ('empty', candidate)]:
        for _ in range(3):
            function(x, w)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            out = function(x, w)
        for replay in range(3):
            graph.replay()
            torch.npu.synchronize()
            actual = out.float().cpu()
            row = dict(shape=[m,n,k], variant=name, replay=replay, finite=bool(actual.isfinite().all()), max_abs=float((actual-reference).abs().max()), close=bool(torch.allclose(actual,reference,atol=.05,rtol=.03)))
            rows.append(row)
            print(json.dumps(row), flush=True)
        del graph
tag = args.dtype + ('-deterministic' if args.deterministic else '')
Path(__file__).resolve().parent.joinpath('results/graph-allocation-probe-' + tag + '.json').write_text(json.dumps(rows, indent=2))
