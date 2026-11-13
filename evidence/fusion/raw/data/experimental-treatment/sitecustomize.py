"""Isolated treatment selection; does not edit installed source."""
import importlib.abc
import importlib.machinery
import os
import sys

MODE = os.environ.get('OPSET_EXPERIMENT', '')
MATMUL = 'vllm_ascend.ops.triton.batch_invariant.matmul'
BACKEND = 'triton.backends.ascend.backend_register'
TARGETS = {'fusion': ('vllm_ascend.ops.fused_moe.fused_moe_0_23_0',),
           'empty-source': (MATMUL,), 'empty-checked': (MATMUL,),
           'empty-ordered': (MATMUL,BACKEND), 'baseline-ordered': (BACKEND,)}

class Loader(importlib.abc.Loader):
    def __init__(self, original): self.original = original
    def create_module(self, spec): return self.original.create_module(spec)
    def exec_module(self, module):
        self.original.exec_module(module)
        if module.__name__ == BACKEND:
            import inspect
            import torch
            import torch_npu
            registry=module.backend_strategy_registry._get_instance().strategies['torch_npu']
            if '_npu_getCurrentRawStreamNoWait' not in inspect.getsource(registry['get_current_stream']):
                raise RuntimeError('unknown stream backend; refuse queue patch')
            if not hasattr(torch_npu._C,'_npu_getCurrentRawStream'):
                raise RuntimeError('ordered raw stream getter unavailable')
            def ordered(device):
                if device is None: device=torch.npu.current_device()
                return torch_npu._C._npu_getCurrentRawStream(device)
            registry['get_current_stream']=ordered
            print('OPSET_ORDERED_STREAM_INSTALLED',flush=True)
        elif MODE == 'fusion':
            from shared_expert_adapter import install
            install(module)
        elif MODE in ('empty-source', 'empty-checked', 'empty-ordered'):
            import inspect
            import linecache
            import textwrap
            original = module.linear_persistent
            source = textwrap.dedent(inspect.getsource(original))
            if source.count('torch.zeros((M, N)') != 1:
                raise RuntimeError('unknown persistent allocation source')
            source = source.replace('torch.zeros((M, N)', 'torch.empty((M, N)')
            filename = '<opset-linear-persistent-empty-source>'
            linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
            namespace = {}
            exec(compile(source, filename, 'exec'), module.__dict__, namespace)
            candidate = namespace['linear_persistent']
            if MODE == 'empty-checked':
                import json
                import traceback
                import torch
                seen = set()
                def checked(x, weight):
                    output = candidate(x, weight)
                    key=(tuple(x.shape),tuple(weight.shape),tuple(x.stride()),tuple(weight.stride()),str(x.dtype))
                    if x.shape[0] < 256 and key not in seen:
                        seen.add(key)
                        reference = original(x,weight)
                        torch.npu.synchronize()
                        input_finite=bool(torch.isfinite(x).all())
                        output_finite=bool(torch.isfinite(output).all())
                        record=dict(device=str(x.device),shape=key,input_finite=input_finite,weight_finite=bool(torch.isfinite(weight).all()),output_finite=output_finite,reference_finite=bool(torch.isfinite(reference).all()),equal=bool(torch.equal(output,reference)),stack=traceback.format_stack(limit=7))
                        print('OPSET_REAL_VALUE '+json.dumps(record),flush=True)
                        if not output_finite:
                            raise RuntimeError('first observed nonfinite persistent input/output; see OPSET_REAL_VALUE')
                    return output
                module.linear_persistent=checked
            else:
                module.linear_persistent = candidate
            print('OPSET_EMPTY_SOURCE_INSTALLED', flush=True)

class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in TARGETS.get(MODE, ()): return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and spec.loader: spec.loader = Loader(spec.loader)
        return spec

if MODE in TARGETS: sys.meta_path.insert(0, Finder())
