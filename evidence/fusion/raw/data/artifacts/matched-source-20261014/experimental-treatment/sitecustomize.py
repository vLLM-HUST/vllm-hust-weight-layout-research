"""Isolated treatment selection; does not edit installed source."""
import importlib.abc
import importlib.machinery
import os
import sys

MODE = os.environ.get('OPSET_EXPERIMENT', '')
TARGETS = {'fusion': 'vllm_ascend.ops.fused_moe.fused_moe_0_23_0',
           'empty-source': 'vllm_ascend.ops.triton.batch_invariant.matmul'}

class Loader(importlib.abc.Loader):
    def __init__(self, original): self.original = original
    def create_module(self, spec): return self.original.create_module(spec)
    def exec_module(self, module):
        self.original.exec_module(module)
        if MODE == 'fusion':
            from shared_expert_adapter import install
            install(module)
        elif MODE == 'empty-source':
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
            module.linear_persistent = namespace['linear_persistent']
            print('OPSET_EMPTY_SOURCE_INSTALLED', flush=True)

class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGETS.get(MODE): return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and spec.loader: spec.loader = Loader(spec.loader)
        return spec

if MODE in TARGETS: sys.meta_path.insert(0, Finder())
