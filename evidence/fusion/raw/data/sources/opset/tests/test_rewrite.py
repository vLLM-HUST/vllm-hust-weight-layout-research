import linecache
import types

import pytest

from vllm_hust_operator_optimizations.optimizations import PersistentMatmulEmpty


@pytest.fixture(autouse=True)
def isolate_rewrite_from_installed_backend(monkeypatch):
    # These are AST unit tests, not qualification of the machine's NPU runtime.
    monkeypatch.setattr(
        "vllm_hust_operator_optimizations.optimizations.persistent_matmul_empty.metadata.version",
        lambda name: {"triton-ascend": "3.5.0", "torch-npu": "2.10.0"}[name],
    )


class FakeTorch:
    @staticmethod
    def zeros(shape, **kwargs):
        return ("zeros", shape, kwargs)

    @staticmethod
    def empty(shape, **kwargs):
        return ("empty", shape, kwargs)


class FakeKernel:
    def __getitem__(self, grid):
        return lambda *args, **kwargs: None


def make_module(allocation: str = "zeros") -> types.ModuleType:
    name = "fake_persistent_matmul"
    filename = f"<{name}_{allocation}>"
    source = f"""
def linear_persistent(x, y):
    M, K = x.shape
    N, _ = y.shape
    output = torch.{allocation}((M, N), dtype=x.dtype, device=x.device)
    linear_persistent_kernel[(1,)](x, y, output)
    return output
"""
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    module = types.ModuleType(name)
    module.__dict__.update(torch=FakeTorch, linear_persistent_kernel=FakeKernel())
    exec(compile(source, filename, "exec"), module.__dict__)  # noqa: S102
    return module


def test_persistent_matmul_replaces_only_allocation():
    module = make_module()
    result = PersistentMatmulEmpty().apply(module)
    x = types.SimpleNamespace(shape=(7, 17), dtype="bf16", device="npu")
    y = types.SimpleNamespace(shape=(13, 17))

    assert result.state == "applied"
    assert module.linear_persistent(x, y)[0] == "empty"


def test_persistent_matmul_is_idempotent_for_upstreamed_change():
    module = make_module("empty")

    assert PersistentMatmulEmpty().apply(module).state == "already-applied"


def test_persistent_matmul_rejects_unknown_source_shape():
    module = make_module()
    source = """
def linear_persistent(x, y):
    M, K = x.shape
    N, _ = y.shape
    return torch.zeros((M, N), dtype=x.dtype, device=x.device)
"""
    filename = "<unknown_persistent_matmul>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, "exec"), module.__dict__)  # noqa: S102

    with pytest.raises(RuntimeError):
        PersistentMatmulEmpty().apply(module)
