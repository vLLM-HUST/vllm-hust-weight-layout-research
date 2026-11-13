from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable
from types import FunctionType, ModuleType
from typing import Any


class CallReplacement(ast.NodeTransformer):
    def __init__(self, old_owner: str, old_name: str, new_name: str) -> None:
        self.old_owner = old_owner
        self.old_name = old_name
        self.new_name = new_name
        self.replacements = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id == self.old_owner
            and function.attr == self.old_name
        ):
            function.attr = self.new_name
            self.replacements += 1
        return node


def replace_single_call(
    module: ModuleType,
    function_name: str,
    *,
    old_owner: str,
    old_name: str,
    new_name: str,
    validator: Callable[[ast.FunctionDef], None] | None = None,
) -> FunctionType:
    original = getattr(module, function_name, None)
    if not isinstance(original, FunctionType):
        raise TypeError(f"{module.__name__}.{function_name} is not a Python function")

    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != function_name:
        raise RuntimeError(f"cannot isolate source for {module.__name__}.{function_name}")
    function_node = functions[0]
    if validator is not None:
        validator(function_node)

    transformer = CallReplacement(old_owner, old_name, new_name)
    transformer.visit(function_node)
    if transformer.replacements != 1:
        raise RuntimeError(
            f"expected one {old_owner}.{old_name} call in "
            f"{module.__name__}.{function_name}, found {transformer.replacements}"
        )

    ast.fix_missing_locations(tree)
    namespace: dict[str, Any] = {}
    code = compile(tree, inspect.getsourcefile(original) or "<operator-optimization>", "exec")
    exec(code, module.__dict__, namespace)  # noqa: S102 - validated local source rewrite
    replacement = namespace[function_name]
    replacement.__module__ = original.__module__
    replacement.__qualname__ = original.__qualname__
    replacement.__doc__ = original.__doc__
    replacement.__wrapped__ = original
    setattr(module, function_name, replacement)
    return replacement
