# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Static analysis for node functions via AST.

Parses Python source code to detect state reads and writes WITHOUT executing
the function.  Works for any framework that passes state as a dict-like
parameter.

Supports multiple state parameters (e.g. ``state``, ``memory``, ``config``)
via the ``param_to_obj`` mapping.  Each parameter is tracked as a separate
object namespace in the resulting `AccessSet`.

Framework-specific call detection (e.g. LangGraph's `Send`/`Command`) is
pluggable via the ``special_call_names`` parameter.

Limitations
-----------
- Assumes state is dict-like with string keys (TypedDict, plain dict).
  Does not support arbitrary Python objects with custom __getitem__ semantics.
- Cannot detect invisible mutations: C extensions, other threads, deserialization.
- Multi-param aliasing: when state and memory could refer to the same object
  at the call site, analysis may under-report dependencies.
- Recursion depth limited to 5 by default; deeper call chains may be under-analyzed.
"""

from __future__ import annotations

import ast
import inspect
import logging
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field

from nat_app.graph.access import AccessSet

logger = logging.getLogger(__name__)

_MUTATING_METHODS = frozenset({
    "append",
    "extend",
    "insert",
    "pop",
    "remove",
    "clear",
    "update",
    "add",
    "discard",
    "setdefault",
    "popitem",
    "sort",
    "reverse",
    "__setitem__",
    "__delitem__",
    "__iadd__",
})

_DEFAULT_OBJ = "state"

_DYNAMIC_EXEC_NAMES = frozenset({"exec", "eval", "compile"})
_KNOWN_SAFE_ATTR_READS = frozenset({"get", "keys", "values", "items", "copy"})


@dataclass
class StaticAnalysisResult:
    """Results of static analysis for a single node function.

    Uncertainty flags (`has_dynamic_exec`, `has_closure_write`, etc.) indicate
    patterns that prevent full confidence. When any flag is True, callers
    should treat the node as dependent (sequential) for safety.
    """

    reads: AccessSet = field(default_factory=AccessSet)
    writes: AccessSet = field(default_factory=AccessSet)
    mutations: AccessSet = field(default_factory=AccessSet)

    detected_special_calls: set[str] = field(default_factory=set)

    has_dynamic_keys: bool = False
    has_unresolved_calls: bool = False
    recursion_depth_hit: bool = False
    source_available: bool = True

    # Uncertainty principle flags (conservative fallback)
    has_dynamic_exec: bool = False
    """True if `exec`, `eval`, or `compile` is called."""
    has_closure_write: bool = False
    """True if writing to a closure freevar."""
    has_global_write: bool = False
    """True if writing to a non-param global."""
    has_unknown_attr_access: bool = False
    """True if `state.attr` where `attr` is not in known-safe set."""
    has_return_lambda_mutates_state: bool = False
    """True if return lambda references `state` (delayed mutation)."""
    has_dynamic_attr: bool = False
    """True if `setattr`/`getattr` with non-Constant `attr` argument."""

    warnings: list[str] = field(default_factory=list)

    @property
    def all_writes(self) -> AccessSet:
        """Union of return-dict writes and in-place mutations.

        Returns:
            Combined AccessSet of writes and mutations.
        """
        result = AccessSet()
        for obj, path in self.writes:
            result.add(obj, path)
        for obj, path in self.mutations:
            result.add(obj, path)
        return result


# ---------------------------------------------------------------------------
# AST Visitor
# ---------------------------------------------------------------------------


class _NodeASTVisitor(ast.NodeVisitor):
    """
    Walks an AST to find state reads, writes, and mutations.

    Tracks multiple state parameters via ``param_to_obj`` mapping.
    """

    def __init__(
        self,
        state_param: str,
        *,
        obj_name: str = _DEFAULT_OBJ,
        param_to_obj: dict[str, str] | None = None,
        special_call_names: frozenset[str] = frozenset(),
        enclosing_func: Callable | None = None,
        depth: int = 0,
        visited_funcs: set | None = None,
        self_state_attrs: dict[str, str] | None = None,
        max_recursion_depth: int = 5,
    ):
        # param_to_obj maps parameter names -> object namespace names
        if param_to_obj is not None:
            self._param_to_obj = dict(param_to_obj)
        else:
            self._param_to_obj = {state_param: obj_name}

        self._primary_param = state_param
        self.reads = AccessSet()
        self.writes = AccessSet()
        self.mutations = AccessSet()
        self.detected_special_calls: set[str] = set()
        self.has_dynamic_keys: bool = False
        self.has_unresolved_calls: bool = False
        self.recursion_depth_hit: bool = False
        self.has_dynamic_exec: bool = False
        self.has_closure_write: bool = False
        self.has_global_write: bool = False
        self.has_unknown_attr_access: bool = False
        self.has_return_lambda_mutates_state: bool = False
        self.has_dynamic_attr: bool = False
        self.warnings: list[str] = []

        self._special_call_names = special_call_names
        self._aliases: dict[str, tuple[str, str]] = {}  # var -> (obj, field)
        self._dict_vars: dict[str, set[str]] = {}
        self._state_aliases: dict[str, str] = {}  # alias_var -> obj_name
        # Maps self.X attribute names to object namespaces (for class methods like Flow)
        self._self_state_attrs: dict[str, str] = dict(self_state_attrs) if self_state_attrs else {}

        self._enclosing_func = enclosing_func
        self._depth = depth
        self._visited_funcs: set = visited_funcs or set()
        self._max_recursion_depth = max_recursion_depth
        self._freevars: set[str] = set()

    @staticmethod
    def _get_base_name_from_subscript(node: ast.expr) -> str | None:
        """Walk subscript chain to root Name. E.g. args[0][\"x\"] -> \"args\"."""
        current: ast.expr = node
        while isinstance(current, ast.Subscript):
            current = current.value
        if isinstance(current, ast.Name):
            return current.id
        return None

    def _get_obj_for_node(self, node: ast.expr) -> str | None:
        """If *node* is a tracked state parameter or alias, return its obj name.

        Args:
            node: An AST expression node to inspect.

        Returns:
            The object namespace name, or None if the node is not tracked.
        """
        if isinstance(node, ast.Name):
            if node.id in self._param_to_obj:
                return self._param_to_obj[node.id]
            if node.id in self._state_aliases:
                return self._state_aliases[node.id]
        # Handle self.state attribute access (for class methods like CrewAI Flow)
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self"
                and node.attr in self._self_state_attrs):
            return self._self_state_attrs[node.attr]
        return None

    def _is_state(self, node: ast.expr) -> bool:
        return self._get_obj_for_node(node) is not None

    def _extract_string_key(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        # Resolve variable names through defaults/globals/closure
        if isinstance(node, ast.Name):
            resolved = self._resolve_name(node.id)
            if isinstance(resolved, str):
                return resolved
        return None

    def _is_alias(self, node: ast.expr) -> tuple[str, str] | None:
        """If *node* aliases a state field, return (obj, field).

        Args:
            node: An AST expression node to inspect.

        Returns:
            A (obj, field) tuple if the node is an alias, or None.
        """
        if isinstance(node, ast.Name) and node.id in self._aliases:
            return self._aliases[node.id]
        return None

    def _extract_nested_path(self, node: ast.expr) -> tuple[str | None, str | None]:
        """Extract ``(obj_name, dotted_path)`` from a chain of subscripts/attributes on state.

        Walks from the outermost node inward to find the state root, collecting
        string keys along the way and joining them with dots.

        Args:
            node: An AST expression node (subscript/attribute chain).

        Returns:
            A (obj_name, dotted_path) tuple, or (None, None) if not a state access.

        Examples:

            state["user"]             -> ("state", "user")
            state["user"]["name"]     -> ("state", "user.name")
            state["a"]["b"]["c"]      -> ("state", "a.b.c")
            state["user"].name        -> ("state", "user.name")
            non_state_var["key"]      -> (None, None)
        """
        parts: list[str] = []
        current = node

        while True:
            if isinstance(current, ast.Subscript):
                key = self._extract_string_key(current.slice)
                if key:
                    parts.append(key)
                else:
                    return (None, None)
                current = current.value
            elif isinstance(current, ast.Attribute):
                # Check if this attribute node IS a state root (e.g. self.state)
                obj = self._get_obj_for_node(current)
                if obj is not None:
                    break
                if current.attr not in ("get",
                                        "keys",
                                        "values",
                                        "items",
                                        "update",
                                        "pop",
                                        "setdefault",
                                        "clear",
                                        "copy"):
                    parts.append(current.attr)
                current = current.value
            else:
                break

        obj = self._get_obj_for_node(current)
        if obj is None or not parts:
            return (None, None)

        parts.reverse()
        return (obj, ".".join(parts))

    def _get_callee_name(self, node: ast.Call) -> str | None:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def visit_Call(self, node: ast.Call):  # pylint: disable=invalid-name
        callee_name = self._get_callee_name(node)
        if callee_name in _DYNAMIC_EXEC_NAMES:
            self.has_dynamic_exec = True

        if callee_name in ("setattr", "getattr") and len(node.args) >= 2:
            if not (isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str)):
                self.has_dynamic_attr = True

        if (isinstance(node.func, ast.Attribute) and self._is_state(node.func.value) and node.func.attr == "get"
                and node.args):
            obj = self._get_obj_for_node(node.func.value)
            key = self._extract_string_key(node.args[0])
            if key and obj:
                self.reads.add(obj, key)
            elif not key:
                self.has_dynamic_keys = True
                self.warnings.append("Dynamic key in state.get()")

        if isinstance(node.func, ast.Attribute) and node.func.attr in _MUTATING_METHODS:
            receiver = node.func.value
            obj, path = self._extract_nested_path(receiver)
            if obj and path:
                self.mutations.add(obj, path)
                self.reads.add(obj, path)
            elif isinstance(receiver, ast.Subscript) and self._is_state(receiver.value):
                self.has_dynamic_keys = True
            alias_info = self._is_alias(receiver)
            if alias_info:
                self.mutations.add(alias_info[0], alias_info[1])
                self.reads.add(alias_info[0], alias_info[1])

        self._check_special_calls(node)
        self._check_state_passed_to_call(node)
        self.generic_visit(node)

    def _check_special_calls(self, node: ast.Call):
        if not self._special_call_names:
            return
        func = node.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name and name in self._special_call_names:
            self.detected_special_calls.add(name)

    def visit_Subscript(self, node: ast.Subscript):  # pylint: disable=invalid-name
        if self._is_state(node.value) and isinstance(node.ctx, ast.Load):
            obj = self._get_obj_for_node(node.value)
            key = self._extract_string_key(node.slice)
            if key and obj:
                self.reads.add(obj, key)
            elif not key:
                self.has_dynamic_keys = True
        if isinstance(node.ctx, ast.Store):
            base_name = self._get_base_name_from_subscript(node)
            if base_name is not None:
                param_names = set(self._param_to_obj) | set(self._state_aliases)
                if base_name in self._freevars:
                    self.has_closure_write = True
                elif base_name not in param_names:
                    self.has_global_write = True
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):  # pylint: disable=invalid-name
        if self._is_state(node.value) and isinstance(node.ctx, ast.Load):
            obj = self._get_obj_for_node(node.value)
            if obj and node.attr not in (
                    "get", "keys", "values", "items", "update", "pop", "setdefault", "clear", "copy"):
                self.reads.add(obj, node.attr)
            if node.attr not in _KNOWN_SAFE_ATTR_READS and node.attr not in _MUTATING_METHODS:
                self.has_unknown_attr_access = True
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):  # pylint: disable=invalid-name
        for target in node.targets:
            self._handle_assign_target(target, node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):  # pylint: disable=invalid-name
        if isinstance(node.target, (ast.Subscript, ast.Attribute)):
            obj, path = self._extract_nested_path(node.target)
            if obj and path:
                self.reads.add(obj, path)
                self.mutations.add(obj, path)
            elif isinstance(node.target, ast.Subscript) and self._is_state(node.target.value):
                self.has_dynamic_keys = True
            if isinstance(node.target, ast.Subscript):
                base_name = self._get_base_name_from_subscript(node.target)
                if base_name is not None:
                    param_names = set(self._param_to_obj) | set(self._state_aliases)
                    if base_name in self._freevars:
                        self.has_closure_write = True
                    elif base_name not in param_names:
                        self.has_global_write = True
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return):
        if node.value is None:
            self.generic_visit(node)
            return
        self._extract_writes_from_expr(node.value)
        self._check_special_in_return(node.value)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete):
        for target in node.targets:
            if isinstance(target, (ast.Subscript, ast.Attribute)):
                obj, path = self._extract_nested_path(target)
                if obj and path:
                    self.mutations.add(obj, path)
                elif isinstance(target, ast.Subscript):
                    base_name = self._get_base_name_from_subscript(target)
                    if base_name is not None:
                        param_names = set(self._param_to_obj) | set(self._state_aliases)
                        if base_name in self._freevars:
                            self.has_closure_write = True
                        elif base_name not in param_names:
                            self.has_global_write = True
        self.generic_visit(node)

    # -- assignment helpers -------------------------------------------------

    def _handle_assign_target(self, target: ast.expr, value: ast.expr):
        # Try nested path first (handles subscripts and attribute chains)
        if isinstance(target, (ast.Subscript, ast.Attribute)):
            obj, path = self._extract_nested_path(target)
            if obj and path:
                self.mutations.add(obj, path)
                if "." in path:
                    self.reads.add(obj, path)
                return
            if isinstance(target, ast.Subscript):
                if obj is None and self._is_state(target.value):
                    self.has_dynamic_keys = True
                    return
                base_name = self._get_base_name_from_subscript(target)
                if base_name is not None:
                    param_names = set(self._param_to_obj) | set(self._state_aliases)
                    if base_name in self._freevars:
                        self.has_closure_write = True
                    elif base_name not in param_names:
                        self.has_global_write = True
                return

        if isinstance(target, ast.Name) and self._is_state(value):
            obj = self._get_obj_for_node(value)
            if obj:
                self._state_aliases[target.id] = obj
            return

        if isinstance(target, ast.Name):
            alias_info = self._resolve_state_source(value)
            if alias_info:
                self._aliases[target.id] = alias_info

            if isinstance(value, ast.Dict):
                keys = set()
                for k in value.keys:
                    if k is not None:
                        s = self._extract_string_key(k)
                        if s:
                            keys.add(s)
                if keys:
                    self._dict_vars[target.id] = keys

            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "dict"):
                keys = set()
                for kw in value.keywords:
                    if kw.arg is not None:
                        keys.add(kw.arg)
                if keys:
                    self._dict_vars[target.id] = keys

        if isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple):
            # strict=False: starred unpacking (a, *rest = x, y, z) yields different lengths
            for t, v in zip(target.elts, value.elts, strict=False):
                if isinstance(t, ast.Name):
                    alias_info = self._resolve_state_source(v)
                    if alias_info:
                        self._aliases[t.id] = alias_info

    def _resolve_state_source(self, node: ast.expr) -> tuple[str, str] | None:
        """If *node* reads a state field, return ``(obj, dotted_path)``.

        Args:
            node: An AST expression node to inspect.

        Returns:
            A (obj, dotted_path) tuple, or None if the node is not a state read.
        """
        # Try nested path first (handles chained subscripts/attributes)
        obj, path = self._extract_nested_path(node)
        if obj and path:
            return (obj, path)
        # Fallback: state.get("key")
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and self._is_state(node.func.value)
                and node.func.attr == "get" and node.args):
            obj2 = self._get_obj_for_node(node.func.value)
            key = self._extract_string_key(node.args[0])
            if obj2 and key:
                return (obj2, key)
        return None

    # -- write extraction ---------------------------------------------------

    def _extract_writes_from_expr(self, node: ast.expr):
        # Return dict writes go to the primary object by default
        obj = next(iter(self._param_to_obj.values())) if self._param_to_obj else _DEFAULT_OBJ

        if isinstance(node, ast.Dict):
            self._extract_dict_keys_as_writes(node, obj)
            return
        if isinstance(node, ast.Name) and node.id in self._dict_vars:
            for key in self._dict_vars[node.id]:
                self.writes.add(obj, key)
            return
        if isinstance(node, ast.IfExp):
            self._extract_writes_from_expr(node.body)
            self._extract_writes_from_expr(node.orelse)
            return
        if isinstance(node, ast.Lambda):
            if self._lambda_references_state(node):
                self.has_return_lambda_mutates_state = True
            return
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "dict":
                keys = set()
                for kw in node.keywords:
                    if kw.arg is not None:
                        keys.add(kw.arg)
                if keys:
                    for key in keys:
                        self.writes.add(obj, key)
                    return
            self.has_dynamic_keys = True
            self.warnings.append("Return value is a function call — write keys unknown")
            return
        if not isinstance(node, ast.Constant):
            if isinstance(node, ast.Name):
                self.warnings.append(f"Return variable '{node.id}' not tracked — write keys unknown")
                self.has_dynamic_keys = True

    def _lambda_references_state(self, node: ast.Lambda) -> bool:
        """True if lambda body references tracked state (param or alias)."""
        param_names = set(self._param_to_obj) | set(self._state_aliases)
        for child in ast.walk(node.body):
            if isinstance(child, ast.Name) and child.id in param_names:
                return True
            if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
                if child.value.id in param_names:
                    return True
        return False

    def _extract_dict_keys_as_writes(self, node: ast.Dict, obj: str):
        for key_node in node.keys:
            if key_node is None:
                self.has_dynamic_keys = True
                self.warnings.append("Dict spread (**) in return — some write keys unknown")
            else:
                key = self._extract_string_key(key_node)
                if key:
                    self.writes.add(obj, key)
                else:
                    self.has_dynamic_keys = True
                    self.warnings.append("Non-literal key in return dict")

    # -- special call detection in return -----------------------------------

    def _check_special_in_return(self, node: ast.expr):
        if not self._special_call_names:
            return
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                name = None
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                if name and name in self._special_call_names:
                    self.detected_special_calls.add(name)

    # -- recursive helper analysis ------------------------------------------

    def _check_state_passed_to_call(self, node: ast.Call):
        state_arg_positions: list[int] = []
        for i, arg in enumerate(node.args):
            if self._is_state(arg):
                state_arg_positions.append(i)

        state_kwarg_keys: list[str] = []
        for kw in node.keywords:
            if kw.arg and self._is_state(kw.value):
                state_kwarg_keys.append(kw.arg)

        state_as_receiver = (isinstance(node.func, ast.Attribute) and self._is_state(node.func.value)
                             and node.func.attr not in _MUTATING_METHODS
                             and node.func.attr not in _KNOWN_SAFE_ATTR_READS)

        if not state_arg_positions and not state_kwarg_keys and not state_as_receiver:
            return

        if isinstance(node.func, ast.Attribute) and node.func.attr in _MUTATING_METHODS:
            return
        if isinstance(node.func, ast.Name) and node.func.id in (
                "dict",
                "list",
                "tuple",
                "set",
                "str",
                "int",
                "float",
                "bool",
                "len",
                "print",
                "sorted",
                "reversed",
                "enumerate",
                "zip",
                "map",
                "filter",
                "isinstance",
                "type",
                "getattr",
                "hasattr",
        ):
            return
        if isinstance(node.func, ast.Name) and node.func.id in self._special_call_names:
            return

        callee = self._resolve_callee(node.func)
        if callee is None:
            self.has_unresolved_calls = True
            func_name = self._callee_name(node.func)
            self.warnings.append(f"State passed to unresolvable function '{func_name}'")
            return

        if self._depth >= self._max_recursion_depth:
            self.recursion_depth_hit = True
            self.warnings.append(f"Recursion depth limit ({self._max_recursion_depth}) reached at {callee.__name__}")
            return

        callee_id = id(callee)
        if callee_id in self._visited_funcs:
            return
        visited = self._visited_funcs | {callee_id}

        sub_result = _analyze_callee(
            callee,
            state_arg_positions=state_arg_positions,
            state_kwarg_keys=state_kwarg_keys,
            special_call_names=self._special_call_names,
            param_to_obj=self._param_to_obj,
            depth=self._depth + 1,
            visited_funcs=visited,
            max_recursion_depth=self._max_recursion_depth,
        )

        if sub_result is not None:
            for obj, path in sub_result.reads:
                self.reads.add(obj, path)
            for obj, path in sub_result.writes:
                self.writes.add(obj, path)
            for obj, path in sub_result.mutations:
                self.mutations.add(obj, path)
            self.detected_special_calls |= sub_result.detected_special_calls
            if sub_result.has_dynamic_keys:
                self.has_dynamic_keys = True
            if sub_result.has_unresolved_calls:
                self.has_unresolved_calls = True
            if sub_result.recursion_depth_hit:
                self.recursion_depth_hit = True
            if sub_result.has_dynamic_exec:
                self.has_dynamic_exec = True
            if sub_result.has_closure_write:
                self.has_closure_write = True
            if sub_result.has_global_write:
                self.has_global_write = True
            if sub_result.has_unknown_attr_access:
                self.has_unknown_attr_access = True
            if sub_result.has_return_lambda_mutates_state:
                self.has_return_lambda_mutates_state = True
            if sub_result.has_dynamic_attr:
                self.has_dynamic_attr = True
            self.warnings.extend(sub_result.warnings)
        else:
            self.has_unresolved_calls = True

    def _resolve_callee(self, func_node: ast.expr) -> Callable | None:
        if self._enclosing_func is None:
            return None
        if isinstance(func_node, ast.Name):
            name = func_node.id
            resolved = self._resolve_name(name)
            if resolved is not None and callable(resolved):
                return resolved

        # Handle dict-dispatch pattern: SOME_DICT[key](state)
        if isinstance(func_node, ast.Subscript):
            container = self._resolve_subscript_container(func_node.value)
            if container is not None and isinstance(container, dict):
                key = self._resolve_subscript_key(func_node.slice)
                if key is not None and key in container:
                    target = container[key]
                    if callable(target):
                        return target

        return None

    def _resolve_name(self, name: str) -> object | None:
        """Resolve a name through globals, closure vars, and default args.

        Args:
            name: The variable name to resolve.

        Returns:
            The resolved value, or None if unresolvable.
        """
        if self._enclosing_func is None:
            return None

        func_globals = getattr(self._enclosing_func, "__globals__", {})
        candidate = func_globals.get(name)
        if candidate is not None:
            return candidate

        func_code = getattr(self._enclosing_func, "__code__", None)
        if func_code:
            free_vars = func_code.co_freevars
            closure_cells = getattr(self._enclosing_func, "__closure__", None) or ()
            for var_name, cell in zip(free_vars, closure_cells, strict=True):
                if var_name == name:
                    try:
                        return cell.cell_contents
                    except ValueError:
                        pass

        # Check default arguments
        defaults = getattr(self._enclosing_func, "__defaults__", None) or ()
        if func_code and defaults:
            arg_names = func_code.co_varnames[:func_code.co_argcount]
            n_defaults = len(defaults)
            defaulted_params = arg_names[len(arg_names) - n_defaults:]
            for param_name, default_val in zip(defaulted_params, defaults, strict=True):
                if param_name == name:
                    return default_val

        return None

    def _resolve_subscript_container(self, node: ast.expr) -> object | None:
        """Resolve the container part of a subscript (e.g. STEP_FUNCTIONS).

        Args:
            node: An AST expression node for the subscript container.

        Returns:
            The resolved container object, or None if unresolvable.
        """
        if isinstance(node, ast.Name):
            return self._resolve_name(node.id)
        return None

    def _resolve_subscript_key(self, node: ast.expr) -> object | None:
        """Resolve the key/index of a subscript to a concrete value.

        Args:
            node: An AST expression node for the subscript key.

        Returns:
            The resolved key value, or None if unresolvable.
        """
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self._resolve_name(node.id)
        return None

    @staticmethod
    def _callee_name(func_node: ast.expr) -> str:
        if isinstance(func_node, ast.Name):
            return func_node.id
        if isinstance(func_node, ast.Attribute):
            return f"...{func_node.attr}"
        return "<unknown>"


# ---------------------------------------------------------------------------
# Recursive callee analysis
# ---------------------------------------------------------------------------


def _analyze_callee(
    callee: Callable,
    *,
    state_arg_positions: list[int],
    state_kwarg_keys: list[str],
    special_call_names: frozenset[str],
    param_to_obj: dict[str, str],
    depth: int,
    visited_funcs: set,
    max_recursion_depth: int = 5,
) -> StaticAnalysisResult | None:
    try:
        source = inspect.getsource(callee)
    except (OSError, TypeError):
        return None

    source = textwrap.dedent(source)
    tree = None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        if "lambda" in source:
            lambda_source = _extract_lambda_source(source)
            if lambda_source:
                try:
                    tree = ast.parse(lambda_source)
                except SyntaxError:
                    pass
        if tree is None:
            return None

    # Find function or lambda node
    func_def = None
    lambda_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_def = node
            break
        if isinstance(node, ast.Lambda):
            lambda_node = node
            break

    if func_def is not None:
        params = func_def.args.posonlyargs + func_def.args.args
        param_names = []
        for p in params:
            if p.arg in ("self", "cls") and not param_names:
                continue
            param_names.append(p.arg)

        state_param: str | None = None
        for pos in state_arg_positions:
            if pos < len(param_names):
                state_param = param_names[pos]
                break
        if state_param is None:
            for kw_name in state_kwarg_keys:
                if kw_name in param_names:
                    state_param = kw_name
                    break
        if state_param is None:
            return None

        obj_name = next(iter(param_to_obj.values()), _DEFAULT_OBJ)
        callee_param_to_obj = {state_param: obj_name}

        visitor = _NodeASTVisitor(
            state_param,
            param_to_obj=callee_param_to_obj,
            special_call_names=special_call_names,
            enclosing_func=callee,
            depth=depth,
            visited_funcs=visited_funcs,
            max_recursion_depth=max_recursion_depth,
        )
        visitor.visit(func_def)

    elif lambda_node is not None:
        state_param = _get_lambda_param_name(lambda_node)
        if state_param is None:
            return None

        # For lambdas, the state param is always position 0
        lambda_params = lambda_node.args.posonlyargs + lambda_node.args.args
        lambda_param_names = [p.arg for p in lambda_params]
        resolved_param: str | None = None
        for pos in state_arg_positions:
            if pos < len(lambda_param_names):
                resolved_param = lambda_param_names[pos]
                break
        if resolved_param is None:
            resolved_param = state_param

        obj_name = next(iter(param_to_obj.values()), _DEFAULT_OBJ)
        callee_param_to_obj = {resolved_param: obj_name}

        visitor = _NodeASTVisitor(
            resolved_param,
            param_to_obj=callee_param_to_obj,
            special_call_names=special_call_names,
            enclosing_func=callee,
            depth=depth,
            visited_funcs=visited_funcs,
            max_recursion_depth=max_recursion_depth,
        )
        visitor._extract_writes_from_expr(lambda_node.body)
        visitor.visit(lambda_node.body)
    else:
        return None

    result = StaticAnalysisResult()
    result.reads = visitor.reads
    result.writes = visitor.writes
    result.mutations = visitor.mutations
    result.detected_special_calls = visitor.detected_special_calls
    result.has_dynamic_keys = visitor.has_dynamic_keys
    result.has_unresolved_calls = visitor.has_unresolved_calls
    result.recursion_depth_hit = visitor.recursion_depth_hit
    result.has_dynamic_exec = visitor.has_dynamic_exec
    result.has_closure_write = visitor.has_closure_write
    result.has_global_write = visitor.has_global_write
    result.has_unknown_attr_access = visitor.has_unknown_attr_access
    result.has_return_lambda_mutates_state = visitor.has_return_lambda_mutates_state
    result.has_dynamic_attr = visitor.has_dynamic_attr
    result.warnings = visitor.warnings
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_function_ast(
    func: Callable,
    special_call_names: set[str] | None = None,
    param_to_obj: dict[str, str] | None = None,
    self_state_attrs: dict[str, str] | None = None,
    max_recursion_depth: int = 5,
) -> StaticAnalysisResult:
    """
    Analyze a function via AST to detect state reads and writes.

    Args:
        func: The node function to analyze.
        special_call_names: Framework-specific call names to detect
            (e.g. ``{"Send", "Command"}`` for LangGraph).
        param_to_obj: Mapping of parameter names to object namespace names.
            Defaults to ``{first_param: "state"}``.  For multi-object
            frameworks: ``{"state": "state", "memory": "memory"}``.
        self_state_attrs: Mapping of ``self.X`` attribute names to object
            namespace names.  For class-method frameworks like CrewAI Flow:
            ``{"state": "state"}`` means ``self.state`` is tracked as
            the ``"state"`` object.
        max_recursion_depth: Max call depth when following callees. Default 5.

    Returns:
        StaticAnalysisResult with reads, writes, mutations as AccessSet objects.
        Confidence is conservative: when uncertain (any uncertainty flag set),
        callers should treat the node as dependent (sequential) for safety.
    """
    frozen_specials = frozenset(special_call_names) if special_call_names else frozenset()
    result = StaticAnalysisResult()

    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        result.source_available = False
        result.warnings.append("Source code not available — AST analysis skipped")
        return result

    source = textwrap.dedent(source)

    # Fix: extract lambda from dict-entry source (e.g. '"key": lambda s: {...},')
    tree = None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        if "lambda" in source:
            lambda_source = _extract_lambda_source(source)
            if lambda_source:
                try:
                    tree = ast.parse(lambda_source)
                except SyntaxError:
                    pass
        if tree is None:
            result.source_available = False
            result.warnings.append("Failed to parse source")
            return result

    # Find the function/lambda node
    func_def = None
    lambda_node = None

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_def = node
            break
        if isinstance(node, ast.Lambda):
            lambda_node = node
            break

    if func_def is not None:
        # Standard function definition
        if param_to_obj is None:
            state_param = _get_state_param_name(func_def)
            if state_param is None:
                if self_state_attrs:
                    # Class method with self.state access (e.g. CrewAI Flow).
                    # Don't map "self" into param_to_obj -- the self_state_attrs
                    # mechanism handles self.X attribute access directly.
                    state_param = "_self_state_placeholder"
                    effective_param_to_obj = {}
                else:
                    result.warnings.append("Function has no parameters — cannot identify state")
                    return result
            else:
                effective_param_to_obj = {state_param: _DEFAULT_OBJ}
        else:
            if not param_to_obj:
                raise ValueError("param_to_obj must contain at least one mapping")
            effective_param_to_obj = dict(param_to_obj)
            state_param = next(iter(effective_param_to_obj))

        visitor = _NodeASTVisitor(
            state_param,
            param_to_obj=effective_param_to_obj,
            special_call_names=frozen_specials,
            enclosing_func=func,
            depth=0,
            visited_funcs={id(func)},
            self_state_attrs=self_state_attrs,
            max_recursion_depth=max_recursion_depth,
        )
        code = getattr(func, "__code__", None)
        visitor._freevars = set(getattr(code, "co_freevars", ())) if code else set()
        visitor.visit(func_def)

    elif lambda_node is not None:
        # Lambda expression
        state_param = _get_lambda_param_name(lambda_node)
        if state_param is None:
            result.warnings.append("Lambda has no parameters — cannot identify state")
            return result

        if param_to_obj is None:
            effective_param_to_obj = {state_param: _DEFAULT_OBJ}
        else:
            if not param_to_obj:
                raise ValueError("param_to_obj must contain at least one mapping")
            effective_param_to_obj = dict(param_to_obj)

        visitor = _NodeASTVisitor(
            state_param,
            param_to_obj=effective_param_to_obj,
            special_call_names=frozen_specials,
            enclosing_func=func,
            depth=0,
            visited_funcs={id(func)},
            max_recursion_depth=max_recursion_depth,
        )
        code = getattr(func, "__code__", None)
        visitor._freevars = set(getattr(code, "co_freevars", ())) if code else set()
        # Visit the lambda body and treat it as a return value
        visitor._extract_writes_from_expr(lambda_node.body)
        visitor.visit(lambda_node.body)

    else:
        result.warnings.append("No function or lambda found in source")
        return result

    result.reads = visitor.reads
    result.writes = visitor.writes
    result.mutations = visitor.mutations
    result.detected_special_calls = visitor.detected_special_calls
    result.has_dynamic_keys = visitor.has_dynamic_keys
    result.has_unresolved_calls = visitor.has_unresolved_calls
    result.recursion_depth_hit = visitor.recursion_depth_hit
    result.has_dynamic_exec = visitor.has_dynamic_exec
    result.has_closure_write = visitor.has_closure_write
    result.has_global_write = visitor.has_global_write
    result.has_unknown_attr_access = visitor.has_unknown_attr_access
    result.has_return_lambda_mutates_state = visitor.has_return_lambda_mutates_state
    result.has_dynamic_attr = visitor.has_dynamic_attr
    result.warnings = visitor.warnings

    func_name = getattr(func, "__name__", type(func).__name__)
    logger.debug(
        "AST analysis of %s: reads=%s, writes=%s, mutations=%s, specials=%s",
        func_name,
        result.reads,
        result.writes,
        result.mutations,
        result.detected_special_calls,
    )

    return result


def _extract_lambda_source(source: str) -> str | None:
    """Extract a lambda expression from dict-entry or assignment source.

    Args:
        source: Raw source code that may contain a lambda.

    Returns:
        The extracted lambda source string, or None if not found.
    """
    try:
        idx = source.index("lambda")
    except ValueError:
        return None
    extracted = source[idx:]
    extracted = extracted.rstrip()
    if extracted.endswith(","):
        extracted = extracted[:-1].rstrip()
    return extracted


def _get_lambda_param_name(lambda_node: ast.Lambda) -> str | None:
    """Extract the first parameter name from a Lambda node.

    Args:
        lambda_node: A Lambda AST node.

    Returns:
        The first parameter name, or None if the lambda has no parameters.
    """
    args = lambda_node.args
    params = args.posonlyargs + args.args
    if not params:
        return None
    return params[0].arg


def _get_state_param_name(func_def: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    args = func_def.args
    params = args.posonlyargs + args.args
    if params:
        first = params[0]
        if first.arg in ("self", "cls"):
            if len(params) > 1:
                return params[1].arg
            # Fall through to vararg if only self
        else:
            return first.arg
    # Include *args when no positional params (e.g. def fn(*args))
    if args.vararg:
        return args.vararg.arg if hasattr(args.vararg, "arg") else str(args.vararg)
    return None
