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
"""
Generic LLM call detection engine.

Uses a framework-provided ``LLMDetector`` to
identify LLM objects in a function's scope and then counts invocation sites
via AST analysis.

The two public entry points are:

- ``discover_llm_names`` -- scope inspection (closures, globals, ``self``
  attributes, dict/list containers, factory return type annotations).
- ``count_llm_calls`` -- combines scope inspection with AST call-site
  counting and worst-case control-flow analysis.
"""

from __future__ import annotations

import ast
import inspect
import logging
import textwrap
import typing
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Literal

from nat_app.graph.protocols import LLMDetector

logger = logging.getLogger(__name__)

_DEFAULT_LOOP_MULTIPLIER: int = 3

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class LLMCallInfo:
    """Per-node result of LLM call detection."""

    call_count: int = 0
    """Worst-case number of LLM invocations detected in the function."""

    llm_names: frozenset[str] = field(default_factory=frozenset)
    """Names in scope that were identified as LLM objects."""

    confidence: Literal["full", "partial", "opaque"] = "full"
    """Detection confidence: "full" (all call sites resolved), "partial"
    (some targets resolved dynamically), or "opaque" (source unavailable)."""

    warnings: list[str] = field(default_factory=list)
    """Diagnostic messages from analysis."""


# ---------------------------------------------------------------------------
# Phase 1 — Scope inspection
# ---------------------------------------------------------------------------


def discover_llm_names(func: Callable, detector: LLMDetector) -> dict[str, Any]:
    """Identify names in *func*'s scope that reference LLM objects.

    Inspects closure variables, referenced globals, bound-method instance
    attributes, dict/list containers (one level deep), and callable return
    type annotations.

    Args:
        func: The callable whose scope to inspect.
        detector: Framework-specific LLM detector.

    Returns:
        Mapping of ``name -> object`` for each LLM found.
    """
    found: dict[str, Any] = {}

    try:
        cv = inspect.getclosurevars(func)
    except TypeError:
        return found

    _scan_namespace(cv.nonlocals, detector, found)
    _scan_namespace(cv.globals, detector, found)

    self_obj = getattr(func, "__self__", None)
    if self_obj is not None:
        for attr_name, attr_val in vars(self_obj).items():
            if detector.is_llm(attr_val):
                found[f"self.{attr_name}"] = attr_val
            elif isinstance(attr_val, dict):
                if any(detector.is_llm(v) for v in attr_val.values()):
                    found[f"self.{attr_name}"] = attr_val
            elif isinstance(attr_val, (list, tuple)):
                if any(detector.is_llm(v) for v in attr_val):
                    found[f"self.{attr_name}"] = attr_val

    return found


def _scan_namespace(
    namespace: Mapping[str, Any],
    detector: LLMDetector,
    found: dict[str, Any],
) -> None:
    """Check each entry in *namespace* for LLM objects (one level deep).

    Args:
        namespace: Name-to-object mapping to scan.
        detector: Framework-specific LLM detector.
        found: Mutable dict to accumulate discovered LLM names into.
    """
    for name, obj in namespace.items():
        if detector.is_llm(obj):
            found[name] = obj
            continue

        if isinstance(obj, dict):
            if any(detector.is_llm(v) for v in obj.values()):
                found[name] = obj
                continue

        if isinstance(obj, (list, tuple)):
            if any(detector.is_llm(v) for v in obj):
                found[name] = obj
                continue

        if hasattr(obj, "__dict__") and not isinstance(obj, type):
            for attr_name, attr_val in vars(obj).items():
                if detector.is_llm(attr_val):
                    found[f"{name}.{attr_name}"] = attr_val

        if callable(obj) and not detector.is_llm(obj):
            try:
                hints = typing.get_type_hints(obj)
                ret = hints.get("return")
                if ret is not None and isinstance(ret, type):
                    sentinel = object.__new__(ret) if not inspect.isabstract(ret) else None
                    if sentinel is not None and detector.is_llm(sentinel):
                        found[name] = obj
            except (NameError, TypeError, AttributeError, ValueError, ImportError) as exc:
                logger.debug("Could not get type hints for %r: %s", name, exc, exc_info=True)


# ---------------------------------------------------------------------------
# Phase 2 — AST call-site counting
# ---------------------------------------------------------------------------


def count_llm_calls(func: Callable, detector: LLMDetector) -> LLMCallInfo:
    """Count LLM invocation sites in *func* using *detector*.

    Combines ``discover_llm_names`` (scope inspection) with an AST walk
    that counts calls to ``detector.invocation_methods`` on the discovered
    names.  Control flow is handled conservatively: ``if/else`` takes the
    ``max`` of branches; loops multiply by ``_DEFAULT_LOOP_MULTIPLIER``.

    Args:
        func: The callable to analyze for LLM calls.
        detector: Framework-specific LLM detector.

    Returns:
        Per-node LLM call detection result with count and confidence.
    """
    llm_names_map = discover_llm_names(func, detector)
    if not llm_names_map:
        return LLMCallInfo()

    llm_names = set(llm_names_map.keys())

    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return LLMCallInfo(
            llm_names=frozenset(llm_names),
            confidence="opaque",
            warnings=["Source code not available for LLM call counting"],
        )

    source = textwrap.dedent(source)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return LLMCallInfo(
            llm_names=frozenset(llm_names),
            confidence="opaque",
            warnings=["Failed to parse source for LLM call counting"],
        )

    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_def = node
            break

    if func_def is None:
        return LLMCallInfo(
            llm_names=frozenset(llm_names),
            confidence="partial",
            warnings=["No function definition found in source"],
        )

    counter = _LLMCallCounter(llm_names, detector.invocation_methods)
    call_count = counter.count_in_body(func_def.body)

    confidence: str = "full"
    warnings: list[str] = []

    if counter.has_dynamic_targets:
        confidence = "partial"
        warnings.append("Some LLM call targets resolved dynamically")

    return LLMCallInfo(
        call_count=call_count,
        llm_names=frozenset(llm_names),
        confidence=confidence,
        warnings=warnings,
    )


class _LLMCallCounter:
    """AST visitor that counts LLM invocation sites with CFG awareness."""

    def __init__(
        self,
        llm_names: set[str],
        invocation_methods: frozenset[str],
    ) -> None:
        self._llm_names = llm_names
        self._methods = invocation_methods
        self.has_dynamic_targets: bool = False

    def count_in_body(self, stmts: list[ast.stmt]) -> int:
        """Worst-case LLM call count for a sequential block of statements.

        Args:
            stmts: List of AST statement nodes to analyze.

        Returns:
            Total worst-case LLM call count for the block.
        """
        total = 0
        for stmt in stmts:
            total += self._count_stmt(stmt)
        return total

    def _count_stmt(self, node: ast.stmt) -> int:
        if isinstance(node, ast.If):
            return self._count_if(node)
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            return self._count_loop(node)
        if isinstance(node, (ast.Try, ast.TryStar)):
            return self._count_try(node)
        if isinstance(node, (ast.With, ast.AsyncWith)):
            return self._count_with(node)
        if isinstance(node, ast.Match):
            return self._count_match(node)
        return self._count_calls_in_node(node)

    def _count_if(self, node: ast.If) -> int:
        body_count = self.count_in_body(node.body)
        else_count = self.count_in_body(node.orelse) if node.orelse else 0
        test_count = self._count_calls_in_node(node.test) if hasattr(node, "test") else 0
        return test_count + max(body_count, else_count)

    def _count_loop(self, node: ast.For | ast.AsyncFor | ast.While) -> int:
        body_count = self.count_in_body(node.body)
        else_count = self.count_in_body(node.orelse) if node.orelse else 0
        header_count = 0
        if isinstance(node, (ast.For, ast.AsyncFor)):
            header_count = self._count_calls_in_node(node.iter)
        elif isinstance(node, ast.While):
            header_count = self._count_calls_in_node(node.test)
        return header_count + body_count * _DEFAULT_LOOP_MULTIPLIER + else_count

    def _count_try(self, node: ast.Try | ast.TryStar) -> int:
        body_count = self.count_in_body(node.body)
        handler_counts = [self.count_in_body(h.body) for h in node.handlers]
        else_count = self.count_in_body(node.orelse) if node.orelse else 0
        finally_count = self.count_in_body(node.finalbody) if node.finalbody else 0
        worst_handler = max(handler_counts) if handler_counts else 0
        return max(body_count + else_count, body_count + worst_handler) + finally_count

    def _count_with(self, node: ast.With | ast.AsyncWith) -> int:
        header = sum(self._count_calls_in_node(item.context_expr) for item in node.items)
        return header + self.count_in_body(node.body)

    def _count_match(self, node: ast.Match) -> int:
        subject_count = self._count_calls_in_node(node.subject)
        case_counts = [(self._count_calls_in_node(c.guard) if c.guard is not None else 0) + self.count_in_body(c.body)
                       for c in node.cases]
        return subject_count + (max(case_counts) if case_counts else 0)

    def _count_calls_in_node(self, node: ast.AST) -> int:
        """Count LLM calls in an arbitrary AST node (expression, statement).

        Args:
            node: AST node to walk for LLM call sites.

        Returns:
            Number of LLM calls found in the node.
        """
        count = 0
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and self._is_llm_call(child.func):
                count += 1
        return count

    def _is_llm_call(self, node: ast.expr) -> bool:
        """Check if a Call's func node is an LLM invocation.

        Args:
            node: The ``func`` attribute of an ``ast.Call`` node.

        Returns:
            ``True`` if the call targets an LLM invocation method.
        """
        if not isinstance(node, ast.Attribute):
            return False
        if node.attr not in self._methods:
            return False
        name = self._resolve_receiver(node.value)
        if name is None:
            self.has_dynamic_targets = True
            return False
        if name not in self._llm_names:
            self.has_dynamic_targets = True
            return False
        return True

    def _resolve_receiver(self, node: ast.expr) -> str | None:
        """Resolve the receiver of a method call to a name string.

        Args:
            node: The receiver expression of a method call.

        Returns:
            Resolved name string, or ``None`` if unresolvable.
        """
        if isinstance(node, ast.Name):
            return node.id

        # self.attr
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self"):
            return f"self.{node.attr}"

        # name.attr (e.g. obj.llm)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            compound = f"{node.value.id}.{node.attr}"
            if compound in self._llm_names:
                return compound

        # Subscript on a known container (e.g. MODELS["main"])
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id in self._llm_names:
                return node.value.id

        return None
