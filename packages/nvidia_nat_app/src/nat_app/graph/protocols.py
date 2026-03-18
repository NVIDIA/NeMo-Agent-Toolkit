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
Framework adapter protocols.

These protocols define the explicit contract that framework-specific packages
must implement.  All algorithms in ``nat_app.graph`` operate on the abstract
`Graph` type produced by these adapters.

For convenience, use `AbstractFrameworkAdapter`
which implements all three protocols with sensible defaults.

A framework integration package implements three things:

1. **GraphExtractor** -- converts the framework's compiled artifact into a ``Graph``.
2. **NodeIntrospector** -- provides callable functions and schema information.
3. **GraphBuilder** -- builds an optimized framework artifact from analysis results.

Optional:

4. **LLMDetector** -- identifies LLM objects and invocation method names for
   the generic LLM call-counting engine used by the priority pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from typing import Protocol
from typing import runtime_checkable

from nat_app.graph.access import ReducerSet
from nat_app.graph.types import Graph


@runtime_checkable
class GraphExtractor(Protocol):
    """
    Extract an abstract `Graph` from a framework-specific compiled artifact.

    Example (LangGraph):

        class LangGraphExtractor:
            def extract(self, source: CompiledStateGraph) -> Graph:
                g = Graph()
                for name, node in source.nodes.items():
                    g.add_node(name, func=node.bound.func)
                # ... add edges, set entry_point, terminal_nodes ...
                return g
    """

    def extract(self, source: Any) -> Graph:
        """
        Extract a ``Graph`` from a framework-specific source artifact.

        Args:
            source: The framework's compiled graph (e.g. ``CompiledStateGraph``).

        Returns:
            An abstract ``Graph`` with nodes, edges, entry point, and terminals.
        """
        ...


@runtime_checkable
class NodeIntrospector(Protocol):
    """
    Provide callable functions and schema information from a framework's graph.

    This protocol separates node-level introspection from graph structure extraction
    so that different analysis strategies can be plugged in.
    """

    def get_node_func(self, node_id: str) -> Callable | None:
        """Return the callable for a node, or None if unavailable.

        Args:
            node_id: Identifier of the node to look up.

        Returns:
            The node's callable, or None if unavailable.
        """
        ...

    def get_state_schema(self) -> type | None:
        """Return the state schema type (e.g. a TypedDict class), or None.

        Returns:
            The state schema type, or None if not available.
        """
        ...

    def get_reducer_fields(self) -> ReducerSet:
        """Return per-object reducer fields (parallel-safe writes).

        Returns:
            A ``ReducerSet`` (``dict[str, set[str]]``).
        """
        ...

    def get_all_schema_fields(self) -> set[str] | None:
        """Return all field names from the state schema, or None.

        Returns:
            All field names from the state schema, or None if unavailable.
        """
        ...

    def get_special_call_names(self) -> set[str]:
        """Return framework-specific call names to detect as optimization barriers.

        E.g. ``{"Send", "Command"}`` for LangGraph.

        Returns:
            Set of call names that act as optimization barriers.
        """
        ...


@runtime_checkable
class LLMDetector(Protocol):
    """
    Identify LLM objects and their invocation methods for a specific framework.

    Framework adapters provide an implementation that encodes their
    framework's LLM type hierarchy and calling conventions.  The generic
    LLM call-counting engine
    (`count_llm_calls`) uses this to
    determine how many LLM call sites exist in each node function.

    Example (LangChain):

        class LangChainLLMDetector:
            def is_llm(self, obj):
                from langchain_core.language_models import BaseLanguageModel
                return isinstance(obj, BaseLanguageModel)

            @property
            def invocation_methods(self):
                return frozenset({"invoke", "ainvoke", "stream", "astream"})
    """

    def is_llm(self, obj: Any) -> bool:
        """Return ``True`` if *obj* is an LLM instance in this framework.

        Args:
            obj: The object to check.

        Returns:
            True if the object is an LLM instance.
        """
        ...

    @property
    def invocation_methods(self) -> frozenset[str]:
        """Method names that constitute an LLM call.

        The analysis engine counts calls to these methods on objects identified
        by `is_llm`.

        Returns:
            Frozen set of method names that constitute an LLM call.
        """
        ...


@runtime_checkable
class GraphBuilder(Protocol):
    """
    Build an optimized framework-specific artifact from compilation results.

    The builder receives the original framework artifact and a ``CompilationResult``
    (or its ``TransformationResult`` subclass) from ``nat_app.graph.scheduling``,
    and produces the optimized version.
    """

    def build(self, original: Any, result: Any) -> Any:
        """
        Build an optimized framework artifact.

        Args:
            original: The original framework artifact (e.g. ``CompiledStateGraph``).
            result: A ``CompilationResult`` (or ``TransformationResult``) from scheduling.

        Returns:
            The optimized framework artifact.
        """
        ...
