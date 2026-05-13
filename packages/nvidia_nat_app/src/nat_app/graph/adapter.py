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
Base framework adapter for integrating agent frameworks with nat_app.

Subclass ``AbstractFrameworkAdapter`` and implement ``extract()`` and
``build()`` to integrate a new framework.  All other methods have sensible
defaults (AST-based node analysis, no reducers, single-state parameter).

Example:

    class MyFrameworkAdapter(AbstractFrameworkAdapter):
        def extract(self, source) -> Graph:
            g = Graph()
            for task in source.tasks:
                g.add_node(task.name, func=task.run)
            # add edges, set entry_point, terminal_nodes ...
            return g

        def build(self, original, result):
            # construct optimized framework artifact from result.optimized_order
            ...

    optimizer = GraphOptimizer(adapter=MyFrameworkAdapter())
    result = optimizer.optimize(my_framework_graph)
    optimized = adapter.build(my_framework_graph, result)
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING
from typing import Any

from nat_app.constraints import OptimizationConfig
from nat_app.graph.access import AccessSet
from nat_app.graph.access import ReducerSet
from nat_app.graph.analysis import NodeAnalysis
from nat_app.graph.scheduling import CompilationResult
from nat_app.graph.static_analysis import analyze_function_ast
from nat_app.graph.types import Graph

if TYPE_CHECKING:
    from nat_app.graph.protocols import LLMDetector


class AbstractFrameworkAdapter(ABC):
    """Abstract base class for framework integration adapters.

    **Required** -- override these:

    - ``extract`` -- convert framework artifact to abstract ``Graph``
    - ``build`` -- construct optimized framework artifact from ``CompilationResult``

    **Optional** -- override for framework-specific behavior:

    - ``get_node_func`` -- return callable for a node (default: reads from Graph metadata)
    - ``get_state_schema`` -- return state schema type
    - ``get_reducer_fields`` -- return per-object reducer fields
    - ``get_all_schema_fields`` -- return all schema field names
    - ``analyze_node`` -- analyze a single node (default: AST analysis)
    - ``get_special_call_names`` -- framework-specific calls to detect
    - ``get_param_to_obj`` -- parameter-to-object mapping for multi-state
    - ``get_llm_detector`` -- LLM detection for the priority pipeline
    """

    # -- Required (abstract) -----------------------------------------------

    @abstractmethod
    def extract(self, source: Any) -> Graph:
        """Convert a framework-specific artifact to an abstract ``Graph``.

        Must add nodes, edges, set ``entry_point``, and ``terminal_nodes``.

        Args:
            source: Framework-specific graph or workflow artifact to convert.

        Returns:
            Abstract graph representation with nodes, edges, and metadata.
        """

    @abstractmethod
    def build(self, original: Any, result: CompilationResult) -> Any:
        """Build an optimized framework artifact from a ``CompilationResult``.

        Use ``result.optimized_order`` (list of parallel stages) to construct
        the optimized version of your framework's graph.

        Args:
            original: The original framework artifact that was analyzed.
            result: Compilation result containing the optimized execution schedule.

        Returns:
            Optimized framework artifact built from the compilation result.
        """

    # -- Optional (sensible defaults) --------------------------------------

    def get_node_func(self, node_id: str) -> Callable | None:
        """Return the callable for a node, or None if unavailable.

        Default: reads ``func`` from the Graph node metadata (set via
        ``graph.add_node(name, func=my_fn)``).  Override for frameworks
        that store callables differently.

        Args:
            node_id: Identifier of the node in the graph.

        Returns:
            The callable associated with the node, or ``None``.
        """
        return None

    def get_state_schema(self) -> type | None:
        """Return the state schema type (e.g. TypedDict, Pydantic model), or None.

        Used for conservative fallback when AST analysis has low confidence.

        Returns:
            The state schema type, or ``None`` if unavailable.
        """
        return None

    def get_reducer_fields(self) -> ReducerSet:
        """Return per-object reducer fields (parallel-safe writes).

        Returns a ``ReducerSet`` (``dict[str, set[str]]``), e.g.
        ``{"state": {"messages"}}`` for a ``messages`` field with an append reducer.

        Default: no reducers (empty dict).

        Returns:
            Per-object mapping of field names that have reducers.
        """
        return {}

    def get_all_schema_fields(self) -> set[str] | None:
        """Return all field names from the state schema, or None.

        Used for conservative fallback when AST analysis can't determine writes.

        Returns:
            Set of all field names in the schema, or ``None``.
        """
        return None

    def get_special_call_names(self) -> set[str]:
        """Return framework-specific call names to detect in AST analysis.

        These calls act as optimization barriers -- nodes that use them
        won't be parallelized.  E.g. ``{"Send", "Command"}`` for LangGraph.

        Default: empty set (no special calls).

        Returns:
            Names of framework-specific calls that act as optimization barriers.
        """
        return set()

    def get_param_to_obj(self) -> dict[str, str] | None:
        """Return parameter-to-object mapping for multi-state frameworks.

        Maps function parameter names to object namespace names for
        ``AccessSet`` tracking.

        E.g. ``{"state": "state", "memory": "memory"}`` for a framework
        where nodes receive both a state dict and a memory object.

        Default: None (single-state, first parameter maps to "state").

        Returns:
            Mapping of parameter names to object namespaces, or ``None``.
        """
        return None

    def get_self_state_attrs(self) -> dict[str, str] | None:
        """Return mapping of ``self.X`` attributes to object namespaces.

        For class-method-based frameworks (like CrewAI Flow) where nodes
        are methods that access state through ``self.state`` rather than
        a function parameter.

        E.g. ``{"state": "state"}`` tells the AST analyzer that
        ``self.state["key"]`` should be tracked as reads/writes on
        the ``"state"`` object.

        Default: None (not a class-method framework).

        Returns:
            Mapping of ``self`` attributes to object namespaces, or ``None``.
        """
        return None

    def get_llm_detector(self) -> LLMDetector | None:
        """Return an LLM detector for this framework, or ``None``.

        When provided, the ``LLMAnalysisStage``
        uses it to count LLM call sites per node for priority assignment.

        Default: ``None`` (no LLM detection, priority stays unassigned).

        Returns:
            Framework-specific LLM detector, or ``None``.
        """
        return None

    def map_profiler_function_to_node(self, function_name: str) -> str | None:
        """Map a profiler function name to a graph node name, or ``None`` if unknown.

        Used by framework-specific code when aggregating profiler output
        into ``ProfiledNodeCost`` dicts for
        ``seed_context`` injection.

        Default assumes 1:1 mapping (function_name == node_name).

        Args:
            function_name: Name of the function from profiler output.

        Returns:
            Corresponding graph node name, or ``None`` if unknown.
        """
        return function_name

    def analyze_node(
        self,
        name: str,
        func: Callable,
        state_schema: type | None = None,
        all_schema_fields: set[str] | None = None,
        *,
        config: OptimizationConfig | None = None,
    ) -> NodeAnalysis:
        """Analyze a single node function for read/write access.

        Default implementation uses the nat_app AST analyzer with
        ``get_special_call_names()``, ``get_param_to_obj()``, and
        ``get_self_state_attrs()``.

        Override for frameworks that need custom introspection
        (e.g. subgraph detection, runtime tracing).

        Args:
            name: Node identifier in the graph.
            func: The callable to analyze.
            state_schema: Optional state schema type for conservative fallback.
            all_schema_fields: Optional set of all schema field names for fallback.

        Returns:
            Analysis result with read/write sets and confidence level.
        """
        analysis = NodeAnalysis(name=name)

        max_depth = config.max_recursion_depth if config else 5
        ast_result = analyze_function_ast(
            func,
            special_call_names=self.get_special_call_names() or None,
            param_to_obj=self.get_param_to_obj(),
            self_state_attrs=self.get_self_state_attrs(),
            max_recursion_depth=max_depth,
        )

        if not ast_result.source_available:
            analysis.source = "unavailable"
            analysis.confidence = "opaque"
            analysis.trace_successful = False
            analysis.warnings.append("Source code not available — node will be kept sequential")
            if all_schema_fields:
                analysis.mutations = AccessSet.from_fields(*all_schema_fields)
                analysis.is_pure = False
            return analysis

        reads = ast_result.reads
        writes = ast_result.writes
        in_place_mutations = ast_result.mutations

        all_mutations = AccessSet()
        for obj, path in writes:
            all_mutations.add(obj, path)
        for obj, path in in_place_mutations:
            all_mutations.add(obj, path)

        analysis.source = "ast"
        analysis.special_calls = ast_result.detected_special_calls

        uncertainty_flags = (ast_result.has_dynamic_keys or ast_result.has_unresolved_calls
                             or ast_result.recursion_depth_hit or ast_result.has_dynamic_exec
                             or ast_result.has_closure_write or ast_result.has_global_write
                             or ast_result.has_unknown_attr_access or ast_result.has_return_lambda_mutates_state
                             or ast_result.has_dynamic_attr)
        warnings_without_writes = not all_mutations and ast_result.warnings

        # Inverted: partial when uncertain, full only when proven safe
        if uncertainty_flags or warnings_without_writes:
            confidence = "partial"
        else:
            confidence = "full"
        analysis.confidence = confidence

        if analysis.confidence != "full" and not all_mutations and all_schema_fields:
            all_mutations = AccessSet.from_fields(*all_schema_fields)
            analysis.warnings.append(f"Confidence {confidence!r} with no detected writes — "
                                     f"conservatively assuming all {len(all_schema_fields)} schema fields")

        analysis.reads = reads
        analysis.writes = writes
        analysis.mutations = all_mutations
        analysis.is_pure = not bool(all_mutations)
        analysis.trace_successful = True
        analysis.warnings.extend(ast_result.warnings)

        return analysis
