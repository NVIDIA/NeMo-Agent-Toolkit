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
DefaultGraphCompiler: the standard graph optimization pipeline for all frameworks.

Chains the 6 built-in stages (extract, validate, topology, node analysis,
edge classification, scheduling) using the adapter's hooks.  Framework
packages can extend this by appending or inserting custom stages.

Priority-aware pipelines opt in by inserting ``LLMAnalysisStage`` and
``PriorityAssignmentStage``:

    from nat_app.compiler import DefaultGraphCompiler
    from nat_app.stages import LLMAnalysisStage, PriorityAssignmentStage

    compiler = DefaultGraphCompiler(adapter)
    compiler.insert_stage_after("edge_classification", LLMAnalysisStage(adapter))
    compiler.insert_stage_after("llm_analysis", PriorityAssignmentStage())

Example:

    from nat_app.compiler import DefaultGraphCompiler
    from my_framework import MyAdapter

    compiler = DefaultGraphCompiler(MyAdapter())
    context = compiler.compile(my_graph)
    optimized_order = context.optimized_order
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from nat_app.compiler.compilation_context import CompilationContext
from nat_app.compiler.compilation_stage import CompilationStage
from nat_app.compiler.pipelined_compiler import AbstractPipelinedCompiler
from nat_app.constraints import OptimizationConfig
from nat_app.graph.adapter import AbstractFrameworkAdapter
from nat_app.graph.models import TransformationResult
from nat_app.stages import EdgeClassificationStage
from nat_app.stages import ExtractStage
from nat_app.stages import NodeAnalysisStage
from nat_app.stages import SchedulingStage
from nat_app.stages import TopologyStage
from nat_app.stages import ValidateStage

logger = logging.getLogger(__name__)


class DefaultGraphCompiler(AbstractPipelinedCompiler[Any, Any]):
    """Standard graph optimization pipeline using built-in stages.

    Provides the 6-stage default pipeline:

    1. ExtractStage -- extract Graph from source via adapter
    2. ValidateStage -- validate graph structure
    3. TopologyStage -- detect cycles, routers
    4. NodeAnalysisStage -- static analysis of node functions
    5. EdgeClassificationStage -- classify edges, find parallel groups
    6. SchedulingStage -- compute branch domains, cycle analysis, execution order

    Framework packages extend by appending or inserting stages:

        compiler = DefaultGraphCompiler(adapter)
        compiler.append_stage(MyBuildStage())
        result = compiler.compile(source)

    To enable priority assignment, insert the LLM analysis and priority
    stages after edge classification:

        from nat_app.stages import LLMAnalysisStage, PriorityAssignmentStage

        compiler.insert_stage_after("edge_classification", LLMAnalysisStage(adapter))
        compiler.insert_stage_after("llm_analysis", PriorityAssignmentStage())
    """

    def __init__(
        self,
        adapter: AbstractFrameworkAdapter,
        config: OptimizationConfig | None = None,
        stages: Sequence[CompilationStage] | None = None,
    ) -> None:
        self.adapter = adapter
        self.config = config or OptimizationConfig()
        super().__init__(stages=stages)

    def default_stages(self) -> Sequence[CompilationStage]:
        """Return the default 6-stage pipeline (extract through scheduling).

        Returns:
            Sequence of compilation stages in execution order.
        """
        return [
            ExtractStage(self.adapter),
            ValidateStage(),
            TopologyStage(),
            NodeAnalysisStage(self.adapter, self.config),
            EdgeClassificationStage(),
            SchedulingStage(self.config),
        ]

    def prepare(self, source: Any, **kwargs: Any) -> Any:
        """Prepare the source for compilation (no-op by default).

        Args:
            source: The framework-specific graph to compile.
            **kwargs: Additional arguments (reserved for future use).

        Returns:
            The source, possibly transformed. Default implementation
            returns it unchanged.
        """
        return source

    def finalize(
        self,
        context: CompilationContext,
        **kwargs: Any,
    ) -> CompilationContext:
        """Finalize the compilation context after all stages (no-op by default).

        Args:
            context: The compilation context after all stages have run.
            **kwargs: Additional arguments (reserved for future use).

        Returns:
            The context, possibly modified. Default implementation
            returns it unchanged.
        """
        return context

    def compile_to_result(self, source: Any, **kwargs: Any) -> Any:
        """Compile and return a ``TransformationResult``.

        Convenience method that runs the full pipeline and converts the
        internal ``CompilationContext`` to a ``TransformationResult``
        in one call.

        Args:
            source: The framework-specific graph to compile.

        Returns:
            A ``TransformationResult`` with the optimized execution order.
        """
        context = self.compile(source, **kwargs)
        return context_to_result(context)

    def append_stage(self, stage: CompilationStage) -> None:
        """Append a stage to the end of the pipeline.

        Args:
            stage: The compilation stage to append.

        Returns:
            None. Modifies the pipeline in place.
        """
        self._stages = (*self._stages, stage)

    def insert_stage_after(self, after_name: str, stage: CompilationStage) -> None:
        """Insert a stage after the named stage.

        Args:
            after_name: The ``name`` of the existing stage to insert after.
            stage: The compilation stage to insert.

        Returns:
            None. Modifies the pipeline in place.
        """
        new_stages = []
        for s in self._stages:
            new_stages.append(s)
            if s.name == after_name:
                new_stages.append(stage)
        self._stages = tuple(new_stages)


def context_to_result(context: CompilationContext) -> TransformationResult:
    """Convert a CompilationContext to a TransformationResult.

    This is the public bridge between the stage-based compilation pipeline
    (which produces a ``CompilationContext``) and
    the ``TransformationResult`` consumed by framework builders.

    Most callers should use ``DefaultGraphCompiler.compile_to_result``
    instead.  Use this function directly when you need to inspect or
    modify the ``CompilationContext`` before converting.

    Args:
        context: The compilation context from the stage-based pipeline.

    Returns:
        A ``TransformationResult`` assembled from the context metadata.
    """
    md = context.metadata
    return TransformationResult(
        graph=md["graph"],
        node_analyses=md["node_analyses"],
        edge_analyses=md.get("edge_analyses", []),
        necessary_edges=md.get("necessary_edges", set()),
        unnecessary_edges=md.get("unnecessary_edges", set()),
        parallel_groups=md.get("parallel_groups", []),
        optimized_order=md.get("optimized_order", []),
        state_evolution=md.get("state_evolution", {}),
        topology=md.get("topology"),
        resolved_constraints=md.get("resolved_constraints", {}),
        reducer_fields=md.get("reducer_fields", {}),
        branch_info=md.get("branch_info", {}),
        cycle_body_analyses=md.get("cycle_body_analyses", {}),
        warnings=md.get("warnings", []),
    )
