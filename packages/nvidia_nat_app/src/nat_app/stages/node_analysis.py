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
"""NodeAnalysisStage: analyze each node's reads, writes, and mutations via static analysis."""

from __future__ import annotations

import logging
from typing import Any

from nat_app.compiler.compilation_context import CompilationContext
from nat_app.constraints import OptimizationConfig
from nat_app.constraints import apply_constraints_to_analysis
from nat_app.graph.adapter import AbstractFrameworkAdapter
from nat_app.graph.analysis import NodeAnalysis

logger = logging.getLogger(__name__)


class NodeAnalysisStage:
    """Analyze each node function for read/write access and resolve constraints.

    Reads: ``graph``, ``state_schema``, ``all_schema_fields``
    Writes: ``node_analyses``, ``node_funcs``, ``resolved_constraints``, ``warnings``
    """

    def __init__(
        self,
        adapter: AbstractFrameworkAdapter,
        config: OptimizationConfig | None = None,
    ) -> None:
        self._adapter = adapter
        self._config = config or OptimizationConfig()

    @property
    def name(self) -> str:
        return "node_analysis"

    def apply(self, context: CompilationContext, **kwargs: Any) -> CompilationContext:
        """Analyze each node for read/write access and resolve constraints.

        Args:
            context: Current compilation context with ``graph``,
                ``state_schema``, ``all_schema_fields`` in metadata.
            **kwargs: Additional arguments (reserved for future use).

        Returns:
            The updated context with ``node_analyses``, ``node_funcs``,
            ``resolved_constraints``, ``state_evolution``, and ``warnings``.
        """
        graph = context.metadata["graph"]
        state_schema = context.metadata.get("state_schema")
        all_schema_fields = context.metadata.get("all_schema_fields")

        analyses: dict[str, NodeAnalysis] = {}
        node_funcs: dict[str, Any] = {}

        for node_name in graph.node_names:
            func = self._adapter.get_node_func(node_name)
            if func is None:
                node_info = graph.get_node(node_name)
                func = node_info.func

            if func is not None:
                node_funcs[node_name] = func
                analyses[node_name] = self._adapter.analyze_node(
                    node_name,
                    func,
                    state_schema,
                    all_schema_fields,
                    config=self._config,
                )
            else:
                analyses[node_name] = NodeAnalysis(name=node_name, confidence="opaque")
                analyses[node_name].warnings.append("No callable found — keeping sequential")

        logger.info("Analyzed %d nodes", len(analyses))

        resolved_constraints, constraint_warnings = apply_constraints_to_analysis(
            analyses, node_funcs, self._config,
        )
        warnings: list[str] = list(constraint_warnings)

        for node_name, analysis in analyses.items():
            if analysis.confidence != "full":
                warnings.append(
                    f"Node '{node_name}': confidence {analysis.confidence!r} — keeping sequential for safety")

        context.metadata["node_analyses"] = analyses
        context.metadata["node_funcs"] = node_funcs
        context.metadata["resolved_constraints"] = resolved_constraints
        context.metadata.setdefault("warnings", []).extend(warnings)

        # State evolution map
        context.metadata["state_evolution"] = {
            node_name: {
                "reads": analysis.reads.all_fields_flat,
                "writes": analysis.mutations.all_fields_flat,
            }
            for node_name, analysis in analyses.items()
        }

        return context
