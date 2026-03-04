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
"""SchedulingStage: compute branch domains, cycle analysis, and optimized order."""

from __future__ import annotations

import logging
from typing import Any

from nat_app.compiler.compilation_context import CompilationContext
from nat_app.constraints import OptimizationConfig
from nat_app.graph.scheduling import analyze_cycle_body
from nat_app.graph.scheduling import compute_branch_info
from nat_app.graph.scheduling import compute_optimized_order
from nat_app.graph.topology import NodeType

logger = logging.getLogger(__name__)


class SchedulingStage:
    """Compute branch domains, intra-cycle parallelism, and final execution order.

    Reads: ``graph``, ``node_analyses``, ``topology``, ``reducer_fields``,
           ``resolved_constraints``
    Writes: ``branch_info``, ``cycle_body_analyses``, ``optimized_order``
    """

    def __init__(self, config: OptimizationConfig | None = None) -> None:
        self._config = config or OptimizationConfig()

    @property
    def name(self) -> str:
        return "scheduling"

    def apply(self, context: CompilationContext, **kwargs: Any) -> CompilationContext:
        """Compute branch domains, cycle analysis, and optimized execution order.

        Args:
            context: Current compilation context with ``graph``,
                ``node_analyses``, ``topology``, ``reducer_fields``,
                ``resolved_constraints`` in metadata.
            **kwargs: Additional arguments (reserved for future use).

        Returns:
            The updated context with ``branch_info``, ``cycle_body_analyses``,
            and ``optimized_order`` in metadata.
        """
        graph = context.metadata["graph"]
        node_analyses = context.metadata["node_analyses"]
        topology = context.metadata["topology"]
        reducer_fields = context.metadata.get("reducer_fields") or {}
        resolved_constraints = context.metadata.get("resolved_constraints") or {}

        # Branch domain analysis
        branch_info = compute_branch_info(graph, topology)
        if branch_info:
            logger.info("Branch domains computed for %d router(s)", len(branch_info))

        # Intra-cycle parallelism
        cycle_body_analyses: dict[str, Any] = {}
        if topology.cycles and not self._config.disable_parallelization:
            for cycle in topology.cycles:
                body_analysis = analyze_cycle_body(
                    cycle,
                    graph,
                    node_analyses,
                    reducer_fields,
                    resolved_constraints,
                )
                if body_analysis is not None:
                    cycle.body_analysis = body_analysis
                    cycle_body_analyses[cycle.entry_node] = body_analysis
                    if body_analysis.has_parallelism:
                        logger.info(
                            "Intra-cycle parallelism: entry=%s, %d stages",
                            cycle.entry_node,
                            len(body_analysis.stages),
                        )
                        for node in body_analysis.body_nodes:
                            topology.node_types[node] = NodeType.CYCLE_MEMBER_PARALLELIZABLE

        # Final execution order
        optimized_order = compute_optimized_order(
            graph,
            node_analyses,
            topology,
            resolved_constraints=resolved_constraints,
            reducer_fields=reducer_fields,
            branch_info=branch_info,
            disable_parallelization=self._config.disable_parallelization,
        )

        context.metadata["branch_info"] = branch_info
        context.metadata["cycle_body_analyses"] = cycle_body_analyses
        context.metadata["optimized_order"] = optimized_order

        return context
