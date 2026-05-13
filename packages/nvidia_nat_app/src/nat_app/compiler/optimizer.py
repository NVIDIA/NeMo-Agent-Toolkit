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
One-call graph optimization orchestrator.

`GraphOptimizer` provides a simple one-call API that wraps the
`DefaultGraphCompiler` pipeline.  For advanced use
cases (custom stages, inter-stage communication), use ``DefaultGraphCompiler``
directly.

Example:

    from nat_app.compiler.optimizer import GraphOptimizer

    optimizer = GraphOptimizer(adapter=MyAdapter())
    # One-call path:
    optimized = optimizer.optimize_and_build(my_graph)
    # Or two-step (when you need the TransformationResult):
    result = optimizer.optimize(my_graph)
    optimized = optimizer.adapter.build(my_graph, result)
"""

from __future__ import annotations

import logging
from typing import Any

from nat_app.compiler.default_graph_compiler import DefaultGraphCompiler
from nat_app.compiler.default_graph_compiler import context_to_result
from nat_app.constraints import OptimizationConfig
from nat_app.graph.adapter import AbstractFrameworkAdapter
from nat_app.graph.models import TransformationResult

logger = logging.getLogger(__name__)


class GraphOptimizer:
    """One-call graph optimization using a framework adapter.

    This is a convenience wrapper around `DefaultGraphCompiler`.
    It runs the standard 6-stage pipeline and returns a
    `TransformationResult`.

    For custom stages or inter-stage data sharing, use ``DefaultGraphCompiler`` directly.

    Example:

        optimizer = GraphOptimizer(adapter=MyCrewAIAdapter())
        optimized = optimizer.optimize_and_build(my_crew_graph)
    """

    def __init__(
        self,
        adapter: AbstractFrameworkAdapter,
        config: OptimizationConfig | None = None,
    ) -> None:
        self.adapter = adapter
        self.config = config or OptimizationConfig()

    def optimize(self, source: Any) -> TransformationResult:
        """Extract, analyze, and compute optimized execution order.

        Args:
            source: The framework's graph artifact.

        Returns:
            A `TransformationResult` containing the optimized execution
            order, node analyses, edge classifications, and more.

        Raises:
            GraphValidationError: If the adapter produces an invalid Graph.
        """
        logger.info("Starting graph optimization...")

        compiler = DefaultGraphCompiler(self.adapter, self.config)
        context = compiler.compile(source)

        return context_to_result(context)

    def optimize_and_build(self, source: Any) -> Any:
        """Optimize the graph and build the framework artifact in one call.

        Equivalent to: self.adapter.build(source, self.optimize(source))

        Args:
            source: The framework's graph artifact.

        Returns:
            The optimized framework artifact from adapter.build().
        """
        result = self.optimize(source)
        return self.adapter.build(source, result)
