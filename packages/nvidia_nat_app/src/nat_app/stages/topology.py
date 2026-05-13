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
"""TopologyStage: analyze graph topology (cycles, routers)."""

from __future__ import annotations

import logging
from typing import Any

from nat_app.compiler.compilation_context import CompilationContext
from nat_app.graph.topology import analyze_graph_topology

logger = logging.getLogger(__name__)


class TopologyStage:
    """Detect cycles, routers, and node types in the graph.

    Reads: ``graph``
    Writes: ``topology``
    """

    @property
    def name(self) -> str:
        return "topology"

    def apply(self, context: CompilationContext, **kwargs: Any) -> CompilationContext:
        """Analyze graph topology (cycles, routers) and store in metadata.

        Args:
            context: Current compilation context with ``graph`` in metadata.
            **kwargs: Additional arguments (reserved for future use).

        Returns:
            The updated context with ``topology`` in metadata.
        """
        graph = context.metadata["graph"]
        topology = analyze_graph_topology(graph)

        if topology.cycles:
            logger.info("Topology: %d cycle(s) detected", len(topology.cycles))
        if topology.routers:
            logger.info("Topology: %d router(s)", len(topology.routers))

        context.metadata["topology"] = topology
        return context
