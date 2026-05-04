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
Speculation planner: composes multiple strategies with conflict resolution.

The ``SpeculationPlanner`` orchestrates one or more ``SpeculationStrategy``
implementations, resolving conflicts when strategies compete for the same
target nodes via priority-based claiming.
"""

from __future__ import annotations

import logging

from nat_app.graph.topology import GraphTopology
from nat_app.graph.topology import analyze_graph_topology
from nat_app.graph.types import Graph
from nat_app.speculation.plan import SpeculationPlan
from nat_app.speculation.safety import SpeculationSafetyConfig
from nat_app.speculation.strategies.base import SpeculationStrategy

logger = logging.getLogger(__name__)


class SpeculationPlanner:
    """Composes multiple speculation strategies with conflict resolution.

    Strategies are evaluated in priority order (highest first).  When
    multiple strategies target the same nodes, the higher-priority
    strategy claims them and the lower-priority strategy's overlapping
    opportunities are filtered out.

    Example::

        from nat_app.speculation.planner import SpeculationPlanner
        from nat_app.speculation.strategies import RouterBranchStrategy

        planner = SpeculationPlanner([RouterBranchStrategy()])
        plans = planner.plan(graph, safety)
    """

    def __init__(self, strategies: list[SpeculationStrategy]) -> None:
        self._strategies = sorted(strategies, key=lambda s: s.priority, reverse=True)

    def plan(
        self,
        graph: Graph,
        safety: SpeculationSafetyConfig | None = None,
        topology: GraphTopology | None = None,
    ) -> list[SpeculationPlan]:
        """Produce speculation plans by composing all registered strategies.

        Args:
            graph: The abstract graph representation.
            safety: Safety configuration for excluding unsafe nodes.
            topology: Pre-computed topology (computed on-demand if ``None``).

        Returns:
            A list of ``SpeculationPlan`` objects from all strategies,
            with conflicts resolved by priority.
        """
        safety = safety or SpeculationSafetyConfig()
        if topology is None:
            topology = analyze_graph_topology(graph)

        all_plans: list[SpeculationPlan] = []
        claimed_nodes: set[str] = set()

        for strategy in self._strategies:
            opportunities = strategy.identify(graph, topology)

            filtered = [opp for opp in opportunities if not opp.candidate_targets & claimed_nodes]

            if not filtered:
                continue

            plans = strategy.plan(filtered, safety, graph)

            for p in plans:
                claimed_nodes |= p.targets_to_launch
                logger.debug(
                    "Strategy '%s' claims %d targets for decision_node '%s'",
                    strategy.name,
                    len(p.targets_to_launch),
                    p.decision_node,
                )

            all_plans.extend(plans)

        return all_plans
