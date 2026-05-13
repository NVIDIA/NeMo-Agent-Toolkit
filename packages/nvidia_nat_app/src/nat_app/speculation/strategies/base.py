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
Base protocol and data structures for speculation strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol
from typing import runtime_checkable

if TYPE_CHECKING:
    from nat_app.graph.topology import GraphTopology
    from nat_app.graph.types import Graph
    from nat_app.speculation.plan import SpeculationPlan
    from nat_app.speculation.safety import SpeculationSafetyConfig


@dataclass
class SpeculationOpportunity:
    """An identified opportunity for speculative execution.

    Produced by ``SpeculationStrategy.identify()``, consumed by
    ``SpeculationStrategy.plan()`` and the ``SpeculationPlanner``.
    """

    strategy: str
    """Name of the strategy that identified this opportunity."""

    decision_node: str
    """Node whose completion resolves the speculation."""

    candidate_targets: set[str]
    """All nodes that could be launched speculatively."""

    priority: float = 0.0
    """Higher values indicate more benefit from speculation."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Strategy-specific analysis data."""


@runtime_checkable
class SpeculationStrategy(Protocol):
    """Pluggable strategy for identifying and planning speculation.

    Implementations identify opportunities in a graph's topology
    and produce ``SpeculationPlan`` objects with strategy-specific
    ``ResolutionPolicy`` instances.
    """

    @property
    def name(self) -> str:
        """Unique strategy identifier (e.g. ``"router_branch"``)."""
        ...

    @property
    def priority(self) -> int:
        """Strategy priority for conflict resolution (higher = first claim)."""
        ...

    def identify(
        self,
        graph: Graph,
        topology: GraphTopology,
    ) -> list[SpeculationOpportunity]:
        """Identify speculation opportunities in the graph.

        Args:
            graph: The abstract graph representation.
            topology: Pre-computed topological analysis.

        Returns:
            A list of opportunities this strategy can exploit.
        """
        ...

    def plan(
        self,
        opportunities: list[SpeculationOpportunity],
        safety: SpeculationSafetyConfig,
        graph: Graph,
    ) -> list[SpeculationPlan]:
        """Build concrete speculation plans from opportunities.

        Args:
            opportunities: Opportunities identified by ``identify()``.
            safety: Safety configuration for excluding unsafe nodes.
            graph: The abstract graph representation.

        Returns:
            A list of ready-to-execute ``SpeculationPlan`` objects.
        """
        ...
