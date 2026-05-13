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
Result types for graph analysis and scheduling.

These dataclasses are the primary output of the compilation pipeline and
the input to framework builders.  They are separated from the algorithm
functions in ``nat_app.graph.scheduling`` so that builder modules can
import lightweight types without pulling in the full scheduling machinery.

For backward compatibility, all types are re-exported from
``nat_app.graph.scheduling``.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum

from nat_app.constraints.models import ResolvedConstraints
from nat_app.graph.access import ReducerSet
from nat_app.graph.analysis import NodeAnalysis
from nat_app.graph.topology import CycleBodyAnalysis
from nat_app.graph.topology import GraphTopology
from nat_app.graph.types import Graph


class EdgeType(Enum):
    """Classification of graph edges."""

    NECESSARY = "necessary"
    UNNECESSARY = "unnecessary"
    CONDITIONAL = "conditional"
    UNKNOWN = "unknown"


@dataclass
class EdgeAnalysis:
    """Analysis of a single edge."""

    source: str
    target: str
    edge_type: EdgeType
    reason: str = ""
    shared_fields: set[str] = field(default_factory=set)


@dataclass
class BranchInfo:
    """Branch domain information for a single router."""

    router_node: str
    branches: dict[str, set[str]]
    """label -> set of nodes exclusively reachable from that branch.

    When the graph uses conditional edges, keys are decision labels
    (e.g. ``"left"``, ``"right"``).  For 1-to-many routing where a
    single label maps to multiple targets, BFS starts from the full
    target group so all reachable nodes are captured under one key.
    """

    merge_nodes: set[str]
    """Nodes reachable from multiple branches (shared downstream)."""

    all_downstream: set[str]
    """All nodes downstream of this router."""


@dataclass
class CompilationResult:
    """Core compilation output consumed by framework builders.

    Contains the optimized execution schedule and the graph/analysis
    data that builders need to construct optimized framework artifacts.
    """

    graph: Graph
    """The analyzed graph."""

    node_analyses: dict[str, NodeAnalysis]
    """Per-node read/write analysis."""

    necessary_edges: set[tuple[str, str]]
    unnecessary_edges: set[tuple[str, str]]

    optimized_order: list[set[str]]
    """Execution stages: each set can run in parallel."""

    topology: GraphTopology | None = None
    branch_info: dict[str, BranchInfo] = field(default_factory=dict)
    cycle_body_analyses: dict[str, CycleBodyAnalysis] = field(default_factory=dict)

    @property
    def stages(self) -> list[set[str]]:
        """Alias for optimized_order.

        Returns:
            The optimized execution order as a list of parallel stages.
        """
        return self.optimized_order

    @property
    def speedup_estimate(self) -> float:
        """Estimated speedup from parallelization.

        Returns:
            Ratio of sequential node count to parallel stage count.
        """
        sequential = sum(len(s) for s in self.optimized_order)
        parallel = len(self.optimized_order)
        return sequential / parallel if parallel else 1.0


@dataclass
class TransformationResult(CompilationResult):
    """Full analysis output including diagnostic/debugging data.

    Extends ``CompilationResult`` with analysis artifacts useful for
    debugging, visualization, and constraint inspection.
    """

    edge_analyses: list[EdgeAnalysis] = field(default_factory=list)
    """Classification of each edge."""

    parallel_groups: list[set[str]] = field(default_factory=list)
    """Groups of nodes that can run in parallel."""

    state_evolution: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    """node -> {"reads": fields, "writes": fields}."""

    resolved_constraints: dict[str, ResolvedConstraints] | None = None
    reducer_fields: ReducerSet = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
