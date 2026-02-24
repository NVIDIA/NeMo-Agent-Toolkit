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
Node-level analysis: read/write profiling, conflict detection, and dependency graphs.

This module provides the framework-agnostic analysis primitives that underpin
graph optimization.  Framework packages use static analysis (or their own
introspection) to populate ``NodeAnalysis`` objects, then use the
functions here to build dependency graphs and find parallel groups.

All functions operate on abstract data structures — no framework imports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field
from typing import Literal

from nat_app.graph.access import AccessSet
from nat_app.graph.access import ReducerSet

logger = logging.getLogger(__name__)


@dataclass
class NodeAnalysis:
    """
    Complete analysis for a single graph node.

    Read/write tracking uses ``AccessSet`` which
    supports multiple state objects and nested path overlap detection.
    For single-object frameworks (e.g. LangGraph), use
    ``AccessSet.from_fields("query", "messages")`` or the ``state_reads`` /
    ``state_writes`` backward-compat properties.
    """

    name: str
    """Node name in the graph."""

    reads: AccessSet = field(default_factory=AccessSet)
    """All state objects/fields this node reads."""

    writes: AccessSet = field(default_factory=AccessSet)
    """State objects/fields this node writes (return dict keys)."""

    mutations: AccessSet = field(default_factory=AccessSet)
    """All mutation points: writes | in-place mutations."""

    confidence: Literal["full", "partial", "opaque"] = "full"
    """Analysis confidence: "full" (all reads/writes determined), "partial"
    (incomplete -- dynamic keys, unresolved calls, or recursion limit),
    or "opaque" (source unavailable or analysis failed)."""

    source: str = "unknown"
    """Analysis source: "ast", "runtime", "subgraph_schema", "unknown"."""

    special_calls: set[str] = field(default_factory=set)
    """Framework-specific special calls detected (e.g. "Send", "Command" for LangGraph)."""

    has_side_effects: bool = False
    """True if node has known external side effects."""

    is_pure: bool = True
    """True if no state mutations were detected."""

    trace_successful: bool = True
    """Whether analysis completed without critical failures."""

    exceptions: list[tuple[str, str]] = field(default_factory=list)
    """Exceptions encountered during analysis (informational)."""

    warnings: list[str] = field(default_factory=list)
    """Diagnostic messages from analysis."""

    # -- Convenience properties for single-object frameworks ----------------

    @property
    def state_reads(self) -> set[str]:
        """Flat field names this node reads (single-object compat)."""
        return self.reads.all_fields_flat

    @state_reads.setter
    def state_reads(self, value: set[str]) -> None:
        self.reads = AccessSet.from_set(value)

    @property
    def state_writes(self) -> set[str]:
        """Flat field names this node writes (single-object compat)."""
        return self.writes.all_fields_flat

    @state_writes.setter
    def state_writes(self, value: set[str]) -> None:
        self.writes = AccessSet.from_set(value)

    # -- Conflict detection ------------------------------------------------

    def conflicts_with(
        self,
        other: NodeAnalysis,
        reducer_fields: ReducerSet | None = None,
    ) -> bool:
        """
        Check if this node conflicts with another (can't run in parallel).

        Two nodes conflict if:
        1. Either has special calls that act as optimization barriers.
        2. One writes to a field the other reads (read-write conflict).
        3. Both write to the same field (write-write conflict)
           UNLESS the field has a reducer (safe for parallel appends).

        Args:
            other: The other node to check against.
            reducer_fields: Per-object reducer fields (parallel-safe writes).

        Returns:
            ``True`` if the nodes conflict and cannot run in parallel.
        """
        barrier_calls = self.special_calls | other.special_calls
        if barrier_calls:
            return True

        reducers = reducer_fields or {}

        my_writes = self.mutations - reducers
        their_writes = other.mutations - reducers
        if my_writes & their_writes:
            return True

        if self.mutations.overlaps(other.reads):
            return True
        if other.mutations.overlaps(self.reads):
            return True

        return False

    def __repr__(self) -> str:
        return (f"NodeAnalysis({self.name}, "
                f"reads={len(self.reads)}, "
                f"writes={len(self.mutations)}, "
                f"confidence={self.confidence}, "
                f"source={self.source!r})")


# ---------------------------------------------------------------------------
# Dependency graph construction
# ---------------------------------------------------------------------------


def build_dependency_graph(
    analyses: dict[str, NodeAnalysis],
    reducer_fields: ReducerSet | None = None,
) -> dict[str, set[str]]:
    """Build a node dependency graph from per-node analyses.

    A node B depends on node A if A writes to a field that B reads
    (excluding reducer fields, which are safe for parallel writes).

    Args:
        analyses: Per-node analysis results keyed by node name.
        reducer_fields: Per-object reducer fields (parallel-safe writes).

    Returns:
        Mapping of each node to the set of nodes it depends on.
    """
    reducers = reducer_fields or {}
    dependencies: dict[str, set[str]] = {name: set() for name in analyses}

    writes_no_reducer = {name: analysis.mutations - reducers for name, analysis in analyses.items()}

    for node_name, analysis in analyses.items():
        for other_name in analyses:
            if other_name == node_name:
                continue
            if writes_no_reducer[other_name].overlaps(analysis.reads):
                dependencies[node_name].add(other_name)

    return dependencies


# ---------------------------------------------------------------------------
# Parallel group finding
# ---------------------------------------------------------------------------


def find_parallel_groups(
    analyses: dict[str, NodeAnalysis],
    dependencies: dict[str, set[str]],
    reducer_fields: ReducerSet | None = None,
) -> list[set[str]]:
    """Find maximal groups of nodes that can run in parallel.

    Nodes are grouped together if they have no mutual data conflicts
    and no mutual dependencies.

    Args:
        analyses: Per-node analysis results keyed by node name.
        dependencies: Node dependency graph from ``build_dependency_graph``.
        reducer_fields: Per-object reducer fields (parallel-safe writes).

    Returns:
        List of node sets, each containing nodes that can run in parallel.
    """
    reducers = reducer_fields or {}
    nodes = list(analyses.keys())
    independent_pairs: list[tuple[str, str]] = []

    conflict_cache: dict[tuple[str, str], bool] = {}

    def cached_conflicts(a_name: str, b_name: str) -> bool:
        key = (min(a_name, b_name), max(a_name, b_name))
        if key not in conflict_cache:
            conflict_cache[key] = analyses[key[0]].conflicts_with(
                analyses[key[1]],
                reducers,
            )
        return conflict_cache[key]

    for i, node_a in enumerate(nodes):
        for node_b in nodes[i + 1:]:
            if not cached_conflicts(node_a, node_b):
                if (node_b not in dependencies.get(node_a, set()) and node_a not in dependencies.get(node_b, set())):
                    independent_pairs.append((node_a, node_b))

    if not independent_pairs:
        return []

    groups = _merge_into_groups(independent_pairs, analyses, reducers, conflict_cache, dependencies)
    return [g for g in groups if len(g) > 1]


def _merge_into_groups(
    pairs: list[tuple[str, str]],
    analyses: dict[str, NodeAnalysis],
    reducer_fields: ReducerSet,
    conflict_cache: dict[tuple[str, str], bool],
    dependencies: dict[str, set[str]],
) -> list[set[str]]:
    """Merge independent pairs into maximal compatible groups.

    Args:
        pairs: Independent node pairs that can run together.
        analyses: Per-node analysis results keyed by node name.
        reducer_fields: Per-object reducer fields (parallel-safe writes).
        conflict_cache: Cached pairwise conflict results.
        dependencies: Node dependency graph.

    Returns:
        List of merged node sets, each containing compatible nodes.
    """
    if not pairs:
        return []
    groups = [set(pair) for pair in pairs]
    changed = True
    while changed:
        changed = False
        new_groups: list[set[str]] = []
        used: set[int] = set()
        for i, group_a in enumerate(groups):
            if i in used:
                continue
            merged = group_a.copy()
            for j, group_b in enumerate(groups[i + 1:], i + 1):
                if j in used:
                    continue
                potential = merged | group_b
                if _group_is_compatible(potential, analyses, reducer_fields, conflict_cache, dependencies):
                    merged = potential
                    used.add(j)
                    changed = True
            new_groups.append(merged)
            used.add(i)
        groups = new_groups
    return groups


def _group_is_compatible(
    group: set[str],
    analyses: dict[str, NodeAnalysis],
    reducer_fields: ReducerSet,
    conflict_cache: dict[tuple[str, str], bool],
    dependencies: dict[str, set[str]],
) -> bool:
    """Check if all nodes in a group are pairwise non-conflicting.

    Args:
        group: Set of node names to check.
        analyses: Per-node analysis results keyed by node name.
        reducer_fields: Per-object reducer fields (parallel-safe writes).
        conflict_cache: Cached pairwise conflict results.
        dependencies: Node dependency graph.

    Returns:
        ``True`` if all nodes in the group are pairwise non-conflicting.
    """
    nodes = list(group)
    for i, a_name in enumerate(nodes):
        for b_name in nodes[i + 1:]:
            if b_name in dependencies.get(a_name, set()) or a_name in dependencies.get(b_name, set()):
                return False
            key = (min(a_name, b_name), max(a_name, b_name))
            if key not in conflict_cache:
                conflict_cache[key] = analyses[key[0]].conflicts_with(
                    analyses[key[1]],
                    reducer_fields,
                )
            if conflict_cache[key]:
                return False
    return True


# ---------------------------------------------------------------------------
# Graph-level analysis result
# ---------------------------------------------------------------------------


@dataclass
class ParallelizationOpportunity:
    """A group of nodes that can potentially run in parallel."""

    nodes: set[str]
    reason: str = ""
    confidence: Literal["full", "partial", "opaque"] = "full"
    preconditions: list[str] = field(default_factory=list)


@dataclass
class GraphAnalysisResult:
    """Complete analysis of a graph for parallelization."""

    node_analyses: dict[str, NodeAnalysis] = field(default_factory=dict)
    parallelizable_groups: list[ParallelizationOpportunity] = field(default_factory=list)
    dependency_graph: dict[str, set[str]] = field(default_factory=dict)
    reducer_fields: ReducerSet = field(default_factory=dict)
    total_nodes: int = 0
    pure_nodes: int = 0
    warnings: list[str] = field(default_factory=list)

    def get_execution_order(self) -> list[set[str]]:
        """Topological sort with parallel grouping.

        Returns:
            Execution stages where each set of nodes can run in parallel.
        """
        remaining = set(self.node_analyses.keys())
        order: list[set[str]] = []
        while remaining:
            ready = set()
            for node in remaining:
                deps = self.dependency_graph.get(node, set())
                if not (deps & remaining):
                    ready.add(node)
            if not ready:
                logger.warning("Circular dependency in: %s", remaining)
                order.append(remaining)
                break
            order.append(ready)
            remaining -= ready
        return order
