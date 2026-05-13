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
Cycle and router detection for graph optimization.

This module identifies:
1. Routers (conditional edges) — decision points
2. Cycles (back edges) — loops that require special handling
3. Optimization boundaries — where parallelization should stop

Cycle detection uses **Tarjan's strongly-connected-components (SCC)**
algorithm, which correctly handles overlapping and nested cycles.
A single SCC may contain multiple elementary cycles; each back-edge
within the SCC produces a separate `CycleInfo`.  Consequently a node
may appear in more than one `CycleInfo` when cycles share nodes.

All functions operate on the abstract `Graph`
type.  No framework-specific imports.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum

from nat_app.graph.types import Graph

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Classification of nodes for optimization."""

    REGULAR = "regular"
    ROUTER = "router"
    CYCLE_MEMBER = "cycle_member"
    CYCLE_MEMBER_PARALLELIZABLE = "cycle_member_par"
    CYCLE_ENTRY = "cycle_entry"
    CYCLE_EXIT = "cycle_exit"


@dataclass
class CycleBodyAnalysis:
    """Intra-cycle parallelization analysis.

    Captures which nodes inside a cycle body can be run in parallel
    within a single loop iteration.
    """

    body_nodes: set[str]
    """Nodes eligible for intra-cycle parallelism (excludes entry/exit)."""

    stages: list[set[str]]
    """Parallel execution stages within one iteration (body nodes only)."""

    entry_node: str
    """Must run first each iteration."""

    exit_node: str
    """Must run last each iteration."""

    has_parallelism: bool
    """True if at least one stage contains more than one node."""

    needs_synthetic_entry: bool = False
    """True when the original entry was absorbed into the body."""


@dataclass
class CycleInfo:
    """Information about a detected cycle.

    A node may appear in multiple ``CycleInfo`` objects when cycles
    overlap or nest (e.g. an inner refinement loop inside an outer
    retry loop that share a common decision node).
    """

    nodes: set[str]
    entry_node: str
    exit_node: str
    back_edge: tuple[str, str]
    body_analysis: CycleBodyAnalysis | None = None


@dataclass
class RouterInfo:
    """Information about a router node."""

    node: str
    branches: dict[str, list[str]]
    is_cycle_exit: bool = False


@dataclass
class GraphTopology:
    """Complete topological analysis of a graph."""

    nodes: set[str]
    edges: list[tuple[str, str]]

    node_types: dict[str, NodeType]
    routers: list[RouterInfo]
    cycles: list[CycleInfo]

    parallelizable_regions: list[set[str]]
    sequential_regions: list[set[str]]


# ---------------------------------------------------------------------------
# Adjacency helpers
# ---------------------------------------------------------------------------


def _build_adj(
    nodes: set[str],
    edges: list[tuple[str, str]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Build forward and reverse adjacency dicts in a single pass."""
    fwd: dict[str, list[str]] = {n: [] for n in nodes}
    rev: dict[str, list[str]] = {n: [] for n in nodes}
    for src, dst in edges:
        if src in fwd:
            fwd[src].append(dst)
        if dst in rev:
            rev[dst].append(src)
    return fwd, rev


# ---------------------------------------------------------------------------
# Core algorithms
# ---------------------------------------------------------------------------


def detect_cycles(graph: Graph) -> list[CycleInfo]:
    """Detect all cycles in *graph* via Tarjan's SCC algorithm.

    Uses strongly-connected-component decomposition so that
    overlapping and nested cycles are all discovered.  Within each
    SCC that contains more than one node, every back-edge (an edge
    whose target is an ancestor in the DFS tree) produces a separate
    `CycleInfo` with the minimal cycle path for that back-edge.

    Operates on the abstract ``Graph`` type.

    Args:
        graph: The graph to analyze for cycles.

    Returns:
        List of `CycleInfo` for every elementary cycle found.
        A node may appear in multiple entries when cycles overlap.
    """
    nodes = graph.node_names
    edges = graph.edge_pairs
    adj, rev_adj = _build_adj(nodes, edges)

    # BFS from graph entry to establish natural visit order.
    # Used to pick the DFS root within each SCC so that back-edge
    # classification aligns with the graph's intended execution flow.
    entry_order: dict[str, int] = {}
    if graph.entry_point:
        bfs_q: deque[str] = deque([graph.entry_point])
        bfs_seen: set[str] = set()
        idx = 0
        while bfs_q:
            n = bfs_q.popleft()
            if n in bfs_seen:
                continue
            bfs_seen.add(n)
            entry_order[n] = idx
            idx += 1
            for succ in adj.get(n, []):
                if succ not in bfs_seen:
                    bfs_q.append(succ)

    # --- Tarjan's SCC (iterative to avoid recursion-limit issues) ---
    index_counter = [0]
    node_index: dict[str, int] = {}
    node_lowlink: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    sccs: list[set[str]] = []

    def _strongconnect(v: str) -> None:
        work: list[tuple[str, int]] = [(v, 0)]
        node_index[v] = node_lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        while work:
            node, ni = work[-1]
            neighbors = adj.get(node, [])
            if ni < len(neighbors):
                work[-1] = (node, ni + 1)
                w = neighbors[ni]
                if w not in node_index:
                    node_index[w] = node_lowlink[w] = index_counter[0]
                    index_counter[0] += 1
                    stack.append(w)
                    on_stack.add(w)
                    work.append((w, 0))
                elif w in on_stack:
                    node_lowlink[node] = min(node_lowlink[node], node_index[w])
            else:
                if node_lowlink[node] == node_index[node]:
                    scc: set[str] = set()
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        scc.add(w)
                        if w == node:
                            break
                    if len(scc) > 1:
                        sccs.append(scc)
                    elif len(scc) == 1:
                        n = next(iter(scc))
                        if n in adj.get(n, []):
                            sccs.append(scc)
                work.pop()
                if work:
                    parent = work[-1][0]
                    node_lowlink[parent] = min(
                        node_lowlink[parent],
                        node_lowlink[node],
                    )

    for n in sorted(nodes):
        if n not in node_index:
            _strongconnect(n)

    # --- Extract CycleInfo per back-edge within each SCC ---
    cycles: list[CycleInfo] = []
    for scc in sccs:
        back_edges = _find_scc_back_edges(scc, adj, rev_adj, entry_order)
        for exit_node, entry_node in back_edges:
            cycle_nodes = _cycle_path_nodes(
                entry_node,
                exit_node,
                scc,
                adj,
                rev_adj,
            )
            cycles.append(
                CycleInfo(
                    nodes=cycle_nodes,
                    entry_node=entry_node,
                    exit_node=exit_node,
                    back_edge=(exit_node, entry_node),
                ))

    return cycles


def _find_scc_back_edges(
    scc: set[str],
    adj: dict[str, list[str]],
    rev_adj: dict[str, list[str]],
    entry_order: dict[str, int] | None = None,
) -> list[tuple[str, str]]:
    """Find back-edges in *scc* using a DFS tree.

    The DFS root is the SCC node closest to the graph entry point
    (by BFS distance via *entry_order*) among those with an external
    predecessor.  This ensures the cycle entry/exit classification
    aligns with the graph's natural execution flow.

    Returns at least one back-edge for any SCC with >1 node.
    """
    if entry_order is None:
        entry_order = {}

    def _rank(n: str) -> float:
        return entry_order.get(n, float("inf"))

    candidates = [n for n in scc if any(p not in scc for p in rev_adj.get(n, []))]
    if candidates:
        root: str = min(candidates, key=_rank)
    else:
        root = min(scc, key=_rank)

    scc_adj: dict[str, list[str]] = {n: [nb for nb in adj.get(n, []) if nb in scc] for n in scc}

    back_edges: list[tuple[str, str]] = []
    visited: set[str] = set()
    on_stack: set[str] = set()

    work: list[tuple[str, int]] = [(root, 0)]
    visited.add(root)
    on_stack.add(root)

    while work:
        node, ni = work[-1]
        neighbors = scc_adj[node]
        if ni < len(neighbors):
            work[-1] = (node, ni + 1)
            neighbor = neighbors[ni]
            if neighbor not in visited:
                visited.add(neighbor)
                on_stack.add(neighbor)
                work.append((neighbor, 0))
            elif neighbor in on_stack:
                back_edges.append((node, neighbor))
        else:
            on_stack.discard(node)
            work.pop()

    if not back_edges:
        if entry_order:
            best_entry = min(scc, key=lambda n: entry_order.get(n, float("inf")))
        else:
            best_entry = min(scc)
        for src in sorted(scc):
            for dst in adj.get(src, []):
                if dst in scc and dst == best_entry:
                    return [(src, dst)]
        # If no edge points to best_entry, fall back to any intra-SCC edge
        for src in sorted(scc):
            for dst in adj.get(src, []):
                if dst in scc:
                    return [(src, dst)]

    return back_edges


def _cycle_path_nodes(
    entry: str,
    exit_node: str,
    scc: set[str],
    adj: dict[str, list[str]],
    rev_adj: dict[str, list[str]],
) -> set[str]:
    """All nodes on ANY path from *entry* to *exit_node* within *scc*.

    Uses forward-backward reachability: a node is part of the cycle
    when it is forward-reachable from *entry* AND backward-reachable
    from *exit_node* (both within the SCC, excluding the back-edge
    direction exit→entry).  This correctly captures parallel branches
    in fan-out/fan-in structures.

    Falls back to the full SCC if no forward path exists.
    """
    if entry == exit_node:
        return {entry}

    is_back_edge = (exit_node, entry)

    # Forward BFS from entry within SCC (excluding the back-edge)
    fwd: set[str] = set()
    q: deque[str] = deque([entry])
    while q:
        n = q.popleft()
        if n in fwd:
            continue
        fwd.add(n)
        for nb in adj.get(n, []):
            if nb in scc and nb not in fwd and (n, nb) != is_back_edge:
                q.append(nb)

    if exit_node not in fwd:
        return set(scc)

    # Backward BFS from exit_node using pre-built rev_adj, scoped to SCC
    bwd: set[str] = set()
    q = deque([exit_node])
    while q:
        n = q.popleft()
        if n in bwd:
            continue
        bwd.add(n)
        for nb in rev_adj.get(n, []):
            if nb in scc and nb not in bwd and (nb, n) != is_back_edge:
                q.append(nb)

    return fwd & bwd


def cycle_node_order(cycle: CycleInfo, edges: list[tuple[str, str]]) -> list[str]:
    """
    Return nodes in a cycle in execution order (entry first, excluding back edge).

    Args:
        cycle: The cycle whose nodes to order.
        edges: All graph edges as (source, target) tuples.

    Returns:
        Cycle nodes ordered from entry, following forward edges.
    """
    cycle_edges = [(s, t) for s, t in edges if s in cycle.nodes and t in cycle.nodes and (s, t) != cycle.back_edge]
    adj, _ = _build_adj(cycle.nodes, cycle_edges)

    order: list[str] = []
    visited: set[str] = set()

    def walk(node: str) -> None:
        if node in visited:
            return
        visited.add(node)
        order.append(node)
        for neighbor in adj.get(node, []):
            walk(neighbor)

    walk(cycle.entry_node)
    for n in cycle.nodes:
        if n not in visited:
            order.append(n)
    return order


def detect_routers(graph: Graph) -> list[RouterInfo]:
    """
    Detect router nodes (nodes with conditional edges).

    Args:
        graph: The graph to scan for routers.

    Returns:
        List of router information for each conditional node.
    """
    routers: list[RouterInfo] = []
    for node, branch_targets in graph.conditional_edge_sources.items():
        routers.append(RouterInfo(
            node=node,
            branches=dict(branch_targets),
            is_cycle_exit=False,
        ))
    return routers


_CYCLE_TYPE_PRIORITY: dict[NodeType, int] = {
    NodeType.CYCLE_ENTRY: 3,
    NodeType.CYCLE_EXIT: 2,
    NodeType.CYCLE_MEMBER: 1,
}


def analyze_graph_topology(graph: Graph) -> GraphTopology:
    """Perform complete topological analysis of a ``Graph``.

    Identifies routers, cycles, and optimization boundaries.

    When a node belongs to multiple overlapping cycles, its
    ``NodeType`` is set to the **most restrictive** classification
    (``CYCLE_ENTRY`` > ``CYCLE_EXIT`` > ``CYCLE_MEMBER``).

    Args:
        graph: The graph to analyze.

    Returns:
        Complete topological analysis with node types, routers, and cycles.
    """
    node_names = graph.node_names
    edges = graph.edge_pairs

    cycles = detect_cycles(graph)
    routers = detect_routers(graph)

    cycle_exits = {c.exit_node for c in cycles}
    for router in routers:
        router.is_cycle_exit = router.node in cycle_exits

    node_types: dict[str, NodeType] = {}
    for name in node_names:
        node_types[name] = NodeType.REGULAR

    for router in routers:
        node_types[router.node] = NodeType.ROUTER

    for cycle in cycles:
        for node in cycle.nodes:
            if node == cycle.entry_node:
                candidate = NodeType.CYCLE_ENTRY
            elif node == cycle.exit_node:
                candidate = NodeType.CYCLE_EXIT
            else:
                candidate = NodeType.CYCLE_MEMBER
            existing = node_types.get(node, NodeType.REGULAR)
            existing_pri = _CYCLE_TYPE_PRIORITY.get(existing, 0)
            candidate_pri = _CYCLE_TYPE_PRIORITY.get(candidate, 0)
            if candidate_pri > existing_pri:
                node_types[node] = candidate

    parallelizable: list[set[str]] = []
    sequential: list[set[str]] = []

    cycle_nodes: set[str] = set()
    for cycle in cycles:
        cycle_nodes.update(cycle.nodes)
        sequential.append(cycle.nodes)

    non_cycle_nodes = node_names - cycle_nodes
    if non_cycle_nodes:
        parallelizable.append(non_cycle_nodes)

    return GraphTopology(
        nodes=node_names,
        edges=edges,
        node_types=node_types,
        routers=routers,
        cycles=cycles,
        parallelizable_regions=parallelizable,
        sequential_regions=sequential,
    )


def find_router_chains(topology: GraphTopology) -> list[list[str]]:
    """Identify contiguous sequences of routers where one feeds into the next.

    A chain ``[R1, R2, R3]`` means R1 has a branch target that is R2,
    and R2 has a branch target that is R3.  Each router appears in at
    most one chain.

    Args:
        topology: Topological analysis containing router information.

    Returns:
        List of chains, each a list of router node names in order.
        Standalone routers (not chained) are not included.
    """
    router_set = {r.node for r in topology.routers}
    if len(router_set) < 2:
        return []

    branch_lookup = {r.node: r.branches for r in topology.routers}

    successor: dict[str, str | None] = {}
    for rnode, branches in branch_lookup.items():
        all_targets: set[str] = set()
        for targets in branches.values():
            all_targets.update(targets)
        router_targets = all_targets & router_set
        successor[rnode] = router_targets.pop() if len(router_targets) == 1 else None

    has_predecessor = {s for s in successor.values() if s is not None}

    chains: list[list[str]] = []
    visited: set[str] = set()
    for rnode in router_set:
        if rnode in visited or rnode in has_predecessor:
            continue
        chain = [rnode]
        visited.add(rnode)
        current = rnode
        while successor.get(current) is not None:
            nxt = successor[current]
            assert nxt is not None
            if nxt in visited:
                break
            chain.append(nxt)
            visited.add(nxt)
            current = nxt
        if len(chain) > 1:
            chains.append(chain)

    return chains


def get_safe_parallelization_groups(
    topology: GraphTopology,
    data_dependencies: dict[str, set[str]],
) -> list[set[str]]:
    """
    Get groups of nodes that can safely be parallelized.

    Takes into account data dependencies, cycle boundaries, and router branches.

    Args:
        topology: Topological analysis with node types and boundaries.
        data_dependencies: Per-node sets of nodes it depends on.

    Returns:
        List of node sets that can safely execute in parallel.
    """
    parallelizable = {n for n, t in topology.node_types.items() if t == NodeType.REGULAR}

    remaining = parallelizable.copy()
    completed: set[str] = set()
    safe_groups: list[set[str]] = []

    while remaining:
        ready: set[str] = set()
        for node in sorted(remaining):
            deps = data_dependencies.get(node, set())
            relevant_deps = deps & parallelizable
            if relevant_deps <= completed:
                ready.add(node)

        if not ready:
            logger.warning(
                "Dependency cycle detected in parallelizable nodes; falling back to sequential groups: %s",
                sorted(remaining),
            )
            safe_groups.extend({n} for n in sorted(remaining))
            return safe_groups

        safe_groups.append(ready)
        completed.update(ready)
        remaining -= ready

    return safe_groups
