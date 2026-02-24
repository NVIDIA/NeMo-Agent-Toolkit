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
Execution order optimization: edge classification, branch analysis, and scheduling.

This module takes the analysis results from `nat_app.graph.analysis` and
the topology from `nat_app.graph.topology`, and computes an optimized
execution order (parallel stages) for the graph.

All functions are framework-agnostic -- they operate on the abstract
`Graph` and analysis dataclasses.

Result types (`CompilationResult`, `TransformationResult`, etc.)
are defined in `nat_app.graph.models` and re-exported here for
backward compatibility.
"""

from __future__ import annotations

import logging
from collections import deque

from nat_app.constraints.models import ResolvedConstraints
from nat_app.constraints.resolution import merge_dependencies
from nat_app.graph.access import AccessSet
from nat_app.graph.access import ReducerSet
from nat_app.graph.analysis import NodeAnalysis
from nat_app.graph.models import BranchInfo
from nat_app.graph.models import CompilationResult
from nat_app.graph.models import EdgeAnalysis
from nat_app.graph.models import EdgeType
from nat_app.graph.models import TransformationResult
from nat_app.graph.topology import CycleBodyAnalysis
from nat_app.graph.topology import CycleInfo
from nat_app.graph.topology import GraphTopology
from nat_app.graph.topology import cycle_node_order
from nat_app.graph.types import Graph

__all__ = [
    "BranchInfo",
    "CompilationResult",
    "EdgeAnalysis",
    "EdgeType",
    "TransformationResult",
    "analyze_cycle_body",
    "classify_edges",
    "compute_branch_info",
    "compute_optimized_order",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Edge classification
# ---------------------------------------------------------------------------


def classify_edges(
    graph: Graph,
    node_analyses: dict[str, NodeAnalysis],
    reducer_fields: ReducerSet | None = None,
) -> list[EdgeAnalysis]:
    """
    Classify each edge in the graph as necessary, unnecessary, or conditional.

    An edge is *necessary* if the target reads a field that the source writes.
    An edge is *unnecessary* if there is no data dependency (candidate for parallelization).
    Conditional edges from routers are preserved as-is.

    Args:
        graph: The graph whose edges to classify.
        node_analyses: Per-node analysis results with reads/writes.
        reducer_fields: Fields with reducer semantics (parallel-safe writes).

    Returns:
        List of edge analyses with classification and reason.
    """
    results: list[EdgeAnalysis] = []

    for edge in graph.edges:
        src, tgt = edge.source, edge.target
        if src not in node_analyses or tgt not in node_analyses:
            results.append(EdgeAnalysis(source=src, target=tgt, edge_type=EdgeType.UNKNOWN, reason="Node not analyzed"))
            continue

        src_analysis = node_analyses[src]
        tgt_analysis = node_analyses[tgt]

        if graph.get_conditional_targets(src) is not None:
            results.append(
                EdgeAnalysis(source=src, target=tgt, edge_type=EdgeType.CONDITIONAL, reason="Router conditional edge"))
            continue

        if src_analysis.confidence != "full" or tgt_analysis.confidence != "full":
            results.append(
                EdgeAnalysis(
                    source=src,
                    target=tgt,
                    edge_type=EdgeType.NECESSARY,
                    reason="Incomplete analysis confidence — kept sequential for safety",
                ))
            continue

        if src_analysis.mutations.overlaps(tgt_analysis.reads, exclude_reducers=reducer_fields or {}):
            results.append(
                EdgeAnalysis(
                    source=src,
                    target=tgt,
                    edge_type=EdgeType.NECESSARY,
                    reason="Target reads source output",
                ))
        else:
            results.append(
                EdgeAnalysis(
                    source=src,
                    target=tgt,
                    edge_type=EdgeType.UNNECESSARY,
                    reason="No data dependency detected",
                ))

    return results


# ---------------------------------------------------------------------------
# Branch domain analysis
# ---------------------------------------------------------------------------


def compute_branch_info(
    graph: Graph,
    topology: GraphTopology,
) -> dict[str, BranchInfo]:
    """
    For each router, compute which nodes are exclusively reachable from
    each conditional branch vs. shared (merge) nodes.

    Cycle nodes are excluded from branch domains.

    Args:
        graph: The graph to analyze.
        topology: Topological analysis containing router information.

    Returns:
        Mapping of router node name to its branch information.
    """
    if not topology.routers:
        return {}

    all_cycle_nodes: set[str] = set()
    cycle_back_edges: set[tuple[str, str]] = set()
    for c in topology.cycles:
        all_cycle_nodes.update(c.nodes)
        cycle_back_edges.add(c.back_edge)

    fwd_adj: dict[str, set[str]] = {}
    for src, tgt in graph.edge_pairs:
        if (src, tgt) not in cycle_back_edges:
            fwd_adj.setdefault(src, set()).add(tgt)

    result: dict[str, BranchInfo] = {}

    for router in topology.routers:
        rnode = router.node
        cond_targets = graph.get_conditional_targets(rnode)
        if not cond_targets:
            continue

        label_reachable: dict[str, set[str]] = {}
        for label, targets in cond_targets.items():
            reachable: set[str] = set()
            queue = deque(targets)
            visited: set[str] = set()
            while queue:
                n = queue.popleft()
                if n in visited or n == rnode:
                    continue
                visited.add(n)
                reachable.add(n)
                for succ in fwd_adj.get(n, set()):
                    queue.append(succ)
            immediate = set(targets) & reachable
            label_reachable[label] = (reachable - all_cycle_nodes) | immediate

        all_reach: set[str] = set()
        merge: set[str] = set()
        for reach in label_reachable.values():
            merge |= (all_reach & reach)
            all_reach |= reach

        branches = {label: (r - merge) for label, r in label_reachable.items()}
        if not all_reach:
            continue

        result[rnode] = BranchInfo(
            router_node=rnode,
            branches=branches,
            merge_nodes=merge,
            all_downstream=all_reach,
        )

    return result


# ---------------------------------------------------------------------------
# Intra-cycle parallelism
# ---------------------------------------------------------------------------


def analyze_cycle_body(
    cycle: CycleInfo,
    graph: Graph,
    node_analyses: dict[str, NodeAnalysis],
    reducer_fields: ReducerSet | None = None,
    resolved_constraints: dict[str, ResolvedConstraints] | None = None,
) -> CycleBodyAnalysis | None:
    """
    Compute parallel stages within a single cycle iteration.

    Returns None if the cycle is too small or safety checks fail.

    Args:
        cycle: The cycle to analyze.
        graph: The containing graph.
        node_analyses: Per-node analysis results with reads/writes.
        reducer_fields: Fields with reducer semantics (parallel-safe writes).
        resolved_constraints: Per-node resolved optimization constraints.

    Returns:
        Intra-cycle parallelization analysis, or None if safety checks fail.
    """
    reducers = reducer_fields or {}
    constraints = resolved_constraints or {}

    body_nodes = cycle.nodes - {cycle.entry_node, cycle.exit_node}
    if len(body_nodes) < 2:
        return CycleBodyAnalysis(
            body_nodes=body_nodes,
            stages=[{n} for n in body_nodes] if body_nodes else [],
            entry_node=cycle.entry_node,
            exit_node=cycle.exit_node,
            has_parallelism=False,
        )

    for node in body_nodes:
        analysis = node_analyses.get(node)
        if analysis is None or analysis.confidence != "full" or analysis.special_calls:
            return CycleBodyAnalysis(
                body_nodes=body_nodes,
                stages=[{n} for n in body_nodes],
                entry_node=cycle.entry_node,
                exit_node=cycle.exit_node,
                has_parallelism=False,
            )
        c = constraints.get(node)
        if c and c.force_sequential:
            return CycleBodyAnalysis(
                body_nodes=body_nodes,
                stages=[{n} for n in body_nodes],
                entry_node=cycle.entry_node,
                exit_node=cycle.exit_node,
                has_parallelism=False,
            )

    writes_no_reducer = {n: node_analyses[n].mutations - reducers for n in body_nodes}

    body_deps: dict[str, set[str]] = {n: set() for n in body_nodes}
    for node_a in body_nodes:
        for node_b in body_nodes:
            if node_a == node_b:
                continue
            if writes_no_reducer[node_a].overlaps(node_analyses[node_b].reads):
                body_deps[node_b].add(node_a)

    if resolved_constraints:
        for node in body_nodes:
            rc = resolved_constraints.get(node)
            if rc and rc.explicit_dependencies:
                for dep in rc.explicit_dependencies:
                    if dep in body_nodes:
                        body_deps[node].add(dep)

    remaining = set(body_nodes)
    stages: list[set[str]] = []
    while remaining:
        ready = {n for n in remaining if not (body_deps.get(n, set()) & remaining)}
        if not ready:
            logger.warning("Circular dependency in cycle body: %s", remaining)
            return CycleBodyAnalysis(
                body_nodes=body_nodes,
                stages=[{n} for n in body_nodes],
                entry_node=cycle.entry_node,
                exit_node=cycle.exit_node,
                has_parallelism=False,
            )
        stages.append(ready)
        remaining -= ready

    needs_synthetic_entry = False
    effective_entry = cycle.entry_node

    entry_analysis = node_analyses.get(cycle.entry_node)
    if entry_analysis is not None and stages:
        can_absorb = True
        if graph.get_conditional_targets(cycle.entry_node) is not None:
            can_absorb = False
        if entry_analysis.confidence != "full":
            can_absorb = False
        if entry_analysis.special_calls:
            can_absorb = False
        entry_constraint = constraints.get(cycle.entry_node)
        if entry_constraint and entry_constraint.force_sequential:
            can_absorb = False
        if can_absorb:
            for peer in stages[0]:
                peer_analysis = node_analyses.get(peer)
                if peer_analysis is None or entry_analysis.conflicts_with(peer_analysis, reducers):
                    can_absorb = False
                    break
        if can_absorb:
            stages[0] = stages[0] | {cycle.entry_node}
            body_nodes = body_nodes | {cycle.entry_node}
            effective_entry = f"__cycle_{cycle.entry_node}_entry__"
            needs_synthetic_entry = True
            logger.info(
                "Cycle entry '%s' absorbed into parallel body (synthetic entry: %s)",
                cycle.entry_node,
                effective_entry,
            )

    router_body = {n for n in body_nodes if graph.get_conditional_targets(n) is not None}
    if router_body:
        stages = [s - router_body for s in stages]
        stages = [s for s in stages if s]
        stages.append(router_body)

    has_parallelism = any(len(s) > 1 for s in stages)
    return CycleBodyAnalysis(
        body_nodes=body_nodes,
        stages=stages,
        entry_node=effective_entry,
        exit_node=cycle.exit_node,
        has_parallelism=has_parallelism,
        needs_synthetic_entry=needs_synthetic_entry,
    )


# ---------------------------------------------------------------------------
# Optimized execution order
# ---------------------------------------------------------------------------


def _build_data_dependencies(
    graph: Graph,
    all_node_names: set[str],
    node_analyses: dict[str, NodeAnalysis],
    node_to_cycles: dict[str, set[int]],
    writes_no_reducer: dict[str, AccessSet],
) -> dict[str, set[str]]:
    """Build dependency dict from write/read overlap and write-write conflicts, excluding nodes that share a cycle."""
    dependencies: dict[str, set[str]] = {name: set() for name in all_node_names}
    # Precompute reachability for graph-based ordering of write-write conflicts.
    reachable_from = {n: graph._compute_reachable(n) for n in all_node_names}
    for node_a, analysis_a in node_analyses.items():
        for node_b, analysis_b in node_analyses.items():
            if node_a == node_b:
                continue
            a_cycles = node_to_cycles.get(node_a)
            if a_cycles and a_cycles & node_to_cycles.get(node_b, set()):
                continue
            if writes_no_reducer[node_a].overlaps(analysis_b.reads):
                dependencies[node_b].add(node_a)
            # Write-write conflict: both nodes write overlapping non-reducer fields.
            # Order by graph reachability (upstream first), else lexicographic tiebreaker.
            elif writes_no_reducer[node_a].overlaps(writes_no_reducer[node_b]):
                if node_b in reachable_from.get(node_a, set()):
                    dependencies[node_b].add(node_a)
                elif node_a in reachable_from.get(node_b, set()):
                    dependencies[node_a].add(node_b)
                elif node_a < node_b:
                    dependencies[node_b].add(node_a)
                else:
                    dependencies[node_a].add(node_b)
    return dependencies


def _apply_confidence_fallbacks(
    dependencies: dict[str, set[str]],
    node_analyses: dict[str, NodeAnalysis],
    edge_pairs: list[tuple[str, str]],
    all_cycle_nodes: set[str],
    all_node_names: set[str],
) -> dict[str, set[str]]:
    """Add edge-based deps for nodes with confidence != 'full' and not in cycles."""
    for name, analysis in node_analyses.items():
        if analysis.confidence != "full" and name not in all_cycle_nodes:
            for src, tgt in edge_pairs:
                if tgt == name and src in all_node_names:
                    dependencies[name].add(src)
    return dependencies


def _apply_constraints(
    dependencies: dict[str, set[str]],
    constraints: dict[str, ResolvedConstraints],
    edge_pairs: list[tuple[str, str]],
    all_node_names: set[str],
) -> dict[str, set[str]]:
    """Merge constraint deps and force_sequential fallback for nodes with no deps."""
    if not constraints:
        return dependencies
    dependencies = merge_dependencies(dependencies, constraints)
    for name in all_node_names:
        if name not in dependencies:
            dependencies[name] = set()
    for name, c in constraints.items():
        if c.force_sequential and not c.explicit_dependencies:
            if not dependencies.get(name):
                for src, tgt in edge_pairs:
                    if tgt == name and src in all_node_names:
                        dependencies[name].add(src)
    return dependencies


def _apply_cycle_body_ordering(
    dependencies: dict[str, set[str]],
    topology: GraphTopology,
    edge_pairs: list[tuple[str, str]],
    node_analyses: dict[str, NodeAnalysis],
    writes_no_reducer: dict[str, AccessSet],
) -> dict[str, set[str]]:
    """Apply cycle back-edges, body parallelism vs sequential, cycle_node_order."""
    if not topology.cycles:
        return dependencies
    for cycle in topology.cycles:
        body = cycle.body_analysis
        if body is not None and body.has_parallelism:
            if not body.needs_synthetic_entry:
                for node in body.body_nodes:
                    dependencies[node].add(body.entry_node)
            for node in body.body_nodes:
                dependencies[body.exit_node].add(node)
            for node_a in body.body_nodes:
                if node_a not in writes_no_reducer:
                    continue
                for node_b in body.body_nodes:
                    if node_a == node_b:
                        continue
                    b_analysis = node_analyses.get(node_b)
                    if b_analysis is None:
                        continue
                    if writes_no_reducer[node_a].overlaps(b_analysis.reads):
                        dependencies[node_b].add(node_a)
        else:
            order = cycle_node_order(cycle, edge_pairs)
            for i in range(len(order) - 1):
                dependencies[order[i + 1]].add(order[i])
    return dependencies


def _apply_cycle_boundary_and_conditional(
    dependencies: dict[str, set[str]],
    graph: Graph,
    edge_pairs: list[tuple[str, str]],
    all_cycle_nodes: set[str],
    cycle_back_edges: set[tuple[str, str]],
) -> dict[str, set[str]]:
    """Add cross-cycle edges and conditional edges from get_conditional_targets."""
    for src, tgt in edge_pairs:
        if (src, tgt) in cycle_back_edges:
            continue
        src_in_cycle = src in all_cycle_nodes
        tgt_in_cycle = tgt in all_cycle_nodes
        if src_in_cycle != tgt_in_cycle:
            dependencies[tgt].add(src)
        cond = graph.get_conditional_targets(src)
        if cond is not None:
            dependencies[tgt].add(src)
    return dependencies


def _apply_branch_dependencies(
    dependencies: dict[str, set[str]],
    branch_info: dict[str, BranchInfo],
    all_cycle_nodes: set[str],
) -> dict[str, set[str]]:
    """Add router as dep for all downstream nodes in branch_info."""
    for rnode, binfo in branch_info.items():
        for node in binfo.all_downstream:
            if node not in all_cycle_nodes:
                dependencies[node].add(rnode)
    return dependencies


def _apply_post_cycle_ordering(
    dependencies: dict[str, set[str]],
    graph: Graph,
    edge_pairs: list[tuple[str, str]],
    all_node_names: set[str],
    all_cycle_nodes: set[str],
    topology: GraphTopology,
) -> dict[str, set[str]]:
    """BFS for pre-cycle nodes; post-cycle nodes depend on cycle exits."""
    if not all_cycle_nodes or not graph.entry_point:
        return dependencies
    adj: dict[str, list[str]] = {}
    for src, tgt in edge_pairs:
        adj.setdefault(src, []).append(tgt)
    pre_cycle_nodes: set[str] = set()
    queue: deque[str] = deque([graph.entry_point])
    visited: set[str] = set()
    while queue:
        node = queue.popleft()
        if node in visited or node in all_cycle_nodes:
            continue
        visited.add(node)
        pre_cycle_nodes.add(node)
        for neighbor in adj.get(node, []):
            queue.append(neighbor)
    post_cycle_nodes = all_node_names - all_cycle_nodes - pre_cycle_nodes
    cycle_exits = {c.exit_node for c in topology.cycles}
    for node in post_cycle_nodes:
        for exit_node in cycle_exits:
            dependencies[node].add(exit_node)
    return dependencies


def _build_parallel_stages(
    all_node_names: set[str],
    dependencies: dict[str, set[str]],
    graph: Graph,
    branch_info: dict[str, BranchInfo],
    constraints: dict[str, ResolvedConstraints],
) -> list[set[str]]:
    """While-loop with _split_by_branch, _stage_order_with_entry_first, force_sequential."""
    node_branch: dict[str, tuple[str, str]] = {}
    for rnode, binfo in branch_info.items():
        for target, exclusive_nodes in binfo.branches.items():
            for n in exclusive_nodes:
                if n not in node_branch:
                    node_branch[n] = (rnode, target)
    remaining = set(all_node_names)
    stages: list[set[str]] = []
    while remaining:
        ready = {n for n in remaining if not (dependencies.get(n, set()) & remaining)}
        if not ready:
            logger.warning("Possible circular dependency in: %s", remaining)
            # Prefer graph's structural entry point when breaking the cycle
            entry = graph.entry_point if graph.entry_point else ""
            ready = {entry} if entry in remaining else {next(iter(remaining))}
        ready = _split_by_branch(ready, node_branch)
        ordered_ready = _stage_order_with_entry_first(ready, graph.entry_point)
        stage: set[str] = set()
        seen_force_sequential = False
        for node in ordered_ready:
            c = constraints.get(node)
            if c and c.force_sequential:
                if seen_force_sequential:
                    continue
                seen_force_sequential = True
            stage.add(node)
        if not stage:
            stage = ready
        stages.append(stage)
        remaining -= stage
    return stages


def compute_optimized_order(
    graph: Graph,
    node_analyses: dict[str, NodeAnalysis],
    topology: GraphTopology,
    resolved_constraints: dict[str, ResolvedConstraints] | None = None,
    reducer_fields: ReducerSet | None = None,
    branch_info: dict[str, BranchInfo] | None = None,
    disable_parallelization: bool = False,
) -> list[set[str]]:
    """Compute optimized execution order based on TRUE dependencies,
    cycles, constraints, confidence, and branch domains.

    Multi-cycle aware: a node that belongs to more than one cycle
    (overlapping / nested cycles) is correctly associated with all
    of them, and intra-cycle dependency skipping only applies when
    both nodes share at least one specific cycle.

    Args:
        graph: The graph to schedule.
        node_analyses: Per-node analysis results with reads/writes. If any graph
            node is missing, it is treated as opaque (confidence="opaque") and
            receives structural dependencies only.
        topology: Topological analysis with cycles and routers.
        resolved_constraints: Per-node resolved optimization constraints.
        reducer_fields: Fields with reducer semantics (parallel-safe writes).
        branch_info: Per-router branch domain information.
        disable_parallelization: If True, emit fully sequential stages.

    Returns:
        List of stages where each stage is a set of nodes that can
        execute in parallel.
    """
    reducers = reducer_fields or {}
    branch_info = branch_info or {}
    constraints = resolved_constraints or {}

    all_node_names = graph.node_names
    edge_pairs = graph.edge_pairs

    missing = all_node_names - node_analyses.keys()
    if missing:
        logger.warning("Nodes missing from analysis, treating as opaque: %s", sorted(missing))
        node_analyses = dict(node_analyses)
        for name in missing:
            node_analyses[name] = NodeAnalysis(name=name, confidence="opaque")

    cycle_node_sets: list[set[str]] = []
    all_cycle_nodes: set[str] = set()
    if topology.cycles:
        for cycle in topology.cycles:
            cycle_node_sets.append(cycle.nodes)
            all_cycle_nodes.update(cycle.nodes)

    writes_no_reducer = {name: analysis.mutations - reducers for name, analysis in node_analyses.items()}

    node_to_cycles: dict[str, set[int]] = {}
    for idx, cs in enumerate(cycle_node_sets):
        for n in cs:
            node_to_cycles.setdefault(n, set()).add(idx)

    dependencies = _build_data_dependencies(graph, all_node_names, node_analyses, node_to_cycles, writes_no_reducer)
    dependencies = _apply_confidence_fallbacks(dependencies, node_analyses, edge_pairs, all_cycle_nodes, all_node_names)
    dependencies = _apply_constraints(dependencies, constraints, edge_pairs, all_node_names)
    dependencies = _apply_cycle_body_ordering(dependencies, topology, edge_pairs, node_analyses, writes_no_reducer)
    cycle_back_edges = {c.back_edge for c in topology.cycles} if topology.cycles else set()
    dependencies = _apply_cycle_boundary_and_conditional(dependencies,
                                                         graph,
                                                         edge_pairs,
                                                         all_cycle_nodes,
                                                         cycle_back_edges)
    dependencies = _apply_branch_dependencies(dependencies, branch_info, all_cycle_nodes)
    dependencies = _apply_post_cycle_ordering(dependencies,
                                              graph,
                                              edge_pairs,
                                              all_node_names,
                                              all_cycle_nodes,
                                              topology)

    if disable_parallelization:
        return _sequential_stages(all_node_names, dependencies, graph.entry_point)
    return _build_parallel_stages(all_node_names, dependencies, graph, branch_info, constraints)


def _split_by_branch(
    ready: set[str],
    node_branch: dict[str, tuple[str, str]],
) -> set[str]:
    """Keep only nodes from compatible branches in a ready set.

    Args:
        ready: Set of nodes ready to execute.
        node_branch: Mapping of node to its (router, branch_label) pair.

    Returns:
        Filtered set of nodes from compatible branches.
    """
    branch_groups: dict[tuple[str, str], set[str]] = {}
    unbranched: set[str] = set()
    for n in ready:
        key = node_branch.get(n)
        if key is None:
            unbranched.add(n)
        else:
            branch_groups.setdefault(key, set()).add(n)

    if len(branch_groups) <= 1:
        return ready

    routers_seen: dict[str, list[tuple[str, str]]] = {}
    for rnode, target in branch_groups:
        routers_seen.setdefault(rnode, []).append((rnode, target))

    keep: set[str] = set(unbranched)
    for rnode, keys in routers_seen.items():
        if len(keys) <= 1:
            for k in keys:
                keep |= branch_groups[k]
        else:
            biggest = max(keys, key=lambda k: len(branch_groups[k]))
            keep |= branch_groups[biggest]

    for key, nodes in branch_groups.items():
        rnode = key[0]
        if len(routers_seen.get(rnode, [])) <= 1:
            keep |= nodes

    return keep


def _sequential_stages(
    node_names: set[str],
    dependencies: dict[str, set[str]],
    entry_point: str,
) -> list[set[str]]:
    """Fallback: fully sequential ordering.

    Args:
        node_names: All node names in the graph.
        dependencies: Per-node dependency sets.
        entry_point: The graph entry-point node.

    Returns:
        List of single-node stages in dependency-respecting order.
    """
    remaining = set(node_names)
    stages: list[set[str]] = []
    while remaining:
        ready = {n for n in remaining if not (dependencies.get(n, set()) & remaining)}
        if not ready:
            ready = {next(iter(remaining))}
        ordered = _stage_order_with_entry_first(ready, entry_point)
        stages.append({ordered[0]})
        remaining.discard(ordered[0])
    return stages


def _stage_order_with_entry_first(nodes: set[str], entry_point: str) -> list[str]:
    lst = sorted(nodes)
    if entry_point in nodes and lst:
        try:
            idx = lst.index(entry_point)
            lst = [lst[idx]] + [n for i, n in enumerate(lst) if i != idx]
        except ValueError:
            pass
    return lst
