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
Hierarchical priority assignment stage.

Assigns ``NodeInfo.priority`` using a topology-aware algorithm that groups
nodes by branch structure (conditional / parallel / linear), propagates
worst-case subtree costs, and applies a configurable N-tier discrete
priority system with **hierarchical ceiling propagation**.

Priority is **relative within each branch group**, not global.  Two nodes
behind different conditional routers are independent populations.  Nested
groups inherit a *ceiling* from their parent context so that child
priorities never exceed the parent branch's assigned tier.

The tier system is configurable via the strategy.  The default
``SJFPriorityStrategy`` uses three tiers (HIGH / MEDIUM / LOW) with two
thresholds.  To customize, pass ``SJFPriorityStrategy(tiers=..., thresholds=...)``
to the stage (``len(thresholds) == len(tiers) - 1``).

Cost sources (resolved in order of precedence):

1. **Custom callable** -- ``cost_fn(ProfiledNodeCost) -> float`` supplied at
   construction time.  Requires ``profiled_node_costs`` in context.
2. **Profiled data** -- ``context.metadata["profiled_node_costs"]`` with a
   ``CostMetric`` preset (or default ``SUBTREE_TIME``).
3. **Static LLM analysis** -- ``context.metadata["llm_analysis"]`` populated
   by ``LLMAnalysisStage``.
4. **No-op** -- when none of the above are available.

Algorithm stages (unchanged regardless of cost source):

1. **Branch grouping** -- classify each node as belonging to a conditional
   router group, a parallel fan-out group, or the linear remainder.
2. **Subtree cost propagation** -- for each branch target, compute the
   worst-case total cost through its entire downstream subtree.
   Skipped for *pre-propagated* metrics (e.g. ``SUBTREE_TIME``).
3. **Hierarchical priority assignment** -- groups are processed in
   topological order (parents before children).  For each group:

   - **Ceiling resolution** -- walk backwards from the group's source node
     through single-predecessor chains to find the nearest ancestor with
     an assigned priority.  That priority becomes the *ceiling*.
   - **N-tier assignment** -- the cost ratio between the heaviest and
     lightest branch determines how many tiers to activate (1 through N,
     gated by the threshold list).  Costs are then mapped to ranks within
     that active range, and each rank is capped relative to the ceiling
     using index arithmetic:

       ``capped_index = min(ceiling_index + rank, len(tiers) - 1)``

   - **Parallel strategy** -- parallel fan-out groups inherit the ceiling
     uniformly (all siblings receive the same tier).

   Top-level groups (no parent context) use the absolute tiers with no
   ceiling.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any
from typing import Protocol
from typing import runtime_checkable

from nat_app.compiler.compilation_context import CompilationContext
from nat_app.graph.llm_detection import LLMCallInfo
from nat_app.graph.types import BranchGroup
from nat_app.graph.types import BranchGroupType
from nat_app.graph.types import CostMetric
from nat_app.graph.types import EdgeKind
from nat_app.graph.types import Graph
from nat_app.graph.types import PriorityLevel
from nat_app.graph.types import ProfiledNodeCost

logger = logging.getLogger(__name__)

__all__ = [
    "BranchGroup",
    "BranchGroupType",
    "PriorityAssignmentStage",
    "PriorityStrategy",
    "SJFPriorityStrategy",
]

# ---------------------------------------------------------------------------
# Cost metric info: (accessor, pre_propagated)
# ---------------------------------------------------------------------------

_COST_METRIC_INFO: dict[CostMetric, tuple[Callable[[ProfiledNodeCost], float], bool]] = {
    CostMetric.LLM_CALLS: (lambda c: float(c.llm_call_count), False),
    CostMetric.WALL_CLOCK_MS: (lambda c: c.total_latency_ms, True),
    CostMetric.PROMPT_TOKENS: (lambda c: float(c.total_prompt_tokens), False),
    CostMetric.COMPLETION_TOKENS: (lambda c: float(c.total_completion_tokens), False),
    CostMetric.TOTAL_TOKENS: (lambda c: float(c.total_tokens), False),
    CostMetric.SUBTREE_TIME: (lambda c: c.subtree_time_ms, True),
}


@runtime_checkable
class PriorityStrategy(Protocol):
    """Protocol for pluggable group-level priority assignment.

    Implementations receive a ``BranchGroup`` and optional ceiling, and return
    the priority tier for each node in the group.
    """

    def assign_group_priorities(self, group: BranchGroup, ceiling: PriorityLevel | None) -> list[PriorityLevel]:
        """Assign priority tiers for each node in the group.

        Args:
            group: The branch group with node names and subtree costs.
            ceiling: Optional ceiling tier from a parent group (for nested
                conditionals). None for top-level groups.

        Returns:
            List of priority tiers, one per node in ``group.node_names``.
        """
        ...


class SJFPriorityStrategy:
    """Shortest-job-first priority strategy: cheapest branch gets highest tier.

    Uses an N-tier system with configurable thresholds. The cost ratio between
    heaviest and lightest branch determines how many tiers to activate.
    Parallel groups inherit the ceiling uniformly.

    Default: three tiers (HIGH / MEDIUM / LOW) with thresholds [1.5, 3.0].
    Pass ``tiers`` and ``thresholds`` to customize (``len(thresholds) == len(tiers) - 1``).
    """

    def __init__(
        self,
        tiers: list[PriorityLevel] | None = None,
        thresholds: list[float] | None = None,
    ) -> None:
        self._tiers = tiers or sorted(PriorityLevel, key=lambda t: t.value, reverse=True)
        self._thresholds = (thresholds if thresholds is not None else [1.5, 3.0])
        if len(self._thresholds) != len(self._tiers) - 1:
            raise ValueError(f"Expected {len(self._tiers) - 1} thresholds for "
                             f"{len(self._tiers)} tiers, got {len(self._thresholds)}")
        self._tier_index: dict[PriorityLevel, int] = {tier: idx for idx, tier in enumerate(self._tiers)}
        self._mid_rank = len(self._tiers) // 2

    def assign_group_priorities(self, group: BranchGroup, ceiling: PriorityLevel | None) -> list[PriorityLevel]:
        """Assign SJF-based priority tiers for each node in the group.

        Parallel groups receive uniform ceiling. Conditional/linear groups
        use N-tier assignment with cost ratio gating.

        Args:
            group: The branch group with node names and subtree costs.
            ceiling: Optional ceiling tier from a parent group. None for
                top-level groups.

        Returns:
            List of priority tiers, one per node in ``group.node_names``.
        """
        if group.group_type == BranchGroupType.PARALLEL:
            effective = ceiling if ceiling is not None else self._tiers[self._mid_rank]
            return [effective] * len(group.node_names)
        return self._assign_with_ceiling(group.subtree_costs, ceiling)

    def _assign_with_ceiling(
        self,
        subtree_costs: list[float],
        ceiling: PriorityLevel | None,
    ) -> list[PriorityLevel]:
        absolute = self._auto_assign_priority(subtree_costs)
        if ceiling is None:
            return absolute
        return [self._cap_priority(p, ceiling) for p in absolute]

    def _cap_priority(self, priority: PriorityLevel, ceiling: PriorityLevel) -> PriorityLevel:
        rank = self._tier_index[priority]
        ceiling_idx = self._tier_index[ceiling]
        return self._tiers[min(ceiling_idx + rank, len(self._tiers) - 1)]

    def _active_tier_count(self, ratio: float) -> int:
        active = 1
        for t in self._thresholds:
            if ratio >= t:
                active += 1
            else:
                break
        return min(active, len(self._tiers))

    @staticmethod
    def _cost_to_rank(cost: float, mn: float, mx: float, active_tiers: int) -> int:
        if cost == mn:
            return 0
        if cost == mx and active_tiers >= 3:
            return active_tiers - 1
        if active_tiers == 2:
            return 1
        cost_range = mx - mn
        if cost_range == 0:
            return active_tiers // 2
        normalized = (cost - mn) / cost_range
        middle_count = active_tiers - 2
        rank = 1 + int(normalized * middle_count)
        return max(1, min(rank, active_tiers - 2))

    def _auto_assign_priority(self, subtree_costs: list[float]) -> list[PriorityLevel]:
        if not subtree_costs:
            return []
        mn, mx = min(subtree_costs), max(subtree_costs)
        if mn == 0:
            mn = 1
        ratio = mx / mn
        active = self._active_tier_count(ratio)
        if active == 1:
            return [self._tiers[self._mid_rank]] * len(subtree_costs)
        return [self._tiers[self._cost_to_rank(c, min(subtree_costs), mx, active)] for c in subtree_costs]


class PriorityAssignmentStage:
    """Hierarchical priority assignment from cost analysis and graph topology.

    Reads: ``graph``, optionally ``profiled_node_costs`` and/or ``llm_analysis``
    Writes: ``NodeInfo.priority`` on each node in a group with nonzero cost.

    Nodes with lower cost (the "fast path") receive
    ``HIGH`` so the inference cluster
    schedules them first.  Nodes without any cost in their group are
    left at ``priority=None``.

    The algorithm is topology-aware: it groups nodes by conditional routers
    and parallel fan-out, propagates worst-case subtree costs through nested
    routers, and assigns discrete priority tiers relative to each group.

    Tier configuration (e.g. custom tiers or thresholds) is done via the
    strategy.  Pass ``SJFPriorityStrategy(tiers=..., thresholds=...)`` when
    using the default SJF strategy with custom tier settings.
    """

    def __init__(
        self,
        cost_fn: Callable[[ProfiledNodeCost], float] | None = None,
        cost_metric: CostMetric | None = None,
        pre_propagated: bool = False,
        strategy: PriorityStrategy | None = None,
    ) -> None:
        self._cost_fn = cost_fn
        self._cost_metric = cost_metric
        self._pre_propagated = pre_propagated
        self._strategy: PriorityStrategy = strategy or SJFPriorityStrategy()

    @property
    def name(self) -> str:
        return "priority_assignment"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def apply(self, context: CompilationContext, **kwargs: Any) -> CompilationContext:
        """Assign NodeInfo.priority from cost analysis and graph topology.

        Groups nodes by branch structure, propagates subtree costs, and
        delegates tier assignment to the strategy. No-op if no cost source.

        Args:
            context: Current compilation context with ``graph``, and
                optionally ``profiled_node_costs`` and/or ``llm_analysis``.
            **kwargs: Additional arguments (reserved for future use).

        Returns:
            The updated context with ``NodeInfo.priority`` set on nodes
            in groups with nonzero cost.
        """
        graph: Graph = context.metadata["graph"]

        node_cost_fn, pre_propagated = self._resolve_cost_source(context)
        if node_cost_fn is None:
            return context

        conditional_pairs: frozenset[tuple[str, str]] = frozenset(
            (e.source, e.target) for e in graph.edges if e.kind == EdgeKind.CONDITIONAL)

        groups = self._extract_branch_groups(
            graph,
            node_cost_fn,
            pre_propagated,
            conditional_pairs,
        )
        group_order = self._build_group_order(groups, graph)

        node_assigned_priority: dict[str, PriorityLevel] = {}
        assigned = 0

        for group_name in group_order:
            group = groups[group_name]

            source = self._extract_group_source(group_name)
            ceiling: PriorityLevel | None = None
            if source is not None:
                ceiling = self._resolve_group_ceiling(
                    source,
                    graph,
                    node_assigned_priority,
                )
            group.ceiling = ceiling
            group.priorities = self._strategy.assign_group_priorities(group, ceiling)

            for node_name, priority_level in zip(group.node_names, group.priorities):
                if not graph.has_node(node_name):
                    continue
                node_info = graph.get_node(node_name)
                if node_info.priority is None:
                    node_info.priority = priority_level.value
                    assigned += 1
                node_assigned_priority[node_name] = priority_level

        total_grouped = sum(len(g.node_names) for g in groups.values())
        logger.info(
            "Priority assignment: %d/%d nodes assigned across %d groups",
            assigned,
            total_grouped,
            len(groups),
        )

        return context

    # ------------------------------------------------------------------
    # Cost source resolution
    # ------------------------------------------------------------------

    def _resolve_cost_source(
        self,
        context: CompilationContext,
    ) -> tuple[Callable[[str], float] | None, bool]:
        """Determine the node cost function and propagation mode.

        Resolution chain (first match wins):
        1. Custom ``cost_fn`` on profiled data
        2. ``CostMetric`` preset on profiled data
        3. Static LLM call counts from ``llm_analysis``
        4. ``None`` (no-op)

        Args:
            context: Compilation context with metadata for cost sources.

        Returns:
            ``(node_cost_fn, pre_propagated)`` or ``(None, False)``.
        """
        profiled: dict[str, ProfiledNodeCost] = context.metadata.get(
            "profiled_node_costs",
            {},
        )
        llm_analysis: dict[str, LLMCallInfo] = context.metadata.get(
            "llm_analysis",
            {},
        )

        if self._cost_fn is not None and profiled:
            fn = self._cost_fn
            return (lambda name, _fn=fn, _p=profiled: _fn(_p[name]) if name in _p else 0.0), self._pre_propagated

        if profiled:
            metric = self._cost_metric or CostMetric.SUBTREE_TIME
            accessor, propagated = _COST_METRIC_INFO[metric]
            return (lambda name, _a=accessor, _p=profiled: _a(_p[name]) if name in _p else 0.0), propagated

        if llm_analysis and any(info.call_count > 0 for info in llm_analysis.values()):
            return (lambda name, _la=llm_analysis: float(_la[name].call_count) if name in _la else 0.0), False

        return None, False

    # ------------------------------------------------------------------
    # Branch group extraction
    # ------------------------------------------------------------------

    def _extract_branch_groups(
        self,
        graph: Graph,
        node_cost_fn: Callable[[str], float],
        pre_propagated: bool,
        conditional_pairs: frozenset[tuple[str, str]],
    ) -> dict[str, BranchGroup]:
        """Classify nodes into conditional, parallel, and linear groups.

        Args:
            graph: The compiled graph to analyze.
            node_cost_fn: Maps a node name to its cost value.
            pre_propagated: If ``True``, costs already include subtree propagation.
            conditional_pairs: Set of ``(source, target)`` conditional edge pairs.

        Returns:
            Mapping of group name to its ``BranchGroup``.
        """
        groups: dict[str, BranchGroup] = {}
        assigned_nodes: set[str] = set()

        # Step A: Conditional router groups
        for router_source, branch_map in graph.conditional_edge_sources.items():
            all_targets: set[str] = set()
            for target_list in branch_map.values():
                all_targets.update(target_list)
            targets = sorted(t for t in all_targets if graph.has_node(t))
            if len(targets) < 2:
                continue

            group_name = f"router:{router_source}"
            if pre_propagated:
                subtree_costs = [node_cost_fn(n) for n in targets]
            else:
                subtree_costs = [
                    self._compute_subtree_cost(
                        n,
                        graph,
                        node_cost_fn,
                        frozenset(),
                        conditional_pairs,
                    ) for n in targets
                ]
            groups[group_name] = BranchGroup(
                name=group_name,
                group_type=BranchGroupType.CONDITIONAL,
                node_names=targets,
                subtree_costs=subtree_costs,
            )
            assigned_nodes.update(targets)

        # Step B: Parallel fan-out groups (unconditional edges with 2+ targets)
        conditional_target_set: set[str] = set()
        for branch_map in graph.conditional_edge_sources.values():
            for target_list in branch_map.values():
                conditional_target_set.update(target_list)

        for node_name in graph.node_names:
            succs = graph.successors(node_name)
            unconditional_targets = sorted(
                t for t in succs
                if graph.has_node(t) and t not in assigned_nodes and (node_name, t) not in conditional_pairs)
            if len(unconditional_targets) < 2:
                continue

            group_name = f"parallel:{node_name}"
            if pre_propagated:
                subtree_costs = [node_cost_fn(n) for n in unconditional_targets]
            else:
                subtree_costs = [
                    self._compute_subtree_cost(
                        n,
                        graph,
                        node_cost_fn,
                        frozenset(),
                        conditional_pairs,
                    ) for n in unconditional_targets
                ]
            groups[group_name] = BranchGroup(
                name=group_name,
                group_type=BranchGroupType.PARALLEL,
                node_names=unconditional_targets,
                subtree_costs=subtree_costs,
            )
            assigned_nodes.update(unconditional_targets)

        # Step C: Remaining nodes go into the "linear" group
        linear_nodes = sorted(n for n in graph.node_names
                              if n not in assigned_nodes and graph.has_node(n) and node_cost_fn(n) > 0)
        if linear_nodes:
            costs = [node_cost_fn(n) for n in linear_nodes]
            groups["linear"] = BranchGroup(
                name="linear",
                group_type=BranchGroupType.LINEAR,
                node_names=linear_nodes,
                subtree_costs=costs,
            )

        return groups

    # ------------------------------------------------------------------
    # Subtree cost propagation
    # ------------------------------------------------------------------

    def _compute_subtree_cost(
        self,
        node: str,
        graph: Graph,
        node_cost_fn: Callable[[str], float],
        visited: frozenset[str],
        conditional_pairs: frozenset[tuple[str, str]],
    ) -> float:
        """Worst-case total cost from *node* through all reachable downstream nodes.

        Unconditional successors all execute, so their costs are summed.
        Conditional successors are mutually exclusive, so we take the max.

        Args:
            node: Starting node name.
            graph: The compiled graph.
            node_cost_fn: Maps a node name to its cost value.
            visited: Already-visited nodes to avoid cycles.
            conditional_pairs: Set of ``(source, target)`` conditional edge pairs.

        Returns:
            Worst-case total cost through the subtree rooted at *node*.
        """
        if node in visited:
            return 0.0
        if not graph.has_node(node):
            return 0.0

        own_cost = node_cost_fn(node)
        succs = graph.successors(node)
        if not succs:
            return own_cost

        new_visited = visited | {node}

        unconditional_cost = 0.0
        conditional_costs: list[float] = []

        for succ in succs:
            child_cost = self._compute_subtree_cost(
                succ,
                graph,
                node_cost_fn,
                new_visited,
                conditional_pairs,
            )
            if (node, succ) in conditional_pairs:
                conditional_costs.append(child_cost)
            else:
                unconditional_cost += child_cost

        conditional_cost = max(conditional_costs) if conditional_costs else 0.0
        return own_cost + unconditional_cost + conditional_cost

    # ------------------------------------------------------------------
    # Hierarchical ceiling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_group_source(group_name: str) -> str | None:
        """Parse the source node from a group name.

        ``"router:R"`` -> ``"R"``, ``"parallel:X"`` -> ``"X"``,
        ``"linear"`` -> ``None``.

        Args:
            group_name: Group name in ``"type:source"`` format.

        Returns:
            Source node name, or ``None`` for unnamed groups.
        """
        if ":" in group_name:
            return group_name.split(":", 1)[1]
        return None

    @staticmethod
    def _resolve_group_ceiling(
        source_node: str,
        graph: Graph,
        node_assigned_priority: dict[str, PriorityLevel],
    ) -> PriorityLevel | None:
        """Walk backwards from *source_node* to find the nearest assigned priority.

        Follows single-predecessor chains only.  Stops at merge points
        (multiple predecessors) or graph roots and returns ``None``.

        Args:
            source_node: Node to start walking backwards from.
            graph: The compiled graph.
            node_assigned_priority: Already-assigned priority mapping.

        Returns:
            Nearest ancestor priority, or ``None`` if none found.
        """
        current = source_node
        visited: set[str] = set()
        while current:
            if current in visited:
                return None
            visited.add(current)
            if current in node_assigned_priority:
                return node_assigned_priority[current]
            preds = graph.predecessors(current)
            if len(preds) == 1:
                current = preds[0]
            else:
                return None
        return None

    # ------------------------------------------------------------------
    # Group processing order
    # ------------------------------------------------------------------

    def _build_group_order(
        self,
        groups: dict[str, BranchGroup],
        graph: Graph,
    ) -> list[str]:
        """Return group names sorted so that parent groups are processed first.

        For each group, determines whether its source node is a target in
        (or reachable via single-predecessor chains from) another group.
        Groups are then sorted by ascending depth in the group tree.

        Args:
            groups: Mapping of group name to ``BranchGroup``.
            graph: The compiled graph for predecessor lookups.

        Returns:
            Group names in parent-first topological order.
        """
        node_to_group: dict[str, str] = {}
        for gname, group in groups.items():
            for node in group.node_names:
                node_to_group[node] = gname

        def _find_parent(gname: str) -> str | None:
            source = self._extract_group_source(gname)
            if source is None:
                return None
            current = source
            visited: set[str] = set()
            while current:
                if current in visited:
                    return None
                visited.add(current)
                if current in node_to_group:
                    return node_to_group[current]
                preds = graph.predecessors(current)
                if len(preds) == 1:
                    current = preds[0]
                else:
                    return None
            return None

        depth_cache: dict[str, int] = {}

        def _depth(gname: str) -> int:
            if gname in depth_cache:
                return depth_cache[gname]
            parent = _find_parent(gname)
            if parent is None or parent not in groups:
                depth_cache[gname] = 0
            else:
                depth_cache[gname] = 1 + _depth(parent)
            return depth_cache[gname]

        for gname in groups:
            _depth(gname)

        return sorted(groups.keys(), key=lambda g: depth_cache[g])
