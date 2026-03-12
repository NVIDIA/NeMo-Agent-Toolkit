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
Speculation planning: concrete plans from graph analysis + safety config.

Bridges the analysis layer (topology, branch info) and the execution layer
by producing ``SpeculationPlan`` objects that tell executors exactly
what to launch, what to exclude, and how to resolve decisions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from nat_app.graph.factory import build_graph_and_adapter
from nat_app.speculation.resolution import ResolutionPolicy
from nat_app.speculation.safety import SpeculationSafetyConfig


@dataclass(frozen=True)
class SpeculationPlan:
    """Concrete speculation plan for a single decision point.

    Produced by ``plan_speculation`` (or ``SpeculationPlanner``),
    consumed by framework executors.  The ``resolution`` policy
    encapsulates strategy-specific logic for determining what to
    keep, cancel, or re-run after the decision node completes.
    """

    strategy: str
    """Strategy that produced this plan (e.g. ``"router_branch"``)."""

    decision_node: str
    """Node whose completion resolves the speculation."""

    targets_to_launch: frozenset[str]
    """Nodes safe to launch speculatively (targets minus excluded)."""

    excluded_nodes: frozenset[str]
    """Nodes excluded from speculation (unsafe or not overridden)."""

    resolution: ResolutionPolicy
    """Strategy-specific policy for resolving speculation outcomes."""

    merge_nodes: frozenset[str]
    """Nodes shared across all branches (never cancelled)."""

    max_branch_depth: int
    """Longest exclusive-branch path length."""

    is_cycle_exit: bool
    """Whether this decision node also controls a cycle back-edge."""

    chain_next: str | None = None
    """Next decision node in a contiguous chain, or ``None`` if terminal.

    When set, some ``targets_to_launch`` may be "deferred" -- reachable
    only through ``chain_next`` and not safe to launch until that node
    also decides.
    """


def plan_speculation(
    nodes: dict[str, Callable | None],
    edges: list[tuple[str, str]],
    conditional_edges: dict[str, dict[str, str | list[str]]] | None = None,
    safety: SpeculationSafetyConfig | None = None,
    self_state_attrs: dict[str, str] | None = None,
) -> list[SpeculationPlan]:
    """Produce concrete speculation plans from graph data and safety config.

    Delegates to ``SpeculationPlanner`` with the default
    ``RouterBranchStrategy``.

    Args:
        nodes: Mapping of node name to callable (or ``None``).
        edges: List of ``(source, target)`` dependency edges.
        conditional_edges: Router/conditional edges.  Maps a router node
            to ``{label: target_node(s)}``.  Each value may be a single
            node name (``str``) or a list for 1-to-many routing.
        safety: Optional safety configuration for excluding unsafe nodes.
        self_state_attrs: For class methods, maps ``self.X`` -> namespace.

    Returns:
        A list of ``SpeculationPlan`` objects, one per decision point
        that has speculative execution opportunities.
    """
    # pylint: disable=import-outside-toplevel
    from nat_app.speculation.planner import SpeculationPlanner
    from nat_app.speculation.strategies.router_branch import RouterBranchStrategy

    safety = safety or SpeculationSafetyConfig()

    graph, _adapter = build_graph_and_adapter(
        nodes, edges,
        conditional_edges=conditional_edges,
        self_state_attrs=self_state_attrs,
    )

    planner = SpeculationPlanner([RouterBranchStrategy()])
    return planner.plan(graph, safety)


def partition_targets(plan: SpeculationPlan, ) -> tuple[frozenset[str], frozenset[str]]:
    """Split targets into (immediate, deferred) based on ``chain_next``.

    *Immediate* targets are reachable without going through the next
    decision node in a chain.  *Deferred* targets are only reachable
    through ``chain_next`` and should not be launched until that node
    decides.

    When ``chain_next`` is ``None``, all targets are immediate.

    Args:
        plan: Speculation plan to partition.

    Returns:
        Tuple of (immediate targets, deferred targets).
    """
    if plan.chain_next is None:
        return plan.targets_to_launch, frozenset()

    cancel_map = getattr(plan.resolution, "cancel_map", None)
    if cancel_map is None:
        return plan.targets_to_launch, frozenset()

    deferred: set[str] = set()
    for _label, cancel_set in cancel_map.items():
        nodes_on_branch = plan.targets_to_launch - cancel_set
        if plan.chain_next in nodes_on_branch:
            deferred.update(nodes_on_branch - {plan.chain_next})

    if not deferred:
        return plan.targets_to_launch, frozenset()

    immediate = plan.targets_to_launch - frozenset(deferred) - frozenset({plan.chain_next})
    return frozenset(immediate | {plan.chain_next}), frozenset(deferred)
