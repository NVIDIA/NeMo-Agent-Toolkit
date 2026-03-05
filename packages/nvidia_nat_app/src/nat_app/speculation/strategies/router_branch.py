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
Router-branch speculation strategy.

Launches all router target branches speculatively, then cancels
unchosen branches once the router decides.  This is the original
(and currently only) speculation strategy in nat_app.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from nat_app.graph.scheduling import compute_branch_info
from nat_app.graph.topology import GraphTopology
from nat_app.graph.topology import analyze_graph_topology
from nat_app.graph.topology import find_router_chains
from nat_app.graph.types import Graph
from nat_app.speculation.plan import SpeculationPlan
from nat_app.speculation.resolution import Resolution
from nat_app.speculation.resolution import ResolutionPolicy
from nat_app.speculation.safety import SpeculationSafetyConfig
from nat_app.speculation.safety import is_marked_speculation_unsafe
from nat_app.speculation.strategies.base import SpeculationOpportunity


@dataclass(frozen=True)
class RouterBranchResolution:
    """Resolution policy for full-branch router speculation.

    Wraps the cancel_map / label_map logic: given a decision label,
    determines which speculatively-launched nodes to keep vs. cancel.
    """

    cancel_map: dict[str, frozenset[str]]
    """``{decision_label: nodes_to_cancel}`` for each possible decision."""

    label_map: dict[str, frozenset[str]] | None
    """Maps decision labels to target node sets (1-to-many), or ``None``."""

    all_targets: frozenset[str]
    """All targets that were launched speculatively."""

    def _resolve_label(self, chosen: str) -> str:
        """Map a resolved target name back to its decision label."""
        if chosen in self.cancel_map:
            return chosen
        if self.label_map:
            for label, targets in self.label_map.items():
                if chosen in targets:
                    return label
        return chosen

    def resolve(self, decision_result: Any) -> Resolution:
        """Resolve speculation given the router's decision output.

        Args:
            decision_result: The router node's output (chosen branch label
                or target name). Converted to string for lookup.

        Returns:
            Resolution with keep/cancel/rerun node sets.
        """
        label = self._resolve_label(str(decision_result))
        cancel = self.cancel_map.get(label, frozenset())
        keep = self.all_targets - cancel
        return Resolution(keep=keep, cancel=cancel, rerun=frozenset())

    def is_on_chosen_path(self, node: str, decision_result: Any) -> bool:
        """Check whether a node is on the chosen path (not cancelled).

        Args:
            node: The node name to check.
            decision_result: The router's decision output.

        Returns:
            True if the node is on the chosen path, False if it should
            be cancelled or is not a speculative target.
        """
        if node not in self.all_targets:
            return False
        return node not in self.get_cancel_set(decision_result)

    def get_cancel_set(self, decision_result: Any) -> frozenset[str]:
        """Return the set of nodes to cancel for the given decision.

        Args:
            decision_result: The router's decision output.

        Returns:
            Frozenset of node names to cancel (unchosen branches).
        """
        label = self._resolve_label(str(decision_result))
        return self.cancel_map.get(label, frozenset())


# -- Satisfy the ResolutionPolicy protocol check at import time --
assert isinstance(RouterBranchResolution(
    cancel_map={},
    label_map=None,
    all_targets=frozenset(),
), ResolutionPolicy)


class RouterBranchStrategy:
    """Full-branch router speculation: launch all, cancel unchosen.

    Identifies routers with multiple branches and builds plans that
    speculatively launch all branch targets.  After the router decides,
    unchosen branches are cancelled via ``RouterBranchResolution``.
    """

    @property
    def name(self) -> str:
        """Unique strategy identifier (e.g. ``"router_branch"``)."""
        return "router_branch"

    @property
    def priority(self) -> int:
        """Strategy priority for conflict resolution (higher = first claim)."""
        return 100

    def identify(
        self,
        graph: Graph,
        topology: GraphTopology,
    ) -> list[SpeculationOpportunity]:
        """Identify router-branch speculation opportunities in the graph.

        Args:
            graph: The compiled graph.
            topology: Precomputed topology with routers and cycles.

        Returns:
            List of speculation opportunities (one per router with branches).
        """
        if not topology.routers:
            return []

        branch_info = compute_branch_info(graph, topology)
        opportunities: list[SpeculationOpportunity] = []

        for rnode, binfo in branch_info.items():
            all_branch_nodes: set[str] = set()
            for exclusive in binfo.branches.values():
                all_branch_nodes |= exclusive

            if not all_branch_nodes:
                continue

            opportunities.append(
                SpeculationOpportunity(
                    strategy=self.name,
                    decision_node=rnode,
                    candidate_targets=all_branch_nodes,
                    priority=len(all_branch_nodes),
                    metadata={
                        "branch_info": binfo,
                        "branches": binfo.branches,
                        "merge_nodes": binfo.merge_nodes,
                    },
                ))

        return opportunities

    def plan(
        self,
        opportunities: list[SpeculationOpportunity],
        safety: SpeculationSafetyConfig,
        graph: Graph,
    ) -> list[SpeculationPlan]:
        """Build speculation plans from opportunities, applying safety exclusions.

        Args:
            opportunities: Opportunities from ``identify``.
            safety: Safety config for excluding unsafe nodes.
            graph: The compiled graph.

        Returns:
            List of speculation plans ready for execution.
        """
        topology = analyze_graph_topology(graph)
        router_lookup = {r.node: r for r in topology.routers}

        chain_successor: dict[str, str] = {}
        for chain in find_router_chains(topology):
            for i in range(len(chain) - 1):
                chain_successor[chain[i]] = chain[i + 1]

        plans: list[SpeculationPlan] = []

        for opp in opportunities:
            rnode = opp.decision_node
            router = router_lookup.get(rnode)
            branches: dict[str, set[str]] = opp.metadata["branches"]
            merge_nodes: set[str] = opp.metadata["merge_nodes"]

            excluded: set[str] = set()
            for node_name in opp.candidate_targets:
                node_func = graph.get_node(node_name).func if graph.has_node(node_name) else None
                if _is_excluded(node_name, node_func, safety):
                    excluded.add(node_name)

            targets_to_launch = frozenset(opp.candidate_targets - excluded)
            if not targets_to_launch:
                continue

            cancel_map: dict[str, frozenset[str]] = {}
            for label, branch_nodes in branches.items():
                chosen_branch = branch_nodes - excluded
                unchosen = targets_to_launch - chosen_branch
                if unchosen:
                    cancel_map[label] = frozenset(unchosen)

            cond_targets = graph.get_conditional_targets(rnode)
            label_map: dict[str, frozenset[str]] | None = None
            if cond_targets is not None:
                label_map = {label: frozenset(targets) for label, targets in cond_targets.items()}

            max_depth = max(
                (len(excl) for excl in branches.values()),
                default=0,
            )

            resolution = RouterBranchResolution(
                cancel_map=cancel_map,
                label_map=label_map,
                all_targets=targets_to_launch,
            )

            plans.append(
                SpeculationPlan(
                    strategy=self.name,
                    decision_node=rnode,
                    targets_to_launch=targets_to_launch,
                    excluded_nodes=frozenset(excluded),
                    resolution=resolution,
                    merge_nodes=frozenset(merge_nodes),
                    max_branch_depth=max_depth,
                    is_cycle_exit=router.is_cycle_exit if router else False,
                    chain_next=chain_successor.get(rnode),
                ))

        return plans


def _is_excluded(
    node_name: str,
    node_func: Callable | None,
    safety: SpeculationSafetyConfig,
) -> bool:
    if node_name in safety.safe_overrides:
        return False
    if node_name in safety.unsafe_nodes:
        return True
    if node_func is not None and is_marked_speculation_unsafe(node_func):
        return True
    return False
