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
"""Tests for graph topology: cycle detection, router detection, and topological analysis."""

import pytest

from nat_app.graph.topology import NodeType
from nat_app.graph.topology import analyze_graph_topology
from nat_app.graph.topology import cycle_node_order
from nat_app.graph.topology import detect_cycles
from nat_app.graph.topology import detect_routers
from nat_app.graph.topology import find_router_chains
from nat_app.graph.topology import get_safe_parallelization_groups
from nat_app.graph.types import Graph
from tests.graph.conftest import cycle_graph as _cycle_graph
from tests.graph.conftest import disjoint_cycles_graph as _disjoint_cycles_graph
from tests.graph.conftest import linear_graph as _linear_graph
from tests.graph.conftest import nested_cycle_graph as _nested_cycle_graph
from tests.graph.conftest import overlapping_cycles_graph as _overlapping_cycles_graph
from tests.graph.conftest import router_graph as _router_graph


class TestDetectCycles:

    def test_no_cycles(self):
        cycles = detect_cycles(_linear_graph())
        assert cycles == []

    def test_simple_cycle(self):
        cycles = detect_cycles(_cycle_graph())
        assert len(cycles) == 1
        assert len(cycles[0].nodes) == 3

    def test_cycle_entry_and_exit(self):
        cycles = detect_cycles(_cycle_graph())
        c = cycles[0]
        assert c.entry_node in c.nodes
        assert c.exit_node in c.nodes
        assert c.back_edge[0] == c.exit_node
        assert c.back_edge[1] == c.entry_node

    def test_self_loop_detected(self):
        """Single-node cycle (self-loop A → A) is detected."""
        g = Graph()
        g.add_node("a")
        g.add_edge("a", "a")
        g.entry_point = "a"
        cycles = detect_cycles(g)
        assert len(cycles) == 1
        c = cycles[0]
        assert c.nodes == {"a"}
        assert c.entry_node == "a"
        assert c.exit_node == "a"
        assert c.back_edge == ("a", "a")

    def test_single_node_no_self_loop_not_cycle(self):
        """Single node without self-loop is not reported as a cycle."""
        g = Graph()
        g.add_node("a")
        g.entry_point = "a"
        cycles = detect_cycles(g)
        assert cycles == []

    def test_self_loop_in_scc_returns_only_self_loop_node(self):
        """Self-loop within multi-node SCC: cycle.nodes is {A}, not {A,B,C}."""
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_node("c")
        g.add_edge("a", "a")  # self-loop
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", "a")
        g.entry_point = "a"
        cycles = detect_cycles(g)
        # Expect at least the self-loop cycle; may also have the A->B->C->A cycle
        self_loop_cycles = [c for c in cycles if c.back_edge == ("a", "a")]
        assert len(self_loop_cycles) == 1
        assert self_loop_cycles[0].nodes == {"a"}

    def test_fallback_uses_entry_order_for_exit_entry(self):
        """When fallback runs, returned edge has dst=best_entry (correct exit/entry)."""
        from nat_app.graph.topology import _find_scc_back_edges

        scc = {"a", "b", "c"}
        adj = {"a": ["b"], "c": ["a"]}  # a→b, c→a (DAG; DFS finds no back edge)
        rev_adj = {"b": ["a"], "a": ["c"]}
        entry_order = {"a": 0, "b": 1, "c": 2}

        result = _find_scc_back_edges(scc, adj, rev_adj, entry_order)
        assert result == [("c", "a")]
        # (exit, entry) = (c, a); a has lowest entry_order, so a is correct entry

    def test_fallback_without_entry_order_uses_lexicographic_entry(self):
        """When fallback runs and entry_order is empty, best_entry is min(scc)."""
        from nat_app.graph.topology import _find_scc_back_edges

        scc = {"x", "y", "z"}
        adj = {"x": ["y"], "z": ["x"]}
        rev_adj = {"y": ["x"], "x": ["z"]}

        result = _find_scc_back_edges(scc, adj, rev_adj, entry_order=None)
        assert result == [("z", "x")]
        # best_entry = min(scc) = "x"; edge z→x points to x


class TestCycleNodeOrder:

    def test_order_starts_at_entry(self):
        g = _cycle_graph()
        cycles = detect_cycles(g)
        order = cycle_node_order(cycles[0], g.edge_pairs)
        assert order[0] == cycles[0].entry_node

    def test_all_nodes_present(self):
        g = _cycle_graph()
        cycles = detect_cycles(g)
        order = cycle_node_order(cycles[0], g.edge_pairs)
        assert set(order) == cycles[0].nodes


class TestDetectRouters:

    def test_no_routers(self):
        routers = detect_routers(_linear_graph())
        assert routers == []

    def test_single_router(self):
        routers = detect_routers(_router_graph())
        assert len(routers) == 1
        assert routers[0].node == "router"

    def test_router_branches(self):
        routers = detect_routers(_router_graph())
        r = routers[0]
        assert "go_left" in r.branches
        assert "go_right" in r.branches


class TestAnalyzeGraphTopology:

    def test_linear_all_regular(self):
        topo = analyze_graph_topology(_linear_graph())
        for nt in topo.node_types.values():
            assert nt == NodeType.REGULAR

    def test_cycle_node_types(self):
        topo = analyze_graph_topology(_cycle_graph())
        types = set(topo.node_types.values())
        assert NodeType.CYCLE_ENTRY in types
        assert NodeType.CYCLE_EXIT in types

    def test_router_node_type(self):
        topo = analyze_graph_topology(_router_graph())
        assert topo.node_types["router"] == NodeType.ROUTER


class TestFindRouterChains:

    def test_no_chains_single_router(self):
        topo = analyze_graph_topology(_router_graph())
        chains = find_router_chains(topo)
        assert chains == []

    def test_chain_detected(self):
        g = Graph()
        g.add_node("r1")
        g.add_node("r2")
        g.add_node("a")
        g.add_node("b")
        g.add_edge("r1", "r2")
        g.add_edge("r1", "a")
        g.add_edge("r2", "b")
        g.add_conditional_edges("r1", {"branch_a": ["a"], "branch_r2": ["r2"]})
        g.add_conditional_edges("r2", {"branch_b": ["b"]})
        g.entry_point = "r1"
        topo = analyze_graph_topology(g)
        chains = find_router_chains(topo)
        assert len(chains) == 1
        assert chains[0] == ["r1", "r2"]


class TestDetectCyclesMultiCycle:
    """Tarjan's SCC finds nested, disjoint, and overlapping cycles."""

    def test_nested_cycles_both_found(self):
        cycles = detect_cycles(_nested_cycle_graph())
        assert len(cycles) >= 2, f"Expected >=2 cycles, got {len(cycles)}"
        all_back_edges = {c.back_edge for c in cycles}
        assert len(all_back_edges) >= 2, "Back-edges should be distinct"
        all_nodes = [c.nodes for c in cycles]
        shared = set.intersection(*all_nodes) if len(all_nodes) >= 2 else set()
        assert "evaluate" in shared or any(
            "evaluate" in c.nodes for c in cycles
        ), "evaluate should participate in at least one cycle"

    def test_nested_shared_node_in_both(self):
        cycles = detect_cycles(_nested_cycle_graph())
        cycles_with_evaluate = [c for c in cycles if "evaluate" in c.nodes]
        assert len(cycles_with_evaluate) >= 2, ("evaluate should appear in both the inner and outer cycle")

    def test_disjoint_cycles_both_found(self):
        cycles = detect_cycles(_disjoint_cycles_graph())
        assert len(cycles) >= 2, f"Expected >=2 disjoint cycles, got {len(cycles)}"
        cycle_node_sets = [c.nodes for c in cycles]
        for i, ns_a in enumerate(cycle_node_sets):
            for ns_b in cycle_node_sets[i + 1:]:
                assert not (ns_a & ns_b), "Disjoint cycles should not share nodes"

    def test_overlapping_cycles_both_found(self):
        cycles = detect_cycles(_overlapping_cycles_graph())
        assert len(cycles) >= 2, f"Expected >=2 overlapping cycles, got {len(cycles)}"
        all_back_edges = {c.back_edge for c in cycles}
        assert len(all_back_edges) >= 2

    def test_single_cycle_regression(self):
        cycles = detect_cycles(_cycle_graph())
        assert len(cycles) == 1
        assert len(cycles[0].nodes) == 3

    def test_no_cycles_regression(self):
        cycles = detect_cycles(_linear_graph())
        assert cycles == []


class TestAnalyzeGraphTopologyMultiCycle:
    """Node-type classification with overlapping cycles."""

    def test_nested_cycle_node_types_priority(self):
        topo = analyze_graph_topology(_nested_cycle_graph())
        assert topo.node_types["evaluate"] == NodeType.CYCLE_ENTRY, ("evaluate is CYCLE_ENTRY in the inner cycle; "
                                                                     "most-restrictive-wins should keep CYCLE_ENTRY")

    def test_nested_cycle_all_cycle_nodes_sequential(self):
        topo = analyze_graph_topology(_nested_cycle_graph())
        all_sequential = set()
        for region in topo.sequential_regions:
            all_sequential |= region
        for name in ("parse", "search", "evaluate", "refine", "decide"):
            assert name in all_sequential, f"{name} should be in sequential regions"

    def test_overlapping_shared_nodes_classified(self):
        topo = analyze_graph_topology(_overlapping_cycles_graph())
        for c in topo.cycles:
            for node in c.nodes:
                assert topo.node_types[node] != NodeType.REGULAR, (f"Cycle member {node} should not be REGULAR")


class TestGetSafeParallelizationGroups:

    def test_cycle_fallback_deterministic_singletons(self):
        """When dependency cycle detected, fall back to singleton groups deterministically."""
        topo = analyze_graph_topology(_linear_graph())
        deps = {"a": {"c"}, "b": {"a"}, "c": {"b"}}
        groups = get_safe_parallelization_groups(topo, deps)
        assert groups == [{"a"}, {"b"}, {"c"}]
        assert all(len(g) == 1 for g in groups)

    def test_cycle_fallback_warns(self, caplog):
        """Cycle fallback logs a warning."""
        import logging

        with caplog.at_level(logging.WARNING):
            topo = analyze_graph_topology(_linear_graph())
            deps = {"a": {"c"}, "b": {"a"}, "c": {"b"}}
            get_safe_parallelization_groups(topo, deps)
        assert "Dependency cycle" in caplog.text
        assert "a" in caplog.text and "b" in caplog.text and "c" in caplog.text

    def test_independent_nodes(self):
        topo = analyze_graph_topology(_linear_graph())
        deps = {"a": set(), "b": set(), "c": set()}
        groups = get_safe_parallelization_groups(topo, deps)
        assert any(len(g) > 1 for g in groups)

    def test_dependent_nodes(self):
        topo = analyze_graph_topology(_linear_graph())
        deps = {"a": set(), "b": {"a"}, "c": {"b"}}
        groups = get_safe_parallelization_groups(topo, deps)
        assert all(len(g) == 1 for g in groups)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
