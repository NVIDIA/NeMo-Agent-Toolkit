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
"""Tests for execution order scheduling: edge classification, branch analysis, and optimized order."""

import pytest

from nat_app.constraints.models import ResolvedConstraints
from nat_app.graph.models import EdgeType
from nat_app.graph.scheduling import analyze_cycle_body
from nat_app.graph.scheduling import classify_edges
from nat_app.graph.scheduling import compute_branch_info
from nat_app.graph.scheduling import compute_optimized_order
from nat_app.graph.topology import CycleInfo
from nat_app.graph.topology import analyze_graph_topology
from nat_app.graph.types import Graph


class TestClassifyEdges:

    def test_necessary_edge(self, make_node):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        analyses = {
            "a": make_node("a", writes={"x"}),
            "b": make_node("b", reads={"x"}),
        }
        results = classify_edges(g, analyses)
        assert len(results) == 1
        assert results[0].edge_type == EdgeType.NECESSARY

    def test_unnecessary_edge(self, make_node):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        analyses = {
            "a": make_node("a", writes={"x"}),
            "b": make_node("b", reads={"y"}),
        }
        results = classify_edges(g, analyses)
        assert results[0].edge_type == EdgeType.UNNECESSARY

    def test_unknown_missing_analysis(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        results = classify_edges(g, {})
        assert results[0].edge_type == EdgeType.UNKNOWN

    def test_conditional_edge(self, make_node):
        g = Graph()
        g.add_node("router")
        g.add_node("target")
        g.add_edge("router", "target")
        g.add_conditional_edges("router", {"branch": ["target"]})
        analyses = {
            "router": make_node("router"),
            "target": make_node("target"),
        }
        results = classify_edges(g, analyses)
        assert results[0].edge_type == EdgeType.CONDITIONAL

    def test_incomplete_confidence_kept_necessary(self, make_node):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        analyses = {
            "a": make_node("a", confidence="partial"),
            "b": make_node("b"),
        }
        results = classify_edges(g, analyses)
        assert results[0].edge_type == EdgeType.NECESSARY

    def test_reducer_only_overlap_unnecessary(self, make_node):
        """Edge is unnecessary when overlap is only on reducer fields (parallel-safe)."""
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        analyses = {
            "a": make_node("a", writes={"messages"}),
            "b": make_node("b", reads={"messages"}),
        }
        reducer_fields = {"state": {"messages"}}
        results = classify_edges(g, analyses, reducer_fields=reducer_fields)
        assert len(results) == 1
        assert results[0].edge_type == EdgeType.UNNECESSARY

    def test_reducer_plus_non_reducer_overlap_necessary(self, make_node):
        """Edge stays necessary when overlap includes non-reducer fields."""
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        analyses = {
            "a": make_node("a", writes={"messages", "x"}),
            "b": make_node("b", reads={"messages", "x"}),
        }
        reducer_fields = {"state": {"messages"}}
        results = classify_edges(g, analyses, reducer_fields=reducer_fields)
        assert len(results) == 1
        assert results[0].edge_type == EdgeType.NECESSARY


class TestComputeBranchInfo:

    def test_no_routers(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        g.entry_point = "a"
        topo = analyze_graph_topology(g)
        info = compute_branch_info(g, topo)
        assert info == {}

    def test_single_router(self):
        g = Graph()
        g.add_node("r")
        g.add_node("a")
        g.add_node("b")
        g.add_node("m")
        g.add_edge("r", "a")
        g.add_edge("r", "b")
        g.add_edge("a", "m")
        g.add_edge("b", "m")
        g.add_conditional_edges("r", {"left": ["a"], "right": ["b"]})
        g.entry_point = "r"
        topo = analyze_graph_topology(g)
        info = compute_branch_info(g, topo)
        assert "r" in info
        assert "a" in info["r"].branches.get("left", set())
        assert "b" in info["r"].branches.get("right", set())
        assert "m" in info["r"].merge_nodes


class TestAnalyzeCycleBody:

    def test_small_cycle_no_parallelism(self, make_node):
        cycle = CycleInfo(
            nodes={"a", "b", "c"},
            entry_node="a",
            exit_node="c",
            back_edge=("c", "a"),
        )
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_node("c")
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", "a")
        analyses = {
            "a": make_node("a", writes={"x"}),
            "b": make_node("b", reads={"x"}, writes={"y"}),
            "c": make_node("c", reads={"y"}),
        }
        result = analyze_cycle_body(cycle, g, analyses)
        assert result is not None
        assert not result.has_parallelism

    def test_parallelizable_cycle(self, make_node):
        cycle = CycleInfo(
            nodes={"entry", "a", "b", "exit"},
            entry_node="entry",
            exit_node="exit",
            back_edge=("exit", "entry"),
        )
        g = Graph()
        for n in ["entry", "a", "b", "exit"]:
            g.add_node(n)
        g.add_edge("entry", "a")
        g.add_edge("entry", "b")
        g.add_edge("a", "exit")
        g.add_edge("b", "exit")
        g.add_edge("exit", "entry")
        analyses = {
            "entry": make_node("entry", writes={"init"}),
            "a": make_node("a", reads={"p"}, writes={"a_out"}),
            "b": make_node("b", reads={"q"}, writes={"b_out"}),
            "exit": make_node("exit", reads={"a_out", "b_out"}),
        }
        result = analyze_cycle_body(cycle, g, analyses)
        assert result is not None
        assert result.has_parallelism


class TestComputeOptimizedOrder:

    def test_linear_chain(self, make_node):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_node("c")
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.entry_point = "a"
        analyses = {
            "a": make_node("a", writes={"x"}),
            "b": make_node("b", reads={"x"}, writes={"y"}),
            "c": make_node("c", reads={"y"}),
        }
        topo = analyze_graph_topology(g)
        order = compute_optimized_order(g, analyses, topo)
        flat = [n for stage in order for n in stage]
        assert flat.index("a") < flat.index("b") < flat.index("c")

    def test_diamond_parallelism(self, make_node, diamond_graph):
        g = diamond_graph
        analyses = {
            "a": make_node("a", writes={"start"}),
            "b": make_node("b", reads={"start"}, writes={"b_out"}),
            "c": make_node("c", reads={"start"}, writes={"c_out"}),
            "d": make_node("d", reads={"b_out", "c_out"}),
        }
        topo = analyze_graph_topology(g)
        order = compute_optimized_order(g, analyses, topo)
        assert any({"b", "c"} <= stage for stage in order), "b and c should be in the same parallel stage"

    def test_disable_parallelization(self, make_node, diamond_graph):
        g = diamond_graph
        analyses = {
            "a": make_node("a", writes={"start"}),
            "b": make_node("b", reads={"start"}, writes={"b_out"}),
            "c": make_node("c", reads={"start"}, writes={"c_out"}),
            "d": make_node("d", reads={"b_out", "c_out"}),
        }
        topo = analyze_graph_topology(g)
        order = compute_optimized_order(g, analyses, topo, disable_parallelization=True)
        assert all(len(stage) == 1 for stage in order)

    def test_all_nodes_present(self, make_node, diamond_graph):
        g = diamond_graph
        analyses = {n: make_node(n) for n in ["a", "b", "c", "d"]}
        topo = analyze_graph_topology(g)
        order = compute_optimized_order(g, analyses, topo)
        all_nodes = set()
        for stage in order:
            all_nodes |= stage
        assert all_nodes == {"a", "b", "c", "d"}

    def test_missing_nodes_treated_as_opaque(self, make_node):
        """Missing nodes in node_analyses are treated as opaque and scheduled safely."""
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_node("c")
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.entry_point = "a"
        analyses = {"a": make_node("a", writes={"x"}), "b": make_node("b", reads={"x"})}
        topo = analyze_graph_topology(g)
        order = compute_optimized_order(g, analyses, topo)
        all_nodes = {n for stage in order for n in stage}
        assert all_nodes == {"a", "b", "c"}
        flat = [n for stage in order for n in stage]
        assert flat.index("a") < flat.index("b") < flat.index("c")

    @pytest.mark.parametrize("confidence", ["partial", "opaque"])
    def test_successor_of_uncertain_node_runs_after_it(self, make_node, confidence):
        """p's writes are not fully known, so c (after p in the graph) must not run before it."""
        g = Graph()
        g.add_node("a")
        g.add_node("p")
        g.add_node("c")
        g.add_edge("a", "p")
        g.add_edge("p", "c")
        g.entry_point = "a"
        analyses = {
            "a": make_node("a", writes={"q"}),
            "p": make_node("p", reads={"q"}, confidence=confidence),
            "c": make_node("c", reads={"r"}),
        }
        topo = analyze_graph_topology(g)
        order = compute_optimized_order(g, analyses, topo)
        stage_of = {n: i for i, stage in enumerate(order) for n in stage}
        assert stage_of["a"] < stage_of["p"] < stage_of["c"]
        assert all(edge.edge_type == EdgeType.NECESSARY for edge in classify_edges(g, analyses))

    def test_uncertain_node_edge_wins_over_an_opposite_data_dependency(self, make_node):
        """p reads x, which its successor c overwrites; d reads what p writes. The p -> c edge
        must not leave a p <-> c cycle that lets d run before p."""
        g = Graph()
        for name in ("a", "p", "c", "d"):
            g.add_node(name)
        g.add_edge("a", "p")
        g.add_edge("p", "c")
        g.add_edge("c", "d")
        g.entry_point = "a"
        analyses = {
            "a": make_node("a", writes={"x"}),
            "p": make_node("p", reads={"x"}, writes={"y"}, confidence="partial"),
            "c": make_node("c", writes={"x"}),
            "d": make_node("d", reads={"y"}),
        }
        topo = analyze_graph_topology(g)
        order = compute_optimized_order(g, analyses, topo)
        stage_of = {n: i for i, stage in enumerate(order) for n in stage}
        assert stage_of["a"] < stage_of["p"] < stage_of["c"]
        assert stage_of["p"] < stage_of["d"]

    def test_uncertain_node_keeps_order_with_indirect_descendants(self, make_node):
        """p reads x, which c overwrites two steps later, and c reads what b writes. p must
        not wait for c, or p -> c -> b -> p becomes a cycle."""
        g = Graph()
        for name in ("a", "p", "b", "c"):
            g.add_node(name)
        g.add_edge("a", "p")
        g.add_edge("p", "b")
        g.add_edge("b", "c")
        g.entry_point = "a"
        analyses = {
            "a": make_node("a", writes={"x"}),
            "p": make_node("p", reads={"x"}, confidence="partial"),
            "b": make_node("b", writes={"y"}),
            "c": make_node("c", reads={"y"}, writes={"x"}),
        }
        topo = analyze_graph_topology(g)
        order = compute_optimized_order(g, analyses, topo)
        stage_of = {n: i for i, stage in enumerate(order) for n in stage}
        assert stage_of["a"] < stage_of["p"] < stage_of["b"] < stage_of["c"]

    def test_partial_node_with_a_known_write_runs_before_its_successor(self, make_node):
        """reset writes a and then clears the state, so its other writes are unknown; use reads b,
        which reset may have cleared."""
        g = Graph()
        for name in ("start", "reset", "use"):
            g.add_node(name)
        g.add_edge("start", "reset")
        g.add_edge("reset", "use")
        g.entry_point = "start"
        analyses = {
            "start": make_node("start", writes={"b"}),
            "reset": make_node("reset", writes={"a"}, confidence="partial"),
            "use": make_node("use", reads={"b"}),
        }
        topo = analyze_graph_topology(g)
        order = compute_optimized_order(g, analyses, topo)
        assert order == [{"start"}, {"reset"}, {"use"}]

    def test_explicit_dependency_against_the_graph_order_is_kept(self, make_node, caplog):
        """p is declared to depend on c, which comes after p in the graph: c runs first and the
        conflict is logged."""
        g = Graph()
        for name in ("a", "p", "c"):
            g.add_node(name)
        g.add_edge("a", "p")
        g.add_edge("p", "c")
        g.entry_point = "a"
        analyses = {
            "a": make_node("a", writes={"q"}),
            "p": make_node("p", reads={"q"}, confidence="partial"),
            "c": make_node("c", reads={"r"}),
        }
        constraints = {"p": ResolvedConstraints(name="p", explicit_dependencies={"c"})}
        topo = analyze_graph_topology(g)
        order = compute_optimized_order(g, analyses, topo, resolved_constraints=constraints)
        stage_of = {n: i for i, stage in enumerate(order) for n in stage}
        assert stage_of["a"] < stage_of["p"]
        assert stage_of["c"] < stage_of["p"]
        assert "Dependencies already make 'p' wait for 'c', against the graph order" in caplog.text

    def test_explicit_dependency_against_the_graph_order_through_other_nodes_is_kept(self, make_node):
        """p is declared to depend on c, and c reads what b writes, so b and c both run before p
        even though p comes before them in the graph."""
        g = Graph()
        for name in ("a", "p", "b", "c"):
            g.add_node(name)
        g.add_edge("a", "p")
        g.add_edge("p", "b")
        g.add_edge("b", "c")
        g.entry_point = "a"
        analyses = {
            "a": make_node("a", writes={"q"}),
            "p": make_node("p", reads={"q"}, confidence="partial"),
            "b": make_node("b", writes={"x"}),
            "c": make_node("c", reads={"x"}),
        }
        constraints = {"p": ResolvedConstraints(name="p", explicit_dependencies={"c"})}
        topo = analyze_graph_topology(g)
        order = compute_optimized_order(g, analyses, topo, resolved_constraints=constraints)
        stage_of = {n: i for i, stage in enumerate(order) for n in stage}
        assert stage_of["a"] < stage_of["p"]
        assert stage_of["b"] < stage_of["c"] < stage_of["p"]

    def test_data_dependency_against_the_graph_order_does_not_reorder_a_partial_node(self, make_node):
        """a reads x and c, below p, writes it: the graph order a -> p -> c is kept instead of
        forming the cycle a -> c -> p -> a."""
        g = Graph()
        for name in ("r", "a", "p", "c"):
            g.add_node(name)
        g.add_edge("r", "a")
        g.add_edge("a", "p")
        g.add_edge("p", "c")
        g.entry_point = "r"
        analyses = {
            "r": make_node("r"),
            "a": make_node("a", reads={"x"}),
            "p": make_node("p", confidence="partial"),
            "c": make_node("c", writes={"x"}),
        }
        topo = analyze_graph_topology(g)
        order = compute_optimized_order(g, analyses, topo)
        stage_of = {n: i for i, stage in enumerate(order) for n in stage}
        assert stage_of["r"] < stage_of["p"]
        assert stage_of["a"] < stage_of["p"] < stage_of["c"]

    def test_write_write_conflict_serializes_nodes(self, make_node, diamond_graph):
        """Two nodes writing the same non-reducer key must not be in the same stage."""
        g = diamond_graph
        analyses = {
            "a": make_node("a", writes={"start"}),
            "b": make_node("b", reads={"start"}, writes={"shared_out"}),
            "c": make_node("c", reads={"start"}, writes={"shared_out"}),
            "d": make_node("d", reads={"shared_out"}),
        }
        topo = analyze_graph_topology(g)
        order = compute_optimized_order(g, analyses, topo)
        for stage in order:
            assert not ({"b", "c"} <= stage), "b and c must not run in parallel (write-write conflict)"


class TestComputeOptimizedOrderMultiCycle:
    """Scheduling with nested and disjoint cycles."""

    def test_nested_cycles_all_nodes_present(self, make_node, nested_cycle_graph):
        g = nested_cycle_graph
        analyses = {n: make_node(n) for n in g.node_names}
        topo = analyze_graph_topology(g)
        order = compute_optimized_order(g, analyses, topo)
        all_nodes = set()
        for stage in order:
            all_nodes |= stage
        assert all_nodes == g.node_names

    def test_nested_cycles_ordering(self, make_node, nested_cycle_graph):
        g = nested_cycle_graph
        analyses = {
            "parse": make_node("parse", writes={"query"}),
            "search": make_node("search", reads={"query"}, writes={"results"}),
            "evaluate": make_node("evaluate", reads={"results"}, writes={"score"}),
            "refine": make_node("refine", reads={"score"}, writes={"results"}),
            "decide": make_node("decide", reads={"score"}),
        }
        topo = analyze_graph_topology(g)
        order = compute_optimized_order(g, analyses, topo)
        flat = [n for stage in order for n in stage]
        assert "parse" in flat
        assert "search" in flat
        assert "evaluate" in flat
        assert "decide" in flat
        assert "refine" in flat

    def test_disjoint_cycles_all_nodes_present(self, make_node, disjoint_cycles_graph):
        g = disjoint_cycles_graph
        analyses = {n: make_node(n) for n in g.node_names}
        topo = analyze_graph_topology(g)
        order = compute_optimized_order(g, analyses, topo)
        all_nodes = set()
        for stage in order:
            all_nodes |= stage
        assert all_nodes == g.node_names

    def test_disjoint_cycles_entry_before_cycles(self, make_node, disjoint_cycles_graph):
        g = disjoint_cycles_graph
        analyses = {
            "entry": make_node("entry", writes={"init"}),
            "loop_a": make_node("loop_a", reads={"init"}, writes={"a_out"}),
            "check_a": make_node("check_a", reads={"a_out"}),
            "bridge": make_node("bridge", reads={"a_out"}, writes={"b_init"}),
            "loop_b": make_node("loop_b", reads={"b_init"}, writes={"b_out"}),
            "check_b": make_node("check_b", reads={"b_out"}),
        }
        topo = analyze_graph_topology(g)
        order = compute_optimized_order(g, analyses, topo)
        flat = [n for stage in order for n in stage]
        assert flat.index("entry") < flat.index("loop_a")

    def test_overlapping_cycles_all_nodes_present(self, make_node, overlapping_cycles_graph):
        g = overlapping_cycles_graph
        analyses = {n: make_node(n) for n in g.node_names}
        topo = analyze_graph_topology(g)
        order = compute_optimized_order(g, analyses, topo)
        all_nodes = set()
        for stage in order:
            all_nodes |= stage
        assert all_nodes == g.node_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
