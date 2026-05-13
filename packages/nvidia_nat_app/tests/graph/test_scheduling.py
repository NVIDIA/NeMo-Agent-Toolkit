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

from nat_app.graph.models import EdgeType
from nat_app.graph.scheduling import analyze_cycle_body
from nat_app.graph.scheduling import classify_edges
from nat_app.graph.scheduling import compute_branch_info
from nat_app.graph.scheduling import compute_optimized_order
from nat_app.graph.topology import CycleInfo
from nat_app.graph.topology import analyze_graph_topology
from nat_app.graph.types import Graph
from tests.conftest import make_node as _node
from tests.graph.conftest import diamond_graph as _diamond_graph
from tests.graph.conftest import disjoint_cycles_graph as _disjoint_cycles_graph
from tests.graph.conftest import nested_cycle_graph as _nested_cycle_graph
from tests.graph.conftest import overlapping_cycles_graph as _overlapping_cycles_graph


class TestClassifyEdges:

    def test_necessary_edge(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        analyses = {
            "a": _node("a", writes={"x"}),
            "b": _node("b", reads={"x"}),
        }
        results = classify_edges(g, analyses)
        assert len(results) == 1
        assert results[0].edge_type == EdgeType.NECESSARY

    def test_unnecessary_edge(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        analyses = {
            "a": _node("a", writes={"x"}),
            "b": _node("b", reads={"y"}),
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

    def test_conditional_edge(self):
        g = Graph()
        g.add_node("router")
        g.add_node("target")
        g.add_edge("router", "target")
        g.add_conditional_edges("router", {"branch": ["target"]})
        analyses = {
            "router": _node("router"),
            "target": _node("target"),
        }
        results = classify_edges(g, analyses)
        assert results[0].edge_type == EdgeType.CONDITIONAL

    def test_incomplete_confidence_kept_necessary(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        analyses = {
            "a": _node("a", confidence="partial"),
            "b": _node("b"),
        }
        results = classify_edges(g, analyses)
        assert results[0].edge_type == EdgeType.NECESSARY

    def test_reducer_only_overlap_unnecessary(self):
        """Edge is unnecessary when overlap is only on reducer fields (parallel-safe)."""
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        analyses = {
            "a": _node("a", writes={"messages"}),
            "b": _node("b", reads={"messages"}),
        }
        reducer_fields = {"state": {"messages"}}
        results = classify_edges(g, analyses, reducer_fields=reducer_fields)
        assert len(results) == 1
        assert results[0].edge_type == EdgeType.UNNECESSARY

    def test_reducer_plus_non_reducer_overlap_necessary(self):
        """Edge stays necessary when overlap includes non-reducer fields."""
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        analyses = {
            "a": _node("a", writes={"messages", "x"}),
            "b": _node("b", reads={"messages", "x"}),
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

    def test_small_cycle_no_parallelism(self):
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
            "a": _node("a", writes={"x"}),
            "b": _node("b", reads={"x"}, writes={"y"}),
            "c": _node("c", reads={"y"}),
        }
        result = analyze_cycle_body(cycle, g, analyses)
        assert result is not None
        assert not result.has_parallelism

    def test_parallelizable_cycle(self):
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
            "entry": _node("entry", writes={"init"}),
            "a": _node("a", reads={"p"}, writes={"a_out"}),
            "b": _node("b", reads={"q"}, writes={"b_out"}),
            "exit": _node("exit", reads={"a_out", "b_out"}),
        }
        result = analyze_cycle_body(cycle, g, analyses)
        assert result is not None
        assert result.has_parallelism


class TestComputeOptimizedOrder:

    def test_linear_chain(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_node("c")
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.entry_point = "a"
        analyses = {
            "a": _node("a", writes={"x"}),
            "b": _node("b", reads={"x"}, writes={"y"}),
            "c": _node("c", reads={"y"}),
        }
        topo = analyze_graph_topology(g)
        order = compute_optimized_order(g, analyses, topo)
        flat = [n for stage in order for n in stage]
        assert flat.index("a") < flat.index("b") < flat.index("c")

    def test_diamond_parallelism(self):
        g = _diamond_graph()
        analyses = {
            "a": _node("a", writes={"start"}),
            "b": _node("b", reads={"start"}, writes={"b_out"}),
            "c": _node("c", reads={"start"}, writes={"c_out"}),
            "d": _node("d", reads={"b_out", "c_out"}),
        }
        topo = analyze_graph_topology(g)
        order = compute_optimized_order(g, analyses, topo)
        assert any({"b", "c"} <= stage for stage in order), "b and c should be in the same parallel stage"

    def test_disable_parallelization(self):
        g = _diamond_graph()
        analyses = {
            "a": _node("a", writes={"start"}),
            "b": _node("b", reads={"start"}, writes={"b_out"}),
            "c": _node("c", reads={"start"}, writes={"c_out"}),
            "d": _node("d", reads={"b_out", "c_out"}),
        }
        topo = analyze_graph_topology(g)
        order = compute_optimized_order(g, analyses, topo, disable_parallelization=True)
        assert all(len(stage) == 1 for stage in order)

    def test_all_nodes_present(self):
        g = _diamond_graph()
        analyses = {n: _node(n) for n in ["a", "b", "c", "d"]}
        topo = analyze_graph_topology(g)
        order = compute_optimized_order(g, analyses, topo)
        all_nodes = set()
        for stage in order:
            all_nodes |= stage
        assert all_nodes == {"a", "b", "c", "d"}

    def test_missing_nodes_treated_as_opaque(self):
        """Missing nodes in node_analyses are treated as opaque and scheduled safely."""
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_node("c")
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.entry_point = "a"
        analyses = {"a": _node("a", writes={"x"}), "b": _node("b", reads={"x"})}
        topo = analyze_graph_topology(g)
        order = compute_optimized_order(g, analyses, topo)
        all_nodes = {n for stage in order for n in stage}
        assert all_nodes == {"a", "b", "c"}
        flat = [n for stage in order for n in stage]
        assert flat.index("a") < flat.index("b") < flat.index("c")

    def test_write_write_conflict_serializes_nodes(self):
        """Two nodes writing the same non-reducer key must not be in the same stage."""
        g = _diamond_graph()
        analyses = {
            "a": _node("a", writes={"start"}),
            "b": _node("b", reads={"start"}, writes={"shared_out"}),
            "c": _node("c", reads={"start"}, writes={"shared_out"}),
            "d": _node("d", reads={"shared_out"}),
        }
        topo = analyze_graph_topology(g)
        order = compute_optimized_order(g, analyses, topo)
        for stage in order:
            assert not ({"b", "c"} <= stage), "b and c must not run in parallel (write-write conflict)"


class TestComputeOptimizedOrderMultiCycle:
    """Scheduling with nested and disjoint cycles."""

    def test_nested_cycles_all_nodes_present(self):
        g = _nested_cycle_graph()
        analyses = {n: _node(n) for n in g.node_names}
        topo = analyze_graph_topology(g)
        order = compute_optimized_order(g, analyses, topo)
        all_nodes = set()
        for stage in order:
            all_nodes |= stage
        assert all_nodes == g.node_names

    def test_nested_cycles_ordering(self):
        g = _nested_cycle_graph()
        analyses = {
            "parse": _node("parse", writes={"query"}),
            "search": _node("search", reads={"query"}, writes={"results"}),
            "evaluate": _node("evaluate", reads={"results"}, writes={"score"}),
            "refine": _node("refine", reads={"score"}, writes={"results"}),
            "decide": _node("decide", reads={"score"}),
        }
        topo = analyze_graph_topology(g)
        order = compute_optimized_order(g, analyses, topo)
        flat = [n for stage in order for n in stage]
        assert "parse" in flat
        assert "search" in flat
        assert "evaluate" in flat
        assert "decide" in flat
        assert "refine" in flat

    def test_disjoint_cycles_all_nodes_present(self):
        g = _disjoint_cycles_graph()
        analyses = {n: _node(n) for n in g.node_names}
        topo = analyze_graph_topology(g)
        order = compute_optimized_order(g, analyses, topo)
        all_nodes = set()
        for stage in order:
            all_nodes |= stage
        assert all_nodes == g.node_names

    def test_disjoint_cycles_entry_before_cycles(self):
        g = _disjoint_cycles_graph()
        analyses = {
            "entry": _node("entry", writes={"init"}),
            "loop_a": _node("loop_a", reads={"init"}, writes={"a_out"}),
            "check_a": _node("check_a", reads={"a_out"}),
            "bridge": _node("bridge", reads={"a_out"}, writes={"b_init"}),
            "loop_b": _node("loop_b", reads={"b_init"}, writes={"b_out"}),
            "check_b": _node("check_b", reads={"b_out"}),
        }
        topo = analyze_graph_topology(g)
        order = compute_optimized_order(g, analyses, topo)
        flat = [n for stage in order for n in stage]
        assert flat.index("entry") < flat.index("loop_a")

    def test_overlapping_cycles_all_nodes_present(self):
        g = _overlapping_cycles_graph()
        analyses = {n: _node(n) for n in g.node_names}
        topo = analyze_graph_topology(g)
        order = compute_optimized_order(g, analyses, topo)
        all_nodes = set()
        for stage in order:
            all_nodes |= stage
        assert all_nodes == g.node_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
