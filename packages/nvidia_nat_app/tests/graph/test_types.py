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
"""Tests for NodeInfo priority attribute and Graph priority propagation."""

import pytest

from nat_app.graph.types import Edge
from nat_app.graph.types import EdgeKind
from nat_app.graph.types import Graph
from nat_app.graph.types import NodeInfo


class TestNodeInfoPriority:

    def test_default_priority_is_none(self):
        info = NodeInfo()
        assert info.priority is None

    def test_explicit_priority(self):
        info = NodeInfo(priority=0.8)
        assert info.priority == 0.8

    def test_priority_with_func(self):
        info = NodeInfo(func=lambda s: s, priority=0.5)
        assert info.func is not None
        assert info.priority == 0.5

    def test_priority_zero_is_not_none(self):
        info = NodeInfo(priority=0.0)
        assert info.priority is not None
        assert info.priority == 0.0


class TestGraphAddNodePriority:

    def test_add_node_default_priority(self):
        g = Graph()
        g.add_node("a", func=None)
        assert g.get_node("a").priority is None

    def test_add_node_explicit_priority(self):
        g = Graph()
        g.add_node("a", func=None, priority=0.9)
        assert g.get_node("a").priority == 0.9

    def test_add_node_priority_with_metadata(self):
        g = Graph()
        g.add_node("a", func=None, priority=0.7, label="fast_route")
        node = g.get_node("a")
        assert node.priority == 0.7
        assert node.metadata["label"] == "fast_route"

    def test_minimal_factory_no_priority(self):
        g = Graph.minimal(
            nodes={
                "a": None, "b": None
            },
            edges=[("a", "b")],
        )
        assert g.get_node("a").priority is None
        assert g.get_node("b").priority is None


class TestGraphSubgraphPriority:

    def test_subgraph_preserves_priority(self):
        g = Graph()
        g.add_node("a", func=None, priority=0.9)
        g.add_node("b", func=None, priority=0.3)
        g.add_node("c", func=None)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.entry_point = "a"

        sub = g.subgraph({"a", "b"})
        assert sub.get_node("a").priority == 0.9
        assert sub.get_node("b").priority == 0.3

    def test_subgraph_preserves_none_priority(self):
        g = Graph()
        g.add_node("a", func=None)
        g.add_node("b", func=None)
        g.add_edge("a", "b")
        g.entry_point = "a"

        sub = g.subgraph({"a"})
        assert sub.get_node("a").priority is None


class TestStructureHashIgnoresPriority:

    def test_conditional_edges_different_branches_different_hash(self):
        """Graphs with same topology but different branch labels must have different structure_hash."""
        g1 = Graph()
        g1.add_node("r", func=None)
        g1.add_node("a", func=None)
        g1.add_node("b", func=None)
        g1.add_conditional_edges("r", {"left": "a", "right": "b"})

        g2 = Graph()
        g2.add_node("r", func=None)
        g2.add_node("a", func=None)
        g2.add_node("b", func=None)
        g2.add_conditional_edges("r", {"x": "a", "y": "b"})

        assert g1.structure_hash != g2.structure_hash

    def test_edge_branch_in_equality(self):
        """Edges with same source/target/kind but different branch must not compare equal."""
        e1 = Edge(source="r", target="a", kind=EdgeKind.CONDITIONAL, branch="left")
        e2 = Edge(source="r", target="a", kind=EdgeKind.CONDITIONAL, branch="right")
        assert e1 != e2
        assert hash(e1) != hash(e2)

    def test_same_hash_different_priorities(self):
        g1 = Graph()
        g1.add_node("a", func=None, priority=0.1)
        g1.add_node("b", func=None, priority=0.9)
        g1.add_edge("a", "b")

        g2 = Graph()
        g2.add_node("a", func=None, priority=0.9)
        g2.add_node("b", func=None, priority=0.1)
        g2.add_edge("a", "b")

        assert g1.structure_hash == g2.structure_hash

    def test_same_hash_with_and_without_priority(self):
        g1 = Graph()
        g1.add_node("a", func=None)
        g1.add_edge("a", "a")

        g2 = Graph()
        g2.add_node("a", func=None, priority=0.5)
        g2.add_edge("a", "a")

        assert g1.structure_hash == g2.structure_hash


class TestDuplicateEdgeDeduplication:

    def test_add_edge_twice_yields_single_edge(self):
        g = Graph()
        g.add_node("a", func=None)
        g.add_node("b", func=None)
        g.add_edge("a", "b")
        g.add_edge("a", "b")
        assert g.edge_count == 1
        assert set(g.successors("a")) == {"b"}
        assert set(g.predecessors("b")) == {"a"}
        assert g.edge_pairs == [("a", "b")]

    def test_add_conditional_edges_duplicate_branch_deduplicated(self):
        g = Graph()
        g.add_node("r", func=None)
        g.add_node("a", func=None)
        g.add_conditional_edges("r", {"branch": "a"})
        g.add_conditional_edges("r", {"branch": "a"})
        assert g.edge_count == 1
        assert set(g.successors("r")) == {"a"}

    def test_add_conditional_edges_replace_removes_old(self):
        """Replace semantics: second call removes old conditional edges for source."""
        g = Graph()
        g.add_node("r", func=None)
        g.add_node("x", func=None)
        g.add_node("y", func=None)
        g.add_conditional_edges("r", {"a": "x"})
        g.add_conditional_edges("r", {"b": "y"})
        assert g.get_conditional_targets("r") == {"b": ["y"]}
        assert g.edge_count == 1
        edge = g.edges[0]
        assert edge.branch == "b"
        assert edge.target == "y"


class TestGraphValidate:

    def test_no_entry_point(self):
        g = Graph()
        g.add_node("a", func=None)
        g.add_node("b", func=None)
        g.add_edge("a", "b")
        issues = g.validate()
        assert "entry_point" in issues[0].lower() or "entry" in issues[0].lower()

    def test_invalid_entry_point(self):
        g = Graph()
        g.add_node("a", func=None)
        g.add_node("b", func=None)
        g.add_edge("a", "b")
        g.entry_point = "nonexistent"
        issues = g.validate()
        assert any("nonexistent" in i for i in issues)

    def test_orphan_nodes(self):
        g = Graph()
        g.add_node("a", func=None)
        g.add_node("b", func=None)
        g.add_node("orphan", func=None)
        g.add_edge("a", "b")
        g.entry_point = "a"
        issues = g.validate()
        assert any("orphan" in i.lower() or "unreachable" in i.lower() for i in issues)

    def test_invalid_edge_source(self):
        g = Graph()
        g.add_node("a", func=None)
        g.add_node("b", func=None)
        g.add_edge("missing", "b")
        g.entry_point = "a"
        issues = g.validate()
        assert any("source" in i.lower() or "missing" in i for i in issues)

    def test_invalid_edge_target(self):
        g = Graph()
        g.add_node("a", func=None)
        g.add_node("b", func=None)
        g.add_edge("a", "missing")
        g.entry_point = "a"
        issues = g.validate()
        assert any("target" in i.lower() or "missing" in i for i in issues)

    def test_invalid_terminal_node(self):
        g = Graph()
        g.add_node("a", func=None)
        g.add_node("b", func=None)
        g.add_edge("a", "b")
        g.entry_point = "a"
        g.terminal_nodes.add("nonexistent")
        issues = g.validate()
        assert any("terminal" in i.lower() or "nonexistent" in i for i in issues)

    def test_valid_graph_returns_empty(self):
        g = Graph()
        g.add_node("a", func=None)
        g.add_node("b", func=None)
        g.add_edge("a", "b")
        g.entry_point = "a"
        issues = g.validate()
        assert issues == []


class TestGraphGetNode:

    def test_get_node_raises_key_error_for_missing(self):
        g = Graph()
        g.add_node("a", func=None)
        with pytest.raises(KeyError, match="missing"):
            g.get_node("missing")


class TestGraphHasNode:

    def test_has_node_true(self):
        g = Graph()
        g.add_node("a", func=None)
        assert g.has_node("a") is True

    def test_has_node_false(self):
        g = Graph()
        g.add_node("a", func=None)
        assert g.has_node("b") is False


class TestGraphToAdjacency:

    def test_to_adjacency_returns_correct_mapping(self):
        g = Graph()
        g.add_node("a", func=None)
        g.add_node("b", func=None)
        g.add_node("c", func=None)
        g.add_edge("a", "b")
        g.add_edge("a", "c")
        g.add_edge("b", "c")
        adj = g.to_adjacency()
        assert set(adj["a"]) == {"b", "c"}
        assert set(adj["b"]) == {"c"}
        assert "c" in adj and adj["c"] == []


class TestGraphConditionalTargets:

    def test_get_conditional_targets_returns_mapping(self):
        g = Graph()
        g.add_node("r", func=None)
        g.add_node("a", func=None)
        g.add_node("b", func=None)
        g.add_conditional_edges("r", {"left": "a", "right": "b"})
        targets = g.get_conditional_targets("r")
        assert targets is not None
        assert "left" in targets
        assert "right" in targets
        assert targets["left"] == ["a"]
        assert targets["right"] == ["b"]

    def test_get_conditional_targets_returns_none_for_non_conditional(self):
        g = Graph()
        g.add_node("a", func=None)
        g.add_node("b", func=None)
        g.add_edge("a", "b")
        assert g.get_conditional_targets("a") is None
        assert g.get_conditional_targets("b") is None

    def test_conditional_edge_sources_property(self):
        g = Graph()
        g.add_node("r", func=None)
        g.add_node("a", func=None)
        g.add_conditional_edges("r", {"branch": "a"})
        sources = g.conditional_edge_sources
        assert "r" in sources
        assert sources["r"]["branch"] == ["a"]


class TestGraphNodeCountNamesNodes:

    def test_node_count(self):
        g = Graph()
        g.add_node("a", func=None)
        g.add_node("b", func=None)
        assert g.node_count == 2

    def test_node_names(self):
        g = Graph()
        g.add_node("a", func=None)
        g.add_node("b", func=None)
        assert g.node_names == {"a", "b"}

    def test_nodes_iterator(self):
        g = Graph()
        g.add_node("a", func=None)
        g.add_node("b", func=None)
        items = list(g.nodes())
        assert len(items) == 2
        names = {n for n, _ in items}
        assert names == {"a", "b"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
