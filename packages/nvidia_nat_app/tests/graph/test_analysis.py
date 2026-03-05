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
"""Tests for NodeAnalysis, dependency graph, parallel group finding, and GraphAnalysisResult."""

import pytest

from nat_app.graph.access import AccessSet
from nat_app.graph.analysis import GraphAnalysisResult
from nat_app.graph.analysis import NodeAnalysis
from nat_app.graph.analysis import build_dependency_graph
from nat_app.graph.analysis import find_parallel_groups
from tests.conftest import make_node as _node

# -- NodeAnalysis.conflicts_with -------------------------------------------


class TestConflictsWith:

    @pytest.mark.parametrize(
        "a_kwargs, b_kwargs, expected",
        [
            ({
                "reads": {"x"}, "writes": {"y"}
            }, {
                "reads": {"z"}, "writes": {"w"}
            }, False),
            ({
                "writes": {"x"}
            }, {
                "writes": {"x"}
            }, True),
            ({
                "writes": {"x"}
            }, {
                "reads": {"x"}
            }, True),
            ({
                "reads": {"x"}
            }, {
                "writes": {"x"}
            }, True),
            ({
                "special_calls": {"Send"}
            }, {}, True),
            ({}, {
                "special_calls": {"Command"}
            }, True),
        ],
        ids=[
            "disjoint_no_conflict",
            "write_write_conflict",
            "read_write_conflict",
            "reverse_read_write_conflict",
            "special_calls_barrier",
            "special_calls_on_other",
        ],
    )
    def test_conflict_detection(self, a_kwargs, b_kwargs, expected):
        a = _node("a", **a_kwargs)
        b = _node("b", **b_kwargs)
        assert a.conflicts_with(b) is expected

    def test_reducer_excludes_write_write(self):
        a = _node("a", writes={"messages"})
        b = _node("b", writes={"messages"})
        reducers = {"state": {"messages"}}
        assert not a.conflicts_with(b, reducer_fields=reducers)

    def test_reducer_does_not_exclude_read_write(self):
        a = _node("a", writes={"messages"})
        b = _node("b", reads={"messages"})
        reducers = {"state": {"messages"}}
        assert a.conflicts_with(b, reducer_fields=reducers)


# -- NodeAnalysis property setters -----------------------------------------


class TestPropertySetters:

    def test_state_reads_getter(self):
        na = _node("a", reads={"x", "y"})
        assert na.state_reads == {"x", "y"}

    def test_state_reads_setter(self):
        na = NodeAnalysis(name="a")
        na.state_reads = {"x", "y"}
        assert na.state_reads == {"x", "y"}
        assert isinstance(na.reads, AccessSet)

    def test_state_writes_getter(self):
        na = _node("a", writes={"x"})
        assert na.state_writes == {"x"}

    def test_state_writes_setter(self):
        na = NodeAnalysis(name="a")
        na.state_writes = {"x", "y"}
        assert na.state_writes == {"x", "y"}

    def test_repr(self):
        na = _node("a", reads={"x"}, writes={"y"})
        r = repr(na)
        assert "a" in r
        assert "confidence=full" in r


# -- build_dependency_graph ------------------------------------------------


class TestBuildDependencyGraph:

    def test_empty(self):
        deps = build_dependency_graph({})
        assert deps == {}

    def test_no_dependencies(self):
        analyses = {
            "a": _node("a", reads={"x"}, writes={"y"}),
            "b": _node("b", reads={"z"}, writes={"w"}),
        }
        deps = build_dependency_graph(analyses)
        assert deps["a"] == set()
        assert deps["b"] == set()

    def test_write_read_dependency(self):
        analyses = {
            "a": _node("a", writes={"x"}),
            "b": _node("b", reads={"x"}),
        }
        deps = build_dependency_graph(analyses)
        assert "a" in deps["b"]
        assert "b" not in deps["a"]

    def test_reducer_exclusion(self):
        analyses = {
            "a": _node("a", writes={"messages"}),
            "b": _node("b", reads={"messages"}),
        }
        reducers = {"state": {"messages"}}
        deps = build_dependency_graph(analyses, reducer_fields=reducers)
        assert deps["b"] == set()

    def test_bidirectional_dependency(self):
        analyses = {
            "a": _node("a", reads={"y"}, writes={"x"}),
            "b": _node("b", reads={"x"}, writes={"y"}),
        }
        deps = build_dependency_graph(analyses)
        assert "b" in deps["a"]
        assert "a" in deps["b"]


# -- find_parallel_groups --------------------------------------------------


class TestFindParallelGroups:

    def test_independent_pair(self):
        analyses = {
            "a": _node("a", reads={"x"}, writes={"y"}),
            "b": _node("b", reads={"z"}, writes={"w"}),
        }
        deps = build_dependency_graph(analyses)
        groups = find_parallel_groups(analyses, deps)
        assert len(groups) == 1
        assert groups[0] == {"a", "b"}

    def test_no_parallel_groups(self):
        analyses = {
            "a": _node("a", writes={"x"}),
            "b": _node("b", reads={"x"}),
        }
        deps = build_dependency_graph(analyses)
        groups = find_parallel_groups(analyses, deps)
        assert groups == []

    def test_three_node_parallel(self):
        analyses = {
            "a": _node("a", reads={"x"}, writes={"a_out"}),
            "b": _node("b", reads={"y"}, writes={"b_out"}),
            "c": _node("c", reads={"z"}, writes={"c_out"}),
        }
        deps = build_dependency_graph(analyses)
        groups = find_parallel_groups(analyses, deps)
        assert any(len(g) == 3 for g in groups)

    def test_dependency_prevents_grouping(self):
        analyses = {
            "a": _node("a", writes={"x"}),
            "b": _node("b", reads={"x"}, writes={"y"}),
            "c": _node("c", reads={"z"}, writes={"w"}),
        }
        deps = build_dependency_graph(analyses)
        groups = find_parallel_groups(analyses, deps)
        for g in groups:
            assert not ({"a", "b"} <= g), "a and b should not be in the same group"

    def test_transitive_merge_respects_dependencies(self):
        """Nodes with dependency must not be merged transitively via intermediate pair."""
        analyses = {
            "a": _node("a", reads=set(), writes={"a_out"}),
            "b": _node("b", reads=set(), writes={"b_out"}),
            "c": _node("c", reads=set(), writes={"c_out"}),
        }
        # C depends on A (e.g. explicit constraint); no data conflict between A and C
        deps = {"a": set(), "b": set(), "c": {"a"}}
        groups = find_parallel_groups(analyses, deps)
        for g in groups:
            assert not ({"a", "c"} <= g), "a and c have dependency, must not be in same group"


# -- GraphAnalysisResult ---------------------------------------------------


class TestGraphAnalysisResult:

    def test_defaults(self):
        r = GraphAnalysisResult()
        assert r.node_analyses == {}
        assert r.total_nodes == 0
        assert r.warnings == []

    def test_get_execution_order_linear(self):
        r = GraphAnalysisResult(
            node_analyses={
                "a": _node("a"),
                "b": _node("b"),
                "c": _node("c"),
            },
            dependency_graph={
                "a": set(),
                "b": {"a"},
                "c": {"b"},
            },
        )
        order = r.get_execution_order()
        assert order[0] == {"a"}
        assert order[1] == {"b"}
        assert order[2] == {"c"}

    def test_get_execution_order_parallel(self):
        r = GraphAnalysisResult(
            node_analyses={
                "a": _node("a"),
                "b": _node("b"),
                "c": _node("c"),
            },
            dependency_graph={
                "a": set(),
                "b": set(),
                "c": {"a", "b"},
            },
        )
        order = r.get_execution_order()
        assert order[0] == {"a", "b"}
        assert order[1] == {"c"}

    def test_get_execution_order_circular(self):
        r = GraphAnalysisResult(
            node_analyses={
                "a": _node("a"),
                "b": _node("b"),
            },
            dependency_graph={
                "a": {"b"},
                "b": {"a"},
            },
        )
        order = r.get_execution_order()
        assert len(order) == 1
        assert order[0] == {"a", "b"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
