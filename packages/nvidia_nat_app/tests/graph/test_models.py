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
"""Tests for graph result types: EdgeType, EdgeAnalysis, BranchInfo, CompilationResult, TransformationResult."""

import pytest

from nat_app.graph.analysis import NodeAnalysis
from nat_app.graph.models import BranchInfo
from nat_app.graph.models import CompilationResult
from nat_app.graph.models import EdgeAnalysis
from nat_app.graph.models import EdgeType
from nat_app.graph.models import TransformationResult
from nat_app.graph.types import Graph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_graph() -> Graph:
    g = Graph()
    g.add_node("a")
    g.add_node("b")
    g.add_edge("a", "b")
    return g


def _minimal_analyses() -> dict[str, NodeAnalysis]:
    return {
        "a": NodeAnalysis(name="a"),
        "b": NodeAnalysis(name="b"),
    }


# -- EdgeType ---------------------------------------------------------------


class TestEdgeType:

    @pytest.mark.parametrize(
        "member, value",
        [
            (EdgeType.NECESSARY, "necessary"),
            (EdgeType.UNNECESSARY, "unnecessary"),
            (EdgeType.CONDITIONAL, "conditional"),
            (EdgeType.UNKNOWN, "unknown"),
        ],
    )
    def test_enum_values(self, member, value):
        assert member.value == value

    def test_all_members(self):
        assert len(EdgeType) == 4


# -- EdgeAnalysis -----------------------------------------------------------


class TestEdgeAnalysis:

    def test_required_fields(self):
        ea = EdgeAnalysis(source="a", target="b", edge_type=EdgeType.NECESSARY)
        assert ea.source == "a"
        assert ea.target == "b"
        assert ea.edge_type is EdgeType.NECESSARY

    def test_defaults(self):
        ea = EdgeAnalysis(source="a", target="b", edge_type=EdgeType.UNKNOWN)
        assert ea.reason == ""
        assert ea.shared_fields == set()

    def test_shared_fields_populated(self):
        ea = EdgeAnalysis(
            source="a",
            target="b",
            edge_type=EdgeType.NECESSARY,
            reason="write-read overlap",
            shared_fields={"query", "messages"},
        )
        assert ea.shared_fields == {"query", "messages"}
        assert ea.reason == "write-read overlap"

    def test_instance_isolation(self):
        ea1 = EdgeAnalysis(source="a", target="b", edge_type=EdgeType.NECESSARY)
        ea2 = EdgeAnalysis(source="a", target="b", edge_type=EdgeType.NECESSARY)
        ea1.shared_fields.add("x")
        assert "x" not in ea2.shared_fields


# -- BranchInfo -------------------------------------------------------------


class TestBranchInfo:

    def test_fields(self):
        bi = BranchInfo(
            router_node="router",
            branches={
                "left": {"a"}, "right": {"b"}
            },
            merge_nodes={"merge"},
            all_downstream={"a", "b", "merge"},
        )
        assert bi.router_node == "router"
        assert "left" in bi.branches
        assert bi.merge_nodes == {"merge"}
        assert len(bi.all_downstream) == 3

    def test_empty_branches(self):
        bi = BranchInfo(
            router_node="r",
            branches={},
            merge_nodes=set(),
            all_downstream=set(),
        )
        assert bi.branches == {}


# -- CompilationResult ------------------------------------------------------


class TestCompilationResult:

    def test_stages_alias(self):
        cr = CompilationResult(
            graph=_minimal_graph(),
            node_analyses=_minimal_analyses(),
            necessary_edges={("a", "b")},
            unnecessary_edges=set(),
            optimized_order=[{"a"}, {"b"}],
        )
        assert cr.stages is cr.optimized_order

    @pytest.mark.parametrize(
        "optimized_order, expected",
        [
            ([{"a", "b", "c"}, {"d"}], 4 / 2),
            ([{"a", "b"}], 2.0),
            ([], 1.0),
        ],
        ids=["multiple_stages", "single_stage", "zero_stages"],
    )
    def test_speedup_estimate(self, optimized_order, expected):
        cr = CompilationResult(
            graph=_minimal_graph(),
            node_analyses=_minimal_analyses(),
            necessary_edges=set(),
            unnecessary_edges=set(),
            optimized_order=optimized_order,
        )
        assert cr.speedup_estimate == expected

    def test_optional_defaults(self):
        cr = CompilationResult(
            graph=_minimal_graph(),
            node_analyses={},
            necessary_edges=set(),
            unnecessary_edges=set(),
            optimized_order=[],
        )
        assert cr.topology is None
        assert cr.branch_info == {}
        assert cr.cycle_body_analyses == {}


# -- TransformationResult ---------------------------------------------------


class TestTransformationResult:

    def test_inherits_compilation_result(self):
        assert issubclass(TransformationResult, CompilationResult)

    def test_field_defaults(self):
        tr = TransformationResult(
            graph=_minimal_graph(),
            node_analyses={},
            necessary_edges=set(),
            unnecessary_edges=set(),
            optimized_order=[],
        )
        assert tr.edge_analyses == []
        assert tr.parallel_groups == []
        assert tr.state_evolution == {}
        assert tr.resolved_constraints is None
        assert tr.reducer_fields == {}
        assert tr.warnings == []

    def test_stages_property_inherited(self):
        tr = TransformationResult(
            graph=_minimal_graph(),
            node_analyses={},
            necessary_edges=set(),
            unnecessary_edges=set(),
            optimized_order=[{"a"}, {"b"}],
        )
        assert tr.stages == [{"a"}, {"b"}]

    def test_speedup_estimate_inherited(self):
        tr = TransformationResult(
            graph=_minimal_graph(),
            node_analyses={},
            necessary_edges=set(),
            unnecessary_edges=set(),
            optimized_order=[{"a", "b"}, {"c"}],
        )
        assert tr.speedup_estimate == 3 / 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
