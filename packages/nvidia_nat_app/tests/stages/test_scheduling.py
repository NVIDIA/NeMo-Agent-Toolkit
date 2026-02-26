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
"""Tests for SchedulingStage: branch info, cycle body, and optimized order."""

import pytest

from nat_app.compiler.compilation_context import CompilationContext
from nat_app.constraints import OptimizationConfig
from nat_app.graph.topology import NodeType
from nat_app.graph.topology import analyze_graph_topology
from nat_app.graph.types import Graph
from nat_app.stages.scheduling import SchedulingStage
from tests.conftest import make_node as _node
from tests.graph.conftest import parallelizable_cycle_graph as _parallelizable_cycle_graph


def _build_ctx(g, analyses):
    topo = analyze_graph_topology(g)
    return CompilationContext(
        compiled=None,
        metadata={
            "graph": g,
            "node_analyses": analyses,
            "topology": topo,
            "reducer_fields": {},
            "resolved_constraints": {},
        },
    )


class TestSchedulingStage:

    def test_name(self):
        stage = SchedulingStage()
        assert stage.name == "scheduling"

    def test_writes_optimized_order(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        g.entry_point = "a"
        analyses = {
            "a": _node("a", writes={"x"}),
            "b": _node("b", reads={"x"}),
        }
        ctx = _build_ctx(g, analyses)
        stage = SchedulingStage()
        ctx = stage.apply(ctx)
        assert "optimized_order" in ctx.metadata
        all_nodes = set()
        for s in ctx.metadata["optimized_order"]:
            all_nodes |= s
        assert all_nodes == {"a", "b"}

    def test_writes_branch_info(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        g.entry_point = "a"
        analyses = {"a": _node("a"), "b": _node("b")}
        ctx = _build_ctx(g, analyses)
        stage = SchedulingStage()
        ctx = stage.apply(ctx)
        assert "branch_info" in ctx.metadata

    def test_writes_cycle_body_analyses(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        g.entry_point = "a"
        analyses = {"a": _node("a"), "b": _node("b")}
        ctx = _build_ctx(g, analyses)
        stage = SchedulingStage()
        ctx = stage.apply(ctx)
        assert "cycle_body_analyses" in ctx.metadata

    def test_disable_parallelization(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_node("c")
        g.add_edge("a", "b")
        g.add_edge("a", "c")
        g.entry_point = "a"
        analyses = {
            "a": _node("a", writes={"init"}),
            "b": _node("b", reads={"init"}, writes={"b_out"}),
            "c": _node("c", reads={"init"}, writes={"c_out"}),
        }
        config = OptimizationConfig(disable_parallelization=True)
        ctx = _build_ctx(g, analyses)
        stage = SchedulingStage(config)
        ctx = stage.apply(ctx)
        assert all(len(s) == 1 for s in ctx.metadata["optimized_order"])

    def test_cycle_with_intra_cycle_parallelism(self):
        """Cycle body with parallelizable nodes sets CYCLE_MEMBER_PARALLELIZABLE."""
        g = _parallelizable_cycle_graph()
        analyses = {
            "entry": _node("entry", writes={"init"}),
            "a": _node("a", reads={"p"}, writes={"a_out"}),
            "b": _node("b", reads={"q"}, writes={"b_out"}),
            "exit": _node("exit", reads={"a_out", "b_out"}),
        }
        ctx = _build_ctx(g, analyses)
        stage = SchedulingStage()
        ctx = stage.apply(ctx)

        cycle_body = ctx.metadata["cycle_body_analyses"]
        assert len(cycle_body) >= 1
        body_analysis = next(iter(cycle_body.values()))
        assert body_analysis.has_parallelism

        topo = ctx.metadata["topology"]
        parallelizable = [n for n, t in topo.node_types.items() if t == NodeType.CYCLE_MEMBER_PARALLELIZABLE]
        assert len(parallelizable) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
