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
"""Tests for GraphOptimizer, context_to_result, and GraphValidationError."""

import pytest

from nat_app.compiler.compilation_context import CompilationContext
from nat_app.compiler.default_graph_compiler import context_to_result
from nat_app.compiler.errors import GraphValidationError
from nat_app.compiler.optimizer import GraphOptimizer
from nat_app.graph.analysis import NodeAnalysis
from nat_app.graph.types import Graph
from tests.conftest import MinimalAdapter as _TestAdapter


class TestGraphValidationError:

    def test_is_value_error(self):
        assert issubclass(GraphValidationError, ValueError)

    def test_issues_stored(self):
        err = GraphValidationError(["issue1", "issue2"])
        assert err.issues == ["issue1", "issue2"]

    def test_message_contains_issues(self):
        err = GraphValidationError(["bad node"])
        assert "bad node" in str(err)


class TestGraphOptimizer:

    def test_optimize_simple_graph(self):
        g = Graph()
        g.add_node("a", func=lambda s: {"x": 1})
        g.add_node("b", func=lambda s: {"y": s["x"]})
        g.add_edge("a", "b")
        g.entry_point = "a"
        g.terminal_nodes = {"b"}

        optimizer = GraphOptimizer(adapter=_TestAdapter())
        result = optimizer.optimize(g)
        assert result.optimized_order is not None
        all_nodes = set()
        for stage in result.optimized_order:
            all_nodes |= stage
        assert all_nodes == {"a", "b"}

    def test_default_config(self):
        optimizer = GraphOptimizer(adapter=_TestAdapter())
        assert optimizer.config is not None

    def test_optimize_and_build_equivalent_to_two_step(self):
        """optimize_and_build returns same result as optimize + adapter.build."""
        g = Graph()
        g.add_node("a", func=lambda s: {"x": 1})
        g.add_node("b", func=lambda s: {"y": s["x"]})
        g.add_edge("a", "b")
        g.entry_point = "a"
        g.terminal_nodes = {"b"}

        optimizer = GraphOptimizer(adapter=_TestAdapter())
        one_call = optimizer.optimize_and_build(g)
        two_step = optimizer.adapter.build(g, optimizer.optimize(g))

        # MinimalAdapter.build returns result; both paths yield equivalent output
        assert one_call.optimized_order == two_step.optimized_order
        assert one_call.graph is two_step.graph
        all_nodes = set()
        for stage in one_call.optimized_order:
            all_nodes |= stage
        assert all_nodes == {"a", "b"}


class TestContextToResult:

    def test_complete_context(self):
        g = Graph()
        g.add_node("a")
        analyses = {"a": NodeAnalysis(name="a")}
        ctx = CompilationContext(
            compiled=None,
            metadata={
                "graph": g,
                "node_analyses": analyses,
                "edge_analyses": [],
                "necessary_edges": {("a", "b")},
                "unnecessary_edges": set(),
                "parallel_groups": [],
                "optimized_order": [{"a"}],
                "state_evolution": {},
                "topology": None,
                "resolved_constraints": {},
                "reducer_fields": {},
                "branch_info": {},
                "cycle_body_analyses": {},
                "warnings": ["test warning"],
            },
        )
        result = context_to_result(ctx)
        assert result.graph is g
        assert result.node_analyses is analyses
        assert result.optimized_order == [{"a"}]
        assert result.warnings == ["test warning"]

    def test_missing_optional_keys(self):
        g = Graph()
        g.add_node("a")
        ctx = CompilationContext(
            compiled=None,
            metadata={
                "graph": g,
                "node_analyses": {
                    "a": NodeAnalysis(name="a")
                },
            },
        )
        result = context_to_result(ctx)
        assert result.edge_analyses == []
        assert result.necessary_edges == set()
        assert result.optimized_order == []
        assert result.warnings == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
