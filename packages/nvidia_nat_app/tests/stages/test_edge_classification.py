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
"""Tests for EdgeClassificationStage: edge sets and parallel groups."""

import pytest

from nat_app.compiler.compilation_context import CompilationContext
from nat_app.graph.types import Graph
from nat_app.stages.edge_classification import EdgeClassificationStage
from tests.conftest import make_node as _node


class TestEdgeClassificationStage:

    def test_name(self):
        stage = EdgeClassificationStage()
        assert stage.name == "edge_classification"

    @pytest.mark.parametrize(
        "b_reads, expected_set",
        [
            ({"x"}, "necessary_edges"),
            ({"y"}, "unnecessary_edges"),
        ],
        ids=["necessary", "unnecessary"],
    )
    def test_edge_classification(self, b_reads, expected_set):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        analyses = {
            "a": _node("a", writes={"x"}),
            "b": _node("b", reads=b_reads),
        }
        ctx = CompilationContext(
            compiled=None,
            metadata={
                "graph": g,
                "node_analyses": analyses,
                "reducer_fields": {},
            },
        )
        stage = EdgeClassificationStage()
        ctx = stage.apply(ctx)
        assert ("a", "b") in ctx.metadata[expected_set]

    def test_writes_all_metadata(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        analyses = {"a": _node("a"), "b": _node("b")}
        ctx = CompilationContext(
            compiled=None,
            metadata={
                "graph": g,
                "node_analyses": analyses,
            },
        )
        stage = EdgeClassificationStage()
        ctx = stage.apply(ctx)
        assert "edge_analyses" in ctx.metadata
        assert "necessary_edges" in ctx.metadata
        assert "unnecessary_edges" in ctx.metadata
        assert "parallel_groups" in ctx.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
