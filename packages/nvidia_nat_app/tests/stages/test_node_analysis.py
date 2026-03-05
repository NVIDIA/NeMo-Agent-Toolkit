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
"""Tests for NodeAnalysisStage: full/opaque analysis and metadata writes."""

import pytest

from nat_app.compiler.compilation_context import CompilationContext
from nat_app.graph.types import Graph
from nat_app.stages.node_analysis import NodeAnalysisStage
from tests.conftest import MinimalAdapter as _TestAdapter


class TestNodeAnalysisStage:

    def test_name(self):
        stage = NodeAnalysisStage(_TestAdapter())
        assert stage.name == "node_analysis"

    def test_function_analyzed(self):
        g = Graph()

        def fn(state):
            return {"result": state["query"]}

        g.add_node("a", func=fn)
        g.entry_point = "a"
        ctx = CompilationContext(compiled=None, metadata={"graph": g})
        stage = NodeAnalysisStage(_TestAdapter())
        ctx = stage.apply(ctx)
        assert "a" in ctx.metadata["node_analyses"]
        assert ctx.metadata["node_analyses"]["a"].confidence == "full"

    def test_no_function_opaque(self):
        g = Graph()
        g.add_node("a")
        g.entry_point = "a"
        ctx = CompilationContext(compiled=None, metadata={"graph": g})
        stage = NodeAnalysisStage(_TestAdapter())
        ctx = stage.apply(ctx)
        assert ctx.metadata["node_analyses"]["a"].confidence == "opaque"

    @pytest.mark.parametrize(
        "metadata_key",
        ["node_funcs", "resolved_constraints", "state_evolution"],
    )
    def test_writes_metadata_key(self, metadata_key):
        g = Graph()
        g.add_node("a", func=lambda s: {"result": s.get("query")})
        g.entry_point = "a"
        ctx = CompilationContext(compiled=None, metadata={"graph": g})
        stage = NodeAnalysisStage(_TestAdapter())
        ctx = stage.apply(ctx)
        assert metadata_key in ctx.metadata

    def test_state_evolution_structure(self):
        g = Graph()

        def fn(state):
            return {"result": state["query"]}

        g.add_node("a", func=fn)
        g.entry_point = "a"
        ctx = CompilationContext(compiled=None, metadata={"graph": g})
        stage = NodeAnalysisStage(_TestAdapter())
        ctx = stage.apply(ctx)
        assert "reads" in ctx.metadata["state_evolution"]["a"]
        assert "writes" in ctx.metadata["state_evolution"]["a"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
