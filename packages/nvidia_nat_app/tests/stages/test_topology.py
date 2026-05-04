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
"""Tests for TopologyStage: topology metadata writes."""

import pytest

from nat_app.compiler.compilation_context import CompilationContext
from nat_app.graph.topology import GraphTopology
from nat_app.graph.types import Graph
from nat_app.stages.topology import TopologyStage


class TestTopologyStage:

    def test_name(self):
        stage = TopologyStage()
        assert stage.name == "topology"

    def test_writes_topology(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        g.entry_point = "a"
        ctx = CompilationContext(compiled=None, metadata={"graph": g})
        stage = TopologyStage()
        ctx = stage.apply(ctx)
        assert "topology" in ctx.metadata
        assert isinstance(ctx.metadata["topology"], GraphTopology)

    def test_no_cycles(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        g.entry_point = "a"
        ctx = CompilationContext(compiled=None, metadata={"graph": g})
        stage = TopologyStage()
        ctx = stage.apply(ctx)
        assert ctx.metadata["topology"].cycles == []

    def test_with_cycle(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        g.add_edge("b", "a")
        g.entry_point = "a"
        ctx = CompilationContext(compiled=None, metadata={"graph": g})
        stage = TopologyStage()
        ctx = stage.apply(ctx)
        assert len(ctx.metadata["topology"].cycles) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
