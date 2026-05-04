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
"""Tests for CompilationContext: property accessors, mutability, defaults."""

import pytest

from nat_app.compiler.compilation_context import CompilationContext


class TestConstruction:

    def test_compiled_stored(self):
        ctx = CompilationContext(compiled="artifact")
        assert ctx.compiled == "artifact"

    def test_metadata_default_empty(self):
        ctx = CompilationContext(compiled=None)
        assert ctx.metadata == {}

    def test_metadata_provided(self):
        ctx = CompilationContext(compiled=None, metadata={"key": "value"})
        assert ctx.metadata["key"] == "value"


class TestPropertyAccessors:

    def test_graph_none_when_missing(self):
        ctx = CompilationContext(compiled=None)
        assert ctx.graph is None

    def test_graph_returns_value(self):
        sentinel = object()
        ctx = CompilationContext(compiled=None, metadata={"graph": sentinel})
        assert ctx.graph is sentinel

    def test_topology_none_when_missing(self):
        ctx = CompilationContext(compiled=None)
        assert ctx.topology is None

    def test_topology_returns_value(self):
        sentinel = object()
        ctx = CompilationContext(compiled=None, metadata={"topology": sentinel})
        assert ctx.topology is sentinel

    def test_node_analyses_none_when_missing(self):
        ctx = CompilationContext(compiled=None)
        assert ctx.node_analyses is None

    def test_node_analyses_returns_value(self):
        analyses = {"a": "analysis_a"}
        ctx = CompilationContext(compiled=None, metadata={"node_analyses": analyses})
        assert ctx.node_analyses is analyses

    def test_optimized_order_none_when_missing(self):
        ctx = CompilationContext(compiled=None)
        assert ctx.optimized_order is None

    def test_optimized_order_returns_value(self):
        order = [{"a"}, {"b"}]
        ctx = CompilationContext(compiled=None, metadata={"optimized_order": order})
        assert ctx.optimized_order is order

    def test_necessary_edges_none_when_missing(self):
        ctx = CompilationContext(compiled=None)
        assert ctx.necessary_edges is None

    def test_necessary_edges_returns_value(self):
        edges = {("a", "b")}
        ctx = CompilationContext(compiled=None, metadata={"necessary_edges": edges})
        assert ctx.necessary_edges is edges

    def test_unnecessary_edges_none_when_missing(self):
        ctx = CompilationContext(compiled=None)
        assert ctx.unnecessary_edges is None

    def test_unnecessary_edges_returns_value(self):
        edges = {("a", "b")}
        ctx = CompilationContext(compiled=None, metadata={"unnecessary_edges": edges})
        assert ctx.unnecessary_edges is edges


class TestMutability:

    def test_compiled_reassignment(self):
        ctx = CompilationContext(compiled="old")
        ctx.compiled = "new"
        assert ctx.compiled == "new"

    def test_metadata_mutation_reflected(self):
        ctx = CompilationContext(compiled=None)
        ctx.metadata["graph"] = "my_graph"
        assert ctx.graph == "my_graph"

    def test_metadata_update_changes_property(self):
        ctx = CompilationContext(compiled=None, metadata={"graph": "old"})
        assert ctx.graph == "old"
        ctx.metadata["graph"] = "new"
        assert ctx.graph == "new"

    def test_instance_isolation(self):
        ctx1 = CompilationContext(compiled=None)
        ctx2 = CompilationContext(compiled=None)
        ctx1.metadata["graph"] = "g1"
        assert ctx2.graph is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
