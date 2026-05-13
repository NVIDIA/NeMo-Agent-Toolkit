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
"""Tests for ExtractStage: adapter extraction and metadata writes."""

import pytest

from nat_app.compiler.compilation_context import CompilationContext
from nat_app.graph.adapter import AbstractFrameworkAdapter
from nat_app.graph.types import Graph
from nat_app.stages.extract import ExtractStage


class _TestAdapter(AbstractFrameworkAdapter):

    def __init__(
        self,
        graph=None,
        reducer_fields=None,
        schema_fields=None,
        state_schema=None,
    ):
        self._graph = graph or Graph()
        self._reducer_fields = reducer_fields or {}
        self._schema_fields = schema_fields
        self._state_schema = state_schema

    def extract(self, source):
        return self._graph

    def build(self, original, result):
        return result

    def get_reducer_fields(self):
        return self._reducer_fields

    def get_all_schema_fields(self):
        return self._schema_fields

    def get_state_schema(self):
        return self._state_schema


class TestExtractStage:

    def test_name(self):
        stage = ExtractStage(_TestAdapter())
        assert stage.name == "extract"

    def test_populates_graph(self):
        g = Graph()
        g.add_node("a")
        stage = ExtractStage(_TestAdapter(graph=g))
        ctx = CompilationContext(compiled="source")
        ctx = stage.apply(ctx)
        assert ctx.metadata["graph"] is g

    def test_populates_reducer_fields(self):
        stage = ExtractStage(_TestAdapter(reducer_fields={"state": {"messages"}}))
        ctx = CompilationContext(compiled="source")
        ctx = stage.apply(ctx)
        assert ctx.metadata["reducer_fields"] == {"state": {"messages"}}

    def test_populates_schema_fields(self):
        stage = ExtractStage(_TestAdapter(schema_fields={"a", "b"}))
        ctx = CompilationContext(compiled="source")
        ctx = stage.apply(ctx)
        assert ctx.metadata["all_schema_fields"] == {"a", "b"}

    def test_populates_state_schema(self):

        class MySchema:
            pass

        stage = ExtractStage(_TestAdapter(state_schema=MySchema))
        ctx = CompilationContext(compiled="source")
        ctx = stage.apply(ctx)
        assert ctx.metadata["state_schema"] is MySchema

    def test_empty_reducer_fields(self):
        stage = ExtractStage(_TestAdapter())
        ctx = CompilationContext(compiled="source")
        ctx = stage.apply(ctx)
        assert ctx.metadata["reducer_fields"] == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
