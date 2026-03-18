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
"""Tests for DefaultGraphCompiler: stages, append/insert, compile_to_result."""

import pytest

from nat_app.compiler.compilation_context import CompilationContext
from nat_app.compiler.default_graph_compiler import DefaultGraphCompiler
from nat_app.graph.models import TransformationResult
from tests.conftest import MinimalAdapter as _SimpleAdapter
from tests.graph.conftest import simple_graph as _simple_graph


class _DummyStage:

    def __init__(self, name_val):
        self._name = name_val

    @property
    def name(self):
        return self._name

    def apply(self, context, **kwargs):
        context.metadata.setdefault("custom_stages", []).append(self._name)
        return context


class TestDefaultStages:

    def test_six_default_stages(self):
        compiler = DefaultGraphCompiler(_SimpleAdapter())
        assert len(compiler.stages) == 6

    def test_stage_names(self):
        compiler = DefaultGraphCompiler(_SimpleAdapter())
        names = [s.name for s in compiler.stages]
        assert names == [
            "extract",
            "validate",
            "topology",
            "node_analysis",
            "edge_classification",
            "scheduling",
        ]


class TestPrepareFinalize:

    def test_prepare_returns_source(self):
        compiler = DefaultGraphCompiler(_SimpleAdapter())
        assert compiler.prepare("src") == "src"

    def test_finalize_returns_context(self):
        compiler = DefaultGraphCompiler(_SimpleAdapter())
        ctx = CompilationContext(compiled="test")
        assert compiler.finalize(ctx) is ctx


class TestAppendInsert:

    def test_append_stage(self):
        compiler = DefaultGraphCompiler(_SimpleAdapter())
        compiler.append_stage(_DummyStage("custom"))
        assert len(compiler.stages) == 7
        assert compiler.stages[-1].name == "custom"

    def test_insert_stage_after(self):
        compiler = DefaultGraphCompiler(_SimpleAdapter())
        compiler.insert_stage_after("topology", _DummyStage("custom"))
        names = [s.name for s in compiler.stages]
        idx = names.index("custom")
        assert names[idx - 1] == "topology"

    def test_insert_after_nonexistent_appends(self):
        compiler = DefaultGraphCompiler(_SimpleAdapter())
        compiler.insert_stage_after("nonexistent", _DummyStage("orphan"))
        # If name not found, stage is NOT appended (no match means no insertion)
        names = [s.name for s in compiler.stages]
        assert "orphan" not in names


class TestCompileToResult:

    def test_returns_transformation_result(self):
        compiler = DefaultGraphCompiler(_SimpleAdapter())
        result = compiler.compile_to_result(_simple_graph())
        assert isinstance(result, TransformationResult)
        all_nodes = set()
        for stage in result.optimized_order:
            all_nodes |= stage
        assert all_nodes == {"a", "b"}


class TestEndToEnd:

    def test_compile_simple_graph(self):
        compiler = DefaultGraphCompiler(_SimpleAdapter())
        ctx = compiler.compile(_simple_graph())
        assert ctx.optimized_order is not None
        all_nodes = set()
        for stage in ctx.optimized_order:
            all_nodes |= stage
        assert all_nodes == {"a", "b"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
