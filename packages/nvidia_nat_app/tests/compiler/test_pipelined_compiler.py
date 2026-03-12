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
"""Tests for AbstractPipelinedCompiler: stage init, pipeline execution, hooks."""

import pytest

from nat_app.compiler.pipelined_compiler import AbstractPipelinedCompiler


class _TrackingStage:

    def __init__(self, name_val, key="visited"):
        self._name = name_val
        self._key = key

    @property
    def name(self):
        return self._name

    def apply(self, context, **kwargs):
        context.metadata.setdefault(self._key, []).append(self._name)
        return context


class _TestPipelinedCompiler(AbstractPipelinedCompiler):

    def default_stages(self):
        return [_TrackingStage("default_a"), _TrackingStage("default_b")]

    def prepare(self, source, **kwargs):
        return source


class TestStageInitialization:

    def test_uses_provided_stages(self):
        custom = [_TrackingStage("custom")]
        c = _TestPipelinedCompiler(stages=custom)
        assert len(c.stages) == 1
        assert c.stages[0].name == "custom"

    def test_uses_default_stages(self):
        c = _TestPipelinedCompiler()
        assert len(c.stages) == 2
        assert c.stages[0].name == "default_a"

    def test_stages_is_tuple(self):
        c = _TestPipelinedCompiler()
        assert isinstance(c.stages, tuple)


class TestCompilePipeline:

    def test_stages_run_in_order(self):
        c = _TestPipelinedCompiler()
        c.compile("source")
        ctx = c.last_context
        assert ctx.metadata["visited"] == ["default_a", "default_b"]

    def test_last_context_stored(self):
        c = _TestPipelinedCompiler()
        assert c.last_context is None
        c.compile("source")
        assert c.last_context is not None

    def test_prepare_called(self):
        c = _TestPipelinedCompiler()
        c.compile("my_source")
        assert c.last_context.compiled == "my_source"

    def test_finalize_returns_compiled(self):
        c = _TestPipelinedCompiler()
        result = c.compile("src")
        assert result == "src"


class TestSeedContext:

    def test_seed_context_called(self):

        class _SeedingCompiler(_TestPipelinedCompiler):

            def seed_context(self, context):
                context.metadata["seeded"] = True

        c = _SeedingCompiler()
        c.compile("x")
        assert c.last_context.metadata.get("seeded") is True


class TestFinalize:

    def test_finalize_override(self):

        class _CustomFinalize(_TestPipelinedCompiler):

            def finalize(self, context, **kwargs):
                return "custom_result"

        c = _CustomFinalize()
        result = c.compile("x")
        assert result == "custom_result"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
