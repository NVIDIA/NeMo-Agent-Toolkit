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
"""Tests for CompilationStage protocol conformance."""

import pytest

from nat_app.compiler.compilation_context import CompilationContext
from nat_app.compiler.compilation_stage import CompilationStage


class _ConformingStage:

    @property
    def name(self) -> str:
        return "test_stage"

    def apply(self, context, **kwargs):
        return context


class _MissingName:

    def apply(self, context, **kwargs):
        return context


class _MissingApply:

    @property
    def name(self) -> str:
        return "bad"


class TestProtocolConformance:

    def test_conforming_is_instance(self):
        stage = _ConformingStage()
        assert isinstance(stage, CompilationStage)

    def test_missing_name_not_instance(self):
        obj = _MissingName()
        assert not isinstance(obj, CompilationStage)

    def test_missing_apply_not_instance(self):
        obj = _MissingApply()
        assert not isinstance(obj, CompilationStage)

    def test_apply_returns_context(self):
        stage = _ConformingStage()
        ctx = CompilationContext(compiled="test")
        result = stage.apply(ctx)
        assert result is ctx

    def test_name_property(self):
        stage = _ConformingStage()
        assert stage.name == "test_stage"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
