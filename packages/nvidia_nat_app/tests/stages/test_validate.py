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
"""Tests for ValidateStage: valid/invalid graph handling."""

import pytest

from nat_app.compiler.compilation_context import CompilationContext
from nat_app.compiler.errors import GraphValidationError
from nat_app.graph.types import Graph
from nat_app.stages.validate import ValidateStage


class TestValidateStage:

    def test_name(self):
        stage = ValidateStage()
        assert stage.name == "validate"

    def test_valid_graph_passes(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        g.entry_point = "a"
        ctx = CompilationContext(compiled=None, metadata={"graph": g})
        stage = ValidateStage()
        result = stage.apply(ctx)
        assert result is ctx

    def test_invalid_graph_raises(self):
        g = Graph()
        g.add_edge("a", "b")  # nodes don't exist
        ctx = CompilationContext(compiled=None, metadata={"graph": g})
        stage = ValidateStage()
        with pytest.raises(GraphValidationError):
            stage.apply(ctx)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
