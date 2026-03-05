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
"""Tests for LLMAnalysisStage."""

import pytest

from nat_app.compiler.compilation_context import CompilationContext
from nat_app.graph.adapter import AbstractFrameworkAdapter
from nat_app.graph.llm_detection import LLMCallInfo
from nat_app.graph.types import Graph
from nat_app.stages.llm_analysis import LLMAnalysisStage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeLLM:

    def invoke(self, prompt: str) -> str:
        return "response"

    def ainvoke(self, prompt: str) -> str:
        return "response"


class _MockDetector:

    @property
    def invocation_methods(self) -> frozenset[str]:
        return frozenset({"invoke", "ainvoke"})

    def is_llm(self, obj) -> bool:
        return isinstance(obj, _FakeLLM)


class _AdapterWithDetector(AbstractFrameworkAdapter):

    def extract(self, source):
        return Graph()

    def build(self, original, result):
        return original

    def get_llm_detector(self):
        return _MockDetector()


class _AdapterNoDetector(AbstractFrameworkAdapter):

    def extract(self, source):
        return Graph()

    def build(self, original, result):
        return original


def _make_context(node_funcs: dict) -> CompilationContext:
    ctx = CompilationContext(compiled=None)
    g = Graph()
    for name in node_funcs:
        g.add_node(name)
    g.entry_point = next(iter(node_funcs)) if node_funcs else ""
    ctx.metadata["graph"] = g
    ctx.metadata["node_funcs"] = node_funcs
    return ctx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLLMAnalysisStageNoDetector:

    def test_no_detector_writes_empty_dict(self):
        adapter = _AdapterNoDetector()
        stage = LLMAnalysisStage(adapter)
        ctx = _make_context({"a": lambda s: s})

        result = stage.apply(ctx)

        assert result.metadata["llm_analysis"] == {}

    def test_stage_name(self):
        adapter = _AdapterNoDetector()
        stage = LLMAnalysisStage(adapter)
        assert stage.name == "llm_analysis"


class TestLLMAnalysisStageWithDetector:

    def test_detects_llm_calls(self):
        llm = _FakeLLM()

        def node_with_llm(state):
            return llm.invoke("hi")

        def node_without_llm(state):
            return state

        adapter = _AdapterWithDetector()
        stage = LLMAnalysisStage(adapter)
        ctx = _make_context({
            "with_llm": node_with_llm,
            "without_llm": node_without_llm,
        })

        result = stage.apply(ctx)
        analysis = result.metadata["llm_analysis"]

        assert "with_llm" in analysis
        assert "without_llm" in analysis
        assert analysis["with_llm"].call_count == 1
        assert analysis["without_llm"].call_count == 0

    def test_non_callable_gets_zero(self):
        adapter = _AdapterWithDetector()
        stage = LLMAnalysisStage(adapter)
        ctx = _make_context({"broken": "not_a_function"})

        result = stage.apply(ctx)
        analysis = result.metadata["llm_analysis"]
        assert analysis["broken"].call_count == 0

    def test_empty_node_funcs(self):
        adapter = _AdapterWithDetector()
        stage = LLMAnalysisStage(adapter)
        ctx = CompilationContext(compiled=None)
        ctx.metadata["graph"] = Graph()
        ctx.metadata["node_funcs"] = {}

        result = stage.apply(ctx)
        assert result.metadata["llm_analysis"] == {}

    def test_results_are_llm_call_info_instances(self):
        llm = _FakeLLM()

        def node_func(state):
            return llm.invoke("hi")

        adapter = _AdapterWithDetector()
        stage = LLMAnalysisStage(adapter)
        ctx = _make_context({"node": node_func})

        result = stage.apply(ctx)
        info = result.metadata["llm_analysis"]["node"]
        assert isinstance(info, LLMCallInfo)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
