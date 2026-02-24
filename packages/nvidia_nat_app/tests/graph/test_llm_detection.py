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
"""Tests for the generic LLM detection engine (discover + count)."""

import pytest

from nat_app.graph.llm_detection import LLMCallInfo
from nat_app.graph.llm_detection import count_llm_calls
from nat_app.graph.llm_detection import discover_llm_names

# ---------------------------------------------------------------------------
# Mock detector
# ---------------------------------------------------------------------------


class _FakeLLM:
    """Sentinel used to represent an LLM in tests."""

    def invoke(self, prompt: str) -> str:
        return "response"

    def ainvoke(self, prompt: str) -> str:
        return "response"

    def stream(self, prompt: str):
        return iter(["response"])


class _MockDetector:
    """Minimal LLMDetector for unit tests."""

    @property
    def invocation_methods(self) -> frozenset[str]:
        return frozenset({"invoke", "ainvoke", "stream"})

    def is_llm(self, obj) -> bool:
        return isinstance(obj, _FakeLLM)


_DETECTOR = _MockDetector()

# ---------------------------------------------------------------------------
# discover_llm_names
# ---------------------------------------------------------------------------


class TestDiscoverLLMNames:

    def test_closure_captured_llm(self):
        llm = _FakeLLM()

        def node_func(state):
            return llm.invoke("hi")

        found = discover_llm_names(node_func, _DETECTOR)
        assert "llm" in found
        assert found["llm"] is llm

    def test_global_llm(self):
        """Functions referencing module-level LLMs should be detected."""
        found = discover_llm_names(_func_using_global_llm, _DETECTOR)
        assert "GLOBAL_LLM" in found

    def test_self_attribute_llm(self):

        class Agent:

            def __init__(self):
                self.llm = _FakeLLM()

            def run(self, state):
                return self.llm.invoke("hi")

        agent = Agent()
        found = discover_llm_names(agent.run, _DETECTOR)
        assert "self.llm" in found

    def test_dict_registry(self):
        registry = {"main": _FakeLLM(), "backup": _FakeLLM()}

        def node_func(state):
            return registry["main"].invoke("hi")

        found = discover_llm_names(node_func, _DETECTOR)
        assert "registry" in found

    def test_list_container(self):
        llms = [_FakeLLM(), _FakeLLM()]

        def node_func(state):
            return llms[0].invoke("hi")

        found = discover_llm_names(node_func, _DETECTOR)
        assert "llms" in found

    def test_nested_object_attribute(self):

        class Config:
            pass

        cfg = Config()
        cfg.llm = _FakeLLM()

        def node_func(state):
            return cfg.llm.invoke("hi")

        found = discover_llm_names(node_func, _DETECTOR)
        assert "cfg.llm" in found

    def test_no_llm_returns_empty(self):
        x = 42

        def node_func(state):
            return x + 1

        found = discover_llm_names(node_func, _DETECTOR)
        assert found == {}

    def test_non_callable_returns_empty(self):
        found = discover_llm_names(42, _DETECTOR)  # type: ignore[arg-type]
        assert found == {}


# Module-level LLM for test_global_llm
GLOBAL_LLM = _FakeLLM()


def _func_using_global_llm(state):
    return GLOBAL_LLM.invoke("hello")


# ---------------------------------------------------------------------------
# count_llm_calls
# ---------------------------------------------------------------------------


class TestCountLLMCalls:

    def test_single_call(self):
        llm = _FakeLLM()

        def node_func(state):
            return llm.invoke("hi")

        result = count_llm_calls(node_func, _DETECTOR)
        assert result.call_count == 1
        assert "llm" in result.llm_names
        assert result.confidence == "full"

    def test_multiple_calls_sequential(self):
        llm = _FakeLLM()

        def node_func(state):
            a = llm.invoke("first")
            b = llm.invoke("second")
            return a + b

        result = count_llm_calls(node_func, _DETECTOR)
        assert result.call_count == 2

    def test_if_else_takes_max(self):
        llm = _FakeLLM()

        def node_func(state):
            if state.get("flag"):
                a = llm.invoke("a")
                b = llm.invoke("b")
                return a + b
            else:
                return llm.invoke("c")

        result = count_llm_calls(node_func, _DETECTOR)
        assert result.call_count == 2  # max(2, 1) = 2

    def test_loop_multiplier(self):
        llm = _FakeLLM()

        def node_func(state):
            results = []
            for item in state["items"]:
                results.append(llm.invoke(item))
            return results

        result = count_llm_calls(node_func, _DETECTOR)
        assert result.call_count == 3  # 1 * default_loop_multiplier(3)

    def test_no_llm_returns_zero(self):

        def node_func(state):
            return state["key"]

        result = count_llm_calls(node_func, _DETECTOR)
        assert result.call_count == 0
        assert result.llm_names == frozenset()
        assert result.confidence == "full"

    def test_self_attribute_calls(self):

        class Agent:

            def __init__(self):
                self.llm = _FakeLLM()

            def run(self, state):
                return self.llm.invoke("hi")

        agent = Agent()
        result = count_llm_calls(agent.run, _DETECTOR)
        assert result.call_count == 1
        assert "self.llm" in result.llm_names

    def test_ainvoke_counted(self):
        llm = _FakeLLM()

        async def node_func(state):
            return await llm.ainvoke("hi")

        result = count_llm_calls(node_func, _DETECTOR)
        assert result.call_count == 1

    def test_llm_call_info_defaults(self):
        info = LLMCallInfo()
        assert info.call_count == 0
        assert info.llm_names == frozenset()
        assert info.confidence == "full"
        assert info.warnings == []

    def test_nested_if_else_takes_max(self):
        llm = _FakeLLM()

        def node_func(state):
            if state.get("a"):
                if state.get("b"):
                    return llm.invoke("x")
                else:
                    return llm.invoke("y") + llm.invoke("z")
            else:
                return llm.invoke("w")

        result = count_llm_calls(node_func, _DETECTOR)
        assert result.call_count == 2

    def test_try_except_takes_max(self):
        llm = _FakeLLM()

        def node_func(state):
            try:
                return llm.invoke("a")
            except Exception:
                return llm.invoke("b")

        result = count_llm_calls(node_func, _DETECTOR)
        assert result.call_count == 2  # worst case: body + handler

    def test_match_includes_subject_and_guard(self):
        llm = _FakeLLM()

        def node_func(state):
            match llm.invoke("subject"):
                case x if llm.invoke("guard"):
                    return x
                case _:
                    return "default"

        result = count_llm_calls(node_func, _DETECTOR)
        # subject (1) + max(guard+body, default body) = 1 + (1+0) or (0+0) = 2
        assert result.call_count >= 2

    def test_with_includes_context_expr(self):
        llm = _FakeLLM()

        def node_func(state):
            with llm.stream("hi"):
                return "done"

        result = count_llm_calls(node_func, _DETECTOR)
        assert result.call_count >= 1  # context_expr (stream) + body

    def test_loop_includes_iter(self):
        llm = _FakeLLM()

        def node_func(state):
            yield from llm.stream(state["items"])

        result = count_llm_calls(node_func, _DETECTOR)
        # iter (1) + body (0) * multiplier = 1
        assert result.call_count >= 1

    def test_dynamic_receiver_sets_partial_confidence(self):
        llm = _FakeLLM()

        def node_func(state):
            # llm in closure so we run the counter; state["x"] is Subscript, not resolvable
            _ = llm.invoke("a")  # known LLM call
            return state["llm"].invoke("hi")  # dynamic receiver

        result = count_llm_calls(node_func, _DETECTOR)
        assert result.confidence == "partial"
        assert any("dynamic" in w.lower() for w in result.warnings)

    def test_llm_call_info_with_values(self):
        info = LLMCallInfo(
            call_count=3,
            llm_names=frozenset({"llm", "backup"}),
            confidence="partial",
            warnings=["Some targets resolved dynamically"],
        )
        assert info.call_count == 3
        assert info.llm_names == frozenset({"llm", "backup"})
        assert info.confidence == "partial"
        assert len(info.warnings) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
