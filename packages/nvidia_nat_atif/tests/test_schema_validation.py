# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Schema-driven extractor matrix: verifies the ATOF→ATIF converter
handles three LLM payload schemas (OpenAI, Anthropic, Gemini) across
three scenarios (simple, nested-with-tool, multi-turn) — plus a
heterogeneous-stream end-to-end test that loads EXMP-06 and confirms
per-event dispatch routes to the correct extractor.

This file is the evidence layer for Phase 10: the converter dispatches
on ``event.data_schema`` per event. The schema map architecture in
:mod:`nat.atof.extractors` lets a single ``SchemaMapLlmExtractor`` engine
serve all three providers via declarative paths + three optional hooks.
The matrix below proves the engine produces equivalent ATIF output for
each provider on the same scenario semantics.

Runnable either via ``pytest`` or as a script:
    uv run pytest packages/nvidia_nat_atif/tests/test_schema_validation.py
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from nat.atof import LLM_EXTRACTOR_REGISTRY
from nat.atof import SCHEMA_REGISTRY
from nat.atof import Event
from nat.atof import ScopeEvent
from nat.atof import register_anthropic_messages_v1
from nat.atof import register_gemini_generate_content_v1
from nat.atof.scripts.atof_to_atif_converter import convert

# ---------------------------------------------------------------------------
# Per-provider payload factories
# ---------------------------------------------------------------------------


class _PayloadFactory:
    """Base contract for shape-specific LLM payload construction.

    Each provider's factory builds the same canonical inputs/outputs into
    its native wire shape so the scenario builders below are
    provider-agnostic. Methods accept ATIF-shape data
    (``[{"role", "content"}]``, plain strings, dicts) and emit the
    provider's native ``data`` payload.
    """

    schema: dict[str, str]

    def llm_input(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        raise NotImplementedError

    def llm_output_text(self, text: str) -> dict[str, Any]:
        raise NotImplementedError

    def llm_output_tool_call(
        self,
        tool_id: str,
        name: str,
        args: dict[str, Any],
        prefix_text: str = "",
    ) -> dict[str, Any]:
        raise NotImplementedError

    def llm_input_with_tool_result(
        self,
        prior_user_msg: str,
        tool_id: str,
        name: str,
        args: dict[str, Any],
        result: str,
    ) -> dict[str, Any]:
        """Build the round-2 LLM input including the prior assistant turn
        (with tool_use) and the tool result echo. Each provider has its
        own transport for tool results — this method encodes the
        provider-correct shape so the extractor's input hook (and the
        converter's role-filter) correctly skip the echoed turns,
        leaving only the original user message (already deduped).
        """
        raise NotImplementedError


class _OpenAiFactory(_PayloadFactory):
    schema = {"name": "openai/chat-completions", "version": "1"}

    def llm_input(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return {"messages": list(messages)}

    def llm_output_text(self, text: str) -> dict[str, Any]:
        return {"content": text}

    def llm_output_tool_call(
        self,
        tool_id: str,
        name: str,
        args: dict[str, Any],
        prefix_text: str = "",
    ) -> dict[str, Any]:
        return {
            "content": prefix_text,
            "tool_calls": [{
                "id": tool_id, "name": name, "arguments": args
            }],
        }

    def llm_input_with_tool_result(
        self,
        prior_user_msg: str,
        tool_id: str,
        name: str,
        args: dict[str, Any],
        result: str,
    ) -> dict[str, Any]:
        return {
            "messages": [
                {
                    "role": "user", "content": prior_user_msg
                },
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": tool_id, "name": name, "arguments": args
                    }],
                },
                {
                    "role": "tool", "tool_call_id": tool_id, "content": result
                },
            ],
        }


class _AnthropicFactory(_PayloadFactory):
    schema = {"name": "anthropic/messages", "version": "1"}

    def llm_input(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "model": "claude-3-5-sonnet-20241022",
            "messages": list(messages),
        }

    def llm_output_text(self, text: str) -> dict[str, Any]:
        return {
            "id": "msg_test",
            "role": "assistant",
            "content": [{
                "type": "text", "text": text
            }],
            "stop_reason": "end_turn",
        }

    def llm_output_tool_call(
        self,
        tool_id: str,
        name: str,
        args: dict[str, Any],
        prefix_text: str = "",
    ) -> dict[str, Any]:
        content_blocks: list[dict[str, Any]] = []
        if prefix_text:
            content_blocks.append({"type": "text", "text": prefix_text})
        content_blocks.append({
            "type": "tool_use",
            "id": tool_id,
            "name": name,
            "input": args,
        })
        return {
            "id": "msg_test",
            "role": "assistant",
            "content": content_blocks,
            "stop_reason": "tool_use",
        }

    def llm_input_with_tool_result(
        self,
        prior_user_msg: str,
        tool_id: str,
        name: str,
        args: dict[str, Any],
        result: str,
    ) -> dict[str, Any]:
        return {
            "model":
                "claude-3-5-sonnet-20241022",
            "messages": [
                {
                    "role": "user", "content": prior_user_msg
                },
                {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use", "id": tool_id, "name": name, "input": args
                    }],
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result", "tool_use_id": tool_id, "content": result
                    }],
                },
            ],
        }


class _GeminiFactory(_PayloadFactory):
    schema = {"name": "gemini/generate-content", "version": "1"}

    def _to_gemini_role(self, role: str) -> str:
        # Gemini uses "model" where OpenAI/Anthropic use "assistant".
        return "model" if role == "assistant" else role

    def llm_input(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        contents = [{
            "role": self._to_gemini_role(m["role"]),
            "parts": [{
                "text": m["content"]
            }],
        } for m in messages]
        return {"contents": contents}

    def llm_output_text(self, text: str) -> dict[str, Any]:
        return {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "text": text
                    }],
                },
                "finishReason": "STOP",
            }, ],
        }

    def llm_output_tool_call(
        self,
        tool_id: str,
        name: str,
        args: dict[str, Any],
        prefix_text: str = "",
    ) -> dict[str, Any]:
        parts: list[dict[str, Any]] = []
        if prefix_text:
            parts.append({"text": prefix_text})
        parts.append({"functionCall": {"name": name, "args": args}})
        return {
            "candidates": [{
                "content": {
                    "role": "model", "parts": parts
                },
                "finishReason": "STOP",
            }, ],
        }

    def llm_input_with_tool_result(
        self,
        prior_user_msg: str,
        tool_id: str,
        name: str,
        args: dict[str, Any],
        result: str,
    ) -> dict[str, Any]:
        # Gemini uses "model" for assistant turns and bundles tool I/O
        # into typed parts (functionCall/functionResponse). The Gemini
        # input hook drops both echoed turns (no text → no message
        # surfaces back to the converter).
        return {
            "contents": [
                {
                    "role": "user", "parts": [{
                        "text": prior_user_msg
                    }]
                },
                {
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "name": name, "args": args
                        }
                    }],
                },
                {
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": name, "response": {
                                "result": result
                            }
                        },
                    }, ],
                },
            ],
        }


_FACTORIES = {
    "openai": _OpenAiFactory(),
    "anthropic": _AnthropicFactory(),
    "gemini": _GeminiFactory(),
}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=list(_FACTORIES.keys()))
def factory(request: pytest.FixtureRequest) -> _PayloadFactory:
    """Parametrize tests across all three providers."""
    return _FACTORIES[request.param]


@pytest.fixture
def opt_in_extractors() -> Iterator[None]:
    """Register Anthropic + Gemini extractors and JSON Schemas, then
    clean up the global registries afterwards. Tests using non-OpenAI
    schemas MUST request this fixture so registration is scoped to
    the test (avoids leakage across the suite)."""
    register_anthropic_messages_v1()
    register_gemini_generate_content_v1()
    try:
        yield
    finally:
        LLM_EXTRACTOR_REGISTRY.pop(("anthropic/messages", "1"), None)
        LLM_EXTRACTOR_REGISTRY.pop(("gemini/generate-content", "1"), None)
        SCHEMA_REGISTRY.pop(("anthropic/messages", "1"), None)
        SCHEMA_REGISTRY.pop(("gemini/generate-content", "1"), None)


def _ts(second: int) -> str:
    """Deterministic RFC 3339 timestamp helper."""
    return f"2026-04-30T00:00:{second:02d}Z"


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------


def _build_simple(factory: _PayloadFactory) -> list[Event]:
    """Scenario: user asks a question, LLM responds in plain text. No tools.

    Expected ATIF: 2 steps (user query, agent reply).
    """
    user_msg = "What's the capital of France?"
    agent_msg = "Paris."
    return [
        ScopeEvent(
            scope_category="start",
            uuid="agent-s",
            parent_uuid=None,
            timestamp=_ts(0),
            name="test_agent",
            attributes=[],
            category="agent",
            data={"input": user_msg},
        ),
        ScopeEvent(
            scope_category="start",
            uuid="llm-s",
            parent_uuid="agent-s",
            timestamp=_ts(1),
            name="test_llm",
            attributes=[],
            category="llm",
            category_profile={"model_name": "test"},
            data=factory.llm_input([{
                "role": "user", "content": user_msg
            }]),
            data_schema=factory.schema,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="llm-s",
            parent_uuid="agent-s",
            timestamp=_ts(2),
            name="test_llm",
            attributes=[],
            category="llm",
            category_profile={"model_name": "test"},
            data=factory.llm_output_text(agent_msg),
            data_schema=factory.schema,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="agent-s",
            parent_uuid=None,
            timestamp=_ts(3),
            name="test_agent",
            attributes=[],
            category="agent",
            data={"response": agent_msg},
        ),
    ]


def _build_nested(factory: _PayloadFactory) -> list[Event]:
    """Scenario: user asks, LLM calls a tool, tool returns, LLM answers.

    Two LLM calls + one tool. Expected ATIF: 3 steps (user query, agent
    with tool_call+observation, agent final reply).
    """
    user_msg = "What is 7 squared?"
    tool_id = "call_pow_1"
    # Gemini synthesizes the tool_call_id as ``name__index`` (no vendor
    # ID supplied). The tool scope's category_profile.tool_call_id must
    # match what the LLM extractor produces, so for the Gemini case we
    # use the synthesized form. OpenAI/Anthropic preserve the explicit ID.
    if isinstance(factory, _GeminiFactory):
        effective_tool_id = "pow__0"
    else:
        effective_tool_id = tool_id
    tool_args = {"base": 7, "exp": 2}
    tool_result = "49"
    agent_final = "7 squared is 49."

    return [
        ScopeEvent(
            scope_category="start",
            uuid="agent-n",
            parent_uuid=None,
            timestamp=_ts(0),
            name="test_agent",
            attributes=[],
            category="agent",
            data={"input": user_msg},
        ),
        ScopeEvent(
            scope_category="start",
            uuid="llm-n-1",
            parent_uuid="agent-n",
            timestamp=_ts(1),
            name="test_llm",
            attributes=[],
            category="llm",
            category_profile={"model_name": "test"},
            data=factory.llm_input([{
                "role": "user", "content": user_msg
            }]),
            data_schema=factory.schema,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="llm-n-1",
            parent_uuid="agent-n",
            timestamp=_ts(2),
            name="test_llm",
            attributes=[],
            category="llm",
            category_profile={"model_name": "test"},
            data=factory.llm_output_tool_call(tool_id, "pow", tool_args),
            data_schema=factory.schema,
        ),
        ScopeEvent(
            scope_category="start",
            uuid="tool-n",
            parent_uuid="agent-n",
            timestamp=_ts(3),
            name="pow",
            attributes=[],
            category="tool",
            category_profile={"tool_call_id": effective_tool_id},
            data=tool_args,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="tool-n",
            parent_uuid="agent-n",
            timestamp=_ts(4),
            name="pow",
            attributes=[],
            category="tool",
            category_profile={"tool_call_id": effective_tool_id},
            data={"result": tool_result},
        ),
        ScopeEvent(
            scope_category="start",
            uuid="llm-n-2",
            parent_uuid="agent-n",
            timestamp=_ts(5),
            name="test_llm",
            attributes=[],
            category="llm",
            category_profile={"model_name": "test"},
            data=factory.llm_input_with_tool_result(
                user_msg,
                effective_tool_id,
                "pow",
                tool_args,
                tool_result,
            ),
            data_schema=factory.schema,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="llm-n-2",
            parent_uuid="agent-n",
            timestamp=_ts(6),
            name="test_llm",
            attributes=[],
            category="llm",
            category_profile={"model_name": "test"},
            data=factory.llm_output_text(agent_final),
            data_schema=factory.schema,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="agent-n",
            parent_uuid=None,
            timestamp=_ts(7),
            name="test_agent",
            attributes=[],
            category="agent",
            data={"response": agent_final},
        ),
    ]


def _build_multi_turn(factory: _PayloadFactory) -> list[Event]:
    """Scenario: two rounds of plain Q&A, no tools.

    Expected ATIF: 4 steps (user1, agent1, user2, agent2). The second
    LLM call's input includes the prior assistant turn — the extractor
    must NOT re-emit it as a user/system step (assistant role is
    skipped by the converter).
    """
    user1 = "Who wrote Pride and Prejudice?"
    agent1 = "Jane Austen wrote Pride and Prejudice."
    user2 = "When was it published?"
    agent2 = "It was published in 1813."

    return [
        ScopeEvent(
            scope_category="start",
            uuid="agent-m",
            parent_uuid=None,
            timestamp=_ts(0),
            name="test_agent",
            attributes=[],
            category="agent",
            data={"input": user1},
        ),
        ScopeEvent(
            scope_category="start",
            uuid="llm-m-1",
            parent_uuid="agent-m",
            timestamp=_ts(1),
            name="test_llm",
            attributes=[],
            category="llm",
            category_profile={"model_name": "test"},
            data=factory.llm_input([{
                "role": "user", "content": user1
            }]),
            data_schema=factory.schema,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="llm-m-1",
            parent_uuid="agent-m",
            timestamp=_ts(2),
            name="test_llm",
            attributes=[],
            category="llm",
            category_profile={"model_name": "test"},
            data=factory.llm_output_text(agent1),
            data_schema=factory.schema,
        ),
        ScopeEvent(
            scope_category="start",
            uuid="llm-m-2",
            parent_uuid="agent-m",
            timestamp=_ts(3),
            name="test_llm",
            attributes=[],
            category="llm",
            category_profile={"model_name": "test"},
            data=factory.llm_input([
                {
                    "role": "user", "content": user1
                },
                {
                    "role": "assistant", "content": agent1
                },
                {
                    "role": "user", "content": user2
                },
            ]),
            data_schema=factory.schema,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="llm-m-2",
            parent_uuid="agent-m",
            timestamp=_ts(4),
            name="test_llm",
            attributes=[],
            category="llm",
            category_profile={"model_name": "test"},
            data=factory.llm_output_text(agent2),
            data_schema=factory.schema,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="agent-m",
            parent_uuid=None,
            timestamp=_ts(5),
            name="test_agent",
            attributes=[],
            category="agent",
            data={"response": agent2},
        ),
    ]


# ---------------------------------------------------------------------------
# Matrix tests: each scenario × each provider
# ---------------------------------------------------------------------------


def test_simple_scenario(factory: _PayloadFactory, opt_in_extractors: None) -> None:
    """All three providers convert a plain Q&A turn into 2 ATIF steps:
    user query + agent reply with the expected text."""
    events = _build_simple(factory)
    trajectory = convert(events)

    sources = [s.source for s in trajectory.steps]
    assert sources == ["user", "agent"], f"{factory.schema['name']}: expected [user, agent], got {sources}"

    user_step, agent_step = trajectory.steps
    assert user_step.message == "What's the capital of France?"
    assert agent_step.message == "Paris."
    assert not agent_step.tool_calls


def test_nested_scenario(factory: _PayloadFactory, opt_in_extractors: None) -> None:
    """All three providers handle a tool round-trip: user query → agent
    with one tool_call and one observation → final agent text."""
    events = _build_nested(factory)
    trajectory = convert(events)

    sources = [s.source for s in trajectory.steps]
    assert sources == ["user", "agent",
                       "agent"], (f"{factory.schema['name']}: expected [user, agent, agent], got {sources}")

    user_step, agent_with_tool, agent_final = trajectory.steps
    assert user_step.message == "What is 7 squared?"

    # Mid-round agent step carries the tool_call and observation.
    assert agent_with_tool.tool_calls, f"{factory.schema['name']}: expected tool_calls on mid agent step"
    assert len(agent_with_tool.tool_calls) == 1, (
        f"{factory.schema['name']}: expected exactly 1 tool_call, got {len(agent_with_tool.tool_calls)}")
    tc = agent_with_tool.tool_calls[0]
    assert tc.function_name == "pow"
    assert tc.arguments == {"base": 7, "exp": 2}

    assert agent_with_tool.observation is not None, f"{factory.schema['name']}: expected observation on mid agent step"
    assert len(agent_with_tool.observation.results) == 1
    assert agent_with_tool.observation.results[0].content == "49"

    # Final round agent step has the answer text and no tool_calls.
    assert agent_final.message == "7 squared is 49."
    assert not agent_final.tool_calls


def test_multi_turn_scenario(factory: _PayloadFactory, opt_in_extractors: None) -> None:
    """All three providers preserve two distinct user turns across two
    LLM rounds. Output: 4 ATIF steps alternating user/agent."""
    events = _build_multi_turn(factory)
    trajectory = convert(events)

    sources = [s.source for s in trajectory.steps]
    assert sources == ["user", "agent", "user",
                       "agent"], (f"{factory.schema['name']}: expected [user, agent, user, agent], got {sources}")

    u1, a1, u2, a2 = trajectory.steps
    assert u1.message == "Who wrote Pride and Prejudice?"
    assert a1.message == "Jane Austen wrote Pride and Prejudice."
    assert u2.message == "When was it published?"
    assert a2.message == "It was published in 1813."


# ---------------------------------------------------------------------------
# Heterogeneous-stream test: one trajectory exercises all three extractors
# ---------------------------------------------------------------------------


def test_heterogeneous_stream_dispatches_per_event(opt_in_extractors: None) -> None:
    """Reproduce EXMP-06 inline (orchestrator routes to OpenAI, Anthropic,
    Gemini in one stream) and assert the converter dispatches per-event:
    every LLM scope-end emits one agent step with the provider-specific
    text, regardless of provider mix.

    This is the strongest end-to-end evidence that the schema-map
    architecture handles heterogeneous streams: per-event dispatch via
    ``event.data_schema``, no producer-side coordination, no per-stream
    schema lock.
    """
    # Inline replication of EXMP-06's three LLM rounds. Kept here (not
    # imported) so the test is self-contained — failures don't depend on
    # the example generator staying in sync.
    user_query = "Two things: (1) write a Python function for factorial, and (2) tell me what 2^32 equals."
    code_answer = "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
    math_answer = "2^32 = 4294967296"
    router_decision = "Plan: claude for code, gemini for math."

    events: list[Event] = [
        ScopeEvent(
            scope_category="start",
            uuid="orch",
            parent_uuid=None,
            timestamp=_ts(0),
            name="router",
            attributes=[],
            category="agent",
            data={"input": user_query},
        ),
        # OpenAI router
        ScopeEvent(
            scope_category="start",
            uuid="llm-r",
            parent_uuid="orch",
            timestamp=_ts(1),
            name="gpt-4o",
            attributes=[],
            category="llm",
            category_profile={"model_name": "gpt-4o"},
            data=_OpenAiFactory().llm_input([{
                "role": "user", "content": user_query
            }]),
            data_schema=_OpenAiFactory().schema,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="llm-r",
            parent_uuid="orch",
            timestamp=_ts(2),
            name="gpt-4o",
            attributes=[],
            category="llm",
            category_profile={"model_name": "gpt-4o"},
            data=_OpenAiFactory().llm_output_text(router_decision),
            data_schema=_OpenAiFactory().schema,
        ),
        # Anthropic code specialist
        ScopeEvent(
            scope_category="start",
            uuid="llm-c",
            parent_uuid="orch",
            timestamp=_ts(3),
            name="claude",
            attributes=[],
            category="llm",
            category_profile={"model_name": "claude-3-5-sonnet"},
            data=_AnthropicFactory().llm_input([{
                "role": "user", "content": "Write factorial"
            }]),
            data_schema=_AnthropicFactory().schema,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="llm-c",
            parent_uuid="orch",
            timestamp=_ts(4),
            name="claude",
            attributes=[],
            category="llm",
            category_profile={"model_name": "claude-3-5-sonnet"},
            data=_AnthropicFactory().llm_output_text(code_answer),
            data_schema=_AnthropicFactory().schema,
        ),
        # Gemini math specialist
        ScopeEvent(
            scope_category="start",
            uuid="llm-g",
            parent_uuid="orch",
            timestamp=_ts(5),
            name="gemini",
            attributes=[],
            category="llm",
            category_profile={"model_name": "gemini-2.0-flash"},
            data=_GeminiFactory().llm_input([{
                "role": "user", "content": "What is 2^32?"
            }]),
            data_schema=_GeminiFactory().schema,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="llm-g",
            parent_uuid="orch",
            timestamp=_ts(6),
            name="gemini",
            attributes=[],
            category="llm",
            category_profile={"model_name": "gemini-2.0-flash"},
            data=_GeminiFactory().llm_output_text(math_answer),
            data_schema=_GeminiFactory().schema,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="orch",
            parent_uuid=None,
            timestamp=_ts(7),
            name="router",
            attributes=[],
            category="agent",
            data={"response": "combined"},
        ),
    ]

    trajectory = convert(events)
    agent_messages = [s.message for s in trajectory.steps if s.source == "agent"]

    # The strongest invariant: every provider's response surfaces as an
    # agent step's message. This is only true if the converter dispatched
    # to the correct extractor for each event — wrong dispatch would
    # either drop content (different ShapeMismatchError) or smuggle the
    # wrong text in.
    assert router_decision in agent_messages, f"OpenAI router output missing — got {agent_messages}"
    assert code_answer in agent_messages, f"Anthropic code output missing — got {agent_messages}"
    assert math_answer in agent_messages, f"Gemini math output missing — got {agent_messages}"


# ---------------------------------------------------------------------------
# Regression: registration is idempotent
# ---------------------------------------------------------------------------


def test_register_anthropic_idempotent() -> None:
    """Calling ``register_anthropic_messages_v1()`` twice is safe. The
    second call overwrites the first registration with an equivalent
    extractor; no error raised."""
    register_anthropic_messages_v1()
    register_anthropic_messages_v1()
    try:
        assert ("anthropic/messages", "1") in LLM_EXTRACTOR_REGISTRY
        assert ("anthropic/messages", "1") in SCHEMA_REGISTRY
    finally:
        LLM_EXTRACTOR_REGISTRY.pop(("anthropic/messages", "1"), None)
        SCHEMA_REGISTRY.pop(("anthropic/messages", "1"), None)


def test_register_gemini_idempotent() -> None:
    """Calling ``register_gemini_generate_content_v1()`` twice is safe."""
    register_gemini_generate_content_v1()
    register_gemini_generate_content_v1()
    try:
        assert ("gemini/generate-content", "1") in LLM_EXTRACTOR_REGISTRY
        assert ("gemini/generate-content", "1") in SCHEMA_REGISTRY
    finally:
        LLM_EXTRACTOR_REGISTRY.pop(("gemini/generate-content", "1"), None)
        SCHEMA_REGISTRY.pop(("gemini/generate-content", "1"), None)


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
