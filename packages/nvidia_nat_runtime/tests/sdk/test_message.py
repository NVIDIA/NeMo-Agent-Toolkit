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
"""Tests for nat.sdk.llm.message — Message, ToolCall, TokenUsage, LLMResponse."""

from pydantic import ValidationError

from nat.sdk.llm.message import LLMResponse
from nat.sdk.llm.message import Message
from nat.sdk.llm.message import TokenUsage
from nat.sdk.llm.message import ToolCall

# ---------------------------------------------------------------------------
# ToolCall
# ---------------------------------------------------------------------------


class TestToolCall:

    def test_creation(self) -> None:
        tc = ToolCall(id="tc-1", name="search", arguments={"query": "test"})
        assert tc.id == "tc-1"
        assert tc.name == "search"
        assert tc.arguments == {"query": "test"}

    def test_default_arguments(self) -> None:
        tc = ToolCall(id="tc-1", name="noop")
        assert tc.arguments == {}

    def test_to_openai_dict(self) -> None:
        tc = ToolCall(id="tc-1", name="search", arguments={"query": "hello"})
        d = tc.to_openai_dict()
        assert d["id"] == "tc-1"
        assert d["type"] == "function"
        assert d["function"]["name"] == "search"
        assert '"query"' in d["function"]["arguments"]
        assert '"hello"' in d["function"]["arguments"]

    def test_frozen(self) -> None:
        tc = ToolCall(id="tc-1", name="test")
        try:
            tc.id = "new"  # type: ignore[misc]
            assert False, "Should have raised"
        except ValidationError:
            pass


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


class TestMessage:

    def test_user_message(self) -> None:
        m = Message(role="user", content="Hello")
        assert m.role == "user"
        assert m.content == "Hello"
        assert m.tool_calls == []
        assert m.tool_call_id is None

    def test_assistant_message_with_tool_calls(self) -> None:
        tc = ToolCall(id="tc-1", name="search", arguments={"q": "x"})
        m = Message(role="assistant", content="Searching...", tool_calls=[tc])
        assert len(m.tool_calls) == 1
        assert m.tool_calls[0].name == "search"

    def test_tool_result_message(self) -> None:
        m = Message(role="tool", content="result data", tool_call_id="tc-1")
        assert m.role == "tool"
        assert m.tool_call_id == "tc-1"

    def test_to_openai_dict_simple(self) -> None:
        m = Message(role="user", content="Hi")
        d = m.to_openai_dict()
        assert d == {"role": "user", "content": "Hi"}

    def test_to_openai_dict_with_tool_calls(self) -> None:
        tc = ToolCall(id="tc-1", name="add", arguments={"a": 1})
        m = Message(role="assistant", content="", tool_calls=[tc])
        d = m.to_openai_dict()
        assert "tool_calls" in d
        assert len(d["tool_calls"]) == 1
        assert d["tool_calls"][0]["function"]["name"] == "add"

    def test_to_openai_dict_tool_result(self) -> None:
        m = Message(role="tool", content="42", tool_call_id="tc-1")
        d = m.to_openai_dict()
        assert d["role"] == "tool"
        assert d["tool_call_id"] == "tc-1"
        assert d["content"] == "42"

    def test_to_openai_dict_no_extra_keys(self) -> None:
        m = Message(role="system", content="Be nice")
        d = m.to_openai_dict()
        assert set(d.keys()) == {"role", "content"}


# ---------------------------------------------------------------------------
# TokenUsage
# ---------------------------------------------------------------------------


class TestTokenUsage:

    def test_defaults(self) -> None:
        u = TokenUsage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0
        assert u.cached_tokens == 0
        assert u.reasoning_tokens == 0

    def test_custom_values(self) -> None:
        u = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cached_tokens=20,
            reasoning_tokens=10,
        )
        assert u.prompt_tokens == 100
        assert u.total_tokens == 150


# ---------------------------------------------------------------------------
# LLMResponse
# ---------------------------------------------------------------------------


class TestLLMResponse:

    def test_creation(self) -> None:
        msg = Message(role="assistant", content="Hi")
        resp = LLMResponse(message=msg)
        assert resp.message.content == "Hi"
        assert resp.usage is None
        assert resp.model is None
        assert resp.finish_reason is None
        assert resp.raw is None

    def test_with_usage(self) -> None:
        msg = Message(role="assistant", content="")
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        resp = LLMResponse(message=msg, usage=usage, model="gpt-4", finish_reason="stop")
        assert resp.usage.total_tokens == 15
        assert resp.model == "gpt-4"
        assert resp.finish_reason == "stop"
