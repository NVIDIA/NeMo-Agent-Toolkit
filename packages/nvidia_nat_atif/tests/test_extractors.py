# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the pluggable payload extractor system.

The converter delegates ``event.data`` parsing to extractors registered in
:mod:`nat.atof.extractors`, keyed on the producer-declared ``data_schema``.
This file covers the default extractors, custom registrations, and
end-to-end integration with :func:`convert`.

Runnable either via ``pytest`` or as a script:
    uv run pytest packages/nvidia_nat_atif/tests/test_extractors.py
    uv run python packages/nvidia_nat_atif/tests/test_extractors.py
"""

from __future__ import annotations

from typing import Any

import pytest

from nat.atof import MarkEvent
from nat.atof import ScopeEvent
from nat.atof.extractors import DEFAULT_LLM_EXTRACTOR
from nat.atof.extractors import DEFAULT_MARK_EXTRACTOR
from nat.atof.extractors import DEFAULT_TOOL_EXTRACTOR
from nat.atof.extractors import GenericToolResultExtractor
from nat.atof.extractors import LLM_EXTRACTOR_REGISTRY
from nat.atof.extractors import LlmPayloadExtractor
from nat.atof.extractors import MARK_EXTRACTOR_REGISTRY
from nat.atof.extractors import MarkPayloadExtractor
from nat.atof.extractors import NatRoleMarkExtractor
from nat.atof.extractors import OpenAiChatCompletionsLlmExtractor
from nat.atof.extractors import TOOL_EXTRACTOR_REGISTRY
from nat.atof.extractors import ToolPayloadExtractor
from nat.atof.extractors import register_llm_extractor
from nat.atof.extractors import register_mark_extractor
from nat.atof.extractors import register_tool_extractor
from nat.atof.extractors import resolve_llm_extractor
from nat.atof.extractors import resolve_mark_extractor
from nat.atof.extractors import resolve_tool_extractor
from nat.atof.scripts.atof_to_atif_converter import convert


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_default_llm_extractor_satisfies_protocol() -> None:
    assert isinstance(DEFAULT_LLM_EXTRACTOR, LlmPayloadExtractor)
    assert isinstance(OpenAiChatCompletionsLlmExtractor(), LlmPayloadExtractor)


def test_default_tool_extractor_satisfies_protocol() -> None:
    assert isinstance(DEFAULT_TOOL_EXTRACTOR, ToolPayloadExtractor)
    assert isinstance(GenericToolResultExtractor(), ToolPayloadExtractor)


def test_default_mark_extractor_satisfies_protocol() -> None:
    assert isinstance(DEFAULT_MARK_EXTRACTOR, MarkPayloadExtractor)
    assert isinstance(NatRoleMarkExtractor(), MarkPayloadExtractor)


# ---------------------------------------------------------------------------
# OpenAI LLM extractor unit tests
# ---------------------------------------------------------------------------


def test_openai_extract_input_messages_flat() -> None:
    messages = DEFAULT_LLM_EXTRACTOR.extract_input_messages(
        {"messages": [{"role": "user", "content": "hi"}]},
    )
    assert messages == [{"role": "user", "content": "hi"}]


def test_openai_extract_input_messages_nested_content() -> None:
    messages = DEFAULT_LLM_EXTRACTOR.extract_input_messages(
        {"content": {"messages": [{"role": "user", "content": "hi"}]}},
    )
    assert messages == [{"role": "user", "content": "hi"}]


def test_openai_extract_input_messages_empty_returns_empty() -> None:
    assert DEFAULT_LLM_EXTRACTOR.extract_input_messages({}) == []
    assert DEFAULT_LLM_EXTRACTOR.extract_input_messages(None) == []


def test_openai_extract_output_text_direct() -> None:
    assert DEFAULT_LLM_EXTRACTOR.extract_output_text({"content": "hello"}) == "hello"


def test_openai_extract_output_text_choices() -> None:
    assert (
        DEFAULT_LLM_EXTRACTOR.extract_output_text(
            {"choices": [{"message": {"content": "hello", "role": "assistant"}}]},
        )
        == "hello"
    )


def test_openai_extract_output_text_missing_returns_empty() -> None:
    assert DEFAULT_LLM_EXTRACTOR.extract_output_text({}) == ""
    assert DEFAULT_LLM_EXTRACTOR.extract_output_text({"foo": "bar"}) == ""


def test_openai_extract_tool_calls_flat_form() -> None:
    tool_calls = DEFAULT_LLM_EXTRACTOR.extract_tool_calls(
        {"tool_calls": [{"id": "c1", "name": "add", "arguments": {"a": 1}}]},
    )
    assert tool_calls == [
        {"tool_call_id": "c1", "function_name": "add", "arguments": {"a": 1}},
    ]


def test_openai_extract_tool_calls_nested_function_form() -> None:
    """OpenAI's actual API uses ``{id, function: {name, arguments}}``."""
    tool_calls = DEFAULT_LLM_EXTRACTOR.extract_tool_calls(
        {
            "tool_calls": [
                {"id": "c1", "function": {"name": "add", "arguments": '{"a": 1}'}},
            ],
        },
    )
    assert tool_calls == [
        {"tool_call_id": "c1", "function_name": "add", "arguments": {"a": 1}},
    ]


def test_openai_extract_tool_calls_handles_unparseable_string_arguments() -> None:
    tool_calls = DEFAULT_LLM_EXTRACTOR.extract_tool_calls(
        {"tool_calls": [{"id": "c1", "name": "foo", "arguments": "not-json"}]},
    )
    assert tool_calls == [
        {"tool_call_id": "c1", "function_name": "foo", "arguments": {"raw": "not-json"}},
    ]


# ---------------------------------------------------------------------------
# Generic tool extractor unit tests
# ---------------------------------------------------------------------------


def test_tool_extractor_unwraps_single_key_result() -> None:
    assert DEFAULT_TOOL_EXTRACTOR.extract_tool_result({"result": "7"}) == "7"
    assert DEFAULT_TOOL_EXTRACTOR.extract_tool_result({"output": 42}) == "42"


def test_tool_extractor_passes_through_none() -> None:
    assert DEFAULT_TOOL_EXTRACTOR.extract_tool_result(None) is None


def test_tool_extractor_serializes_dicts() -> None:
    assert DEFAULT_TOOL_EXTRACTOR.extract_tool_result({"a": 1, "b": 2}) == '{"a":1,"b":2}'


def test_tool_extractor_passes_through_string() -> None:
    assert DEFAULT_TOOL_EXTRACTOR.extract_tool_result("plain string") == "plain string"


# ---------------------------------------------------------------------------
# Mark extractor unit tests
# ---------------------------------------------------------------------------


def test_mark_extractor_lifts_valid_role() -> None:
    assert DEFAULT_MARK_EXTRACTOR.extract_role_and_content(
        {"role": "user", "content": "hi"},
    ) == ("user", "hi")


def test_mark_extractor_prefers_content_over_message() -> None:
    assert DEFAULT_MARK_EXTRACTOR.extract_role_and_content(
        {"role": "system", "content": "from content", "message": "from message"},
    ) == ("system", "from content")


def test_mark_extractor_falls_back_to_message_when_no_content() -> None:
    assert DEFAULT_MARK_EXTRACTOR.extract_role_and_content(
        {"role": "agent", "message": "hi"},
    ) == ("agent", "hi")


def test_mark_extractor_rejects_invalid_role() -> None:
    assert DEFAULT_MARK_EXTRACTOR.extract_role_and_content(
        {"role": "foo", "content": "x"},
    ) is None


def test_mark_extractor_rejects_non_dict() -> None:
    assert DEFAULT_MARK_EXTRACTOR.extract_role_and_content("plain-string") is None
    assert DEFAULT_MARK_EXTRACTOR.extract_role_and_content(None) is None


# ---------------------------------------------------------------------------
# Resolvers
# ---------------------------------------------------------------------------


def test_resolve_llm_extractor_returns_default_for_none_schema() -> None:
    assert resolve_llm_extractor(None) is DEFAULT_LLM_EXTRACTOR


def test_resolve_llm_extractor_returns_default_for_unregistered_schema() -> None:
    assert resolve_llm_extractor({"name": "acme/unknown", "version": "1"}) is DEFAULT_LLM_EXTRACTOR


def test_resolve_llm_extractor_returns_registered_extractor() -> None:
    assert (
        resolve_llm_extractor({"name": "openai/chat-completions", "version": "1"})
        is DEFAULT_LLM_EXTRACTOR
    )


def test_resolve_tool_extractor_always_returns_default_without_registration() -> None:
    assert resolve_tool_extractor(None) is DEFAULT_TOOL_EXTRACTOR
    assert resolve_tool_extractor({"name": "x", "version": "1"}) is DEFAULT_TOOL_EXTRACTOR


def test_resolve_mark_extractor_always_returns_default_without_registration() -> None:
    assert resolve_mark_extractor(None) is DEFAULT_MARK_EXTRACTOR
    assert resolve_mark_extractor({"name": "x", "version": "1"}) is DEFAULT_MARK_EXTRACTOR


# ---------------------------------------------------------------------------
# Registration validation
# ---------------------------------------------------------------------------


class _FakeLlmExtractor:
    def extract_input_messages(self, data: Any) -> list[dict[str, Any]]:
        return []

    def extract_output_text(self, data: Any) -> str:
        return ""

    def extract_tool_calls(self, data: Any) -> list[dict[str, Any]]:
        return []


def test_register_llm_extractor_rejects_empty_key() -> None:
    with pytest.raises(ValueError):
        register_llm_extractor("", "1", _FakeLlmExtractor())
    with pytest.raises(ValueError):
        register_llm_extractor("x", "", _FakeLlmExtractor())


def test_register_llm_extractor_rejects_non_conforming_extractor() -> None:
    with pytest.raises(TypeError):
        register_llm_extractor("x", "1", object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# End-to-end: custom Anthropic-shaped extractor enables a new producer
# ---------------------------------------------------------------------------


class _AnthropicMessagesV1:
    """Minimal Anthropic-messages extractor for the integration test.

    Accepts:
    - Input: ``{"input": [{"role", "parts": [{"text"}...]}]}``
    - Output: ``{"output_blocks": [{"type": "text", "text": ...}]}``
    """

    def extract_input_messages(self, data: Any) -> list[dict[str, Any]]:
        if not isinstance(data, dict):
            return []
        items = data.get("input")
        if not isinstance(items, list):
            return []
        messages: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            parts = item.get("parts") or []
            text = ""
            if isinstance(parts, list):
                text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
            if role:
                messages.append({"role": role, "content": text})
        return messages

    def extract_output_text(self, data: Any) -> str:
        if not isinstance(data, dict):
            return ""
        blocks = data.get("output_blocks") or []
        if not isinstance(blocks, list):
            return ""
        return "".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")

    def extract_tool_calls(self, data: Any) -> list[dict[str, Any]]:
        return []


def test_registering_anthropic_extractor_enables_conversion() -> None:
    """A producer declaring ``anthropic/messages@1`` can be converted once
    a matching extractor is registered — without this test's registration
    the same payload would trigger :class:`ShapeMismatchError`."""
    ds = {"name": "anthropic/messages", "version": "1"}
    register_llm_extractor("anthropic/messages", "1", _AnthropicMessagesV1())
    try:
        events = [
            ScopeEvent(
                scope_category="start",
                uuid="root-001",
                parent_uuid=None,
                timestamp="2026-01-01T00:00:00Z",
                name="agent",
                category="agent",
                data={"input": "3 + 4?"},
            ),
            ScopeEvent(
                scope_category="start",
                uuid="llm-001",
                parent_uuid="root-001",
                timestamp="2026-01-01T00:00:01Z",
                name="claude",
                category="llm",
                category_profile={"model_name": "claude"},
                data={"input": [{"role": "user", "parts": [{"text": "3 + 4?"}]}]},
                data_schema=ds,
            ),
            ScopeEvent(
                scope_category="end",
                uuid="llm-001",
                parent_uuid="root-001",
                timestamp="2026-01-01T00:00:02Z",
                name="claude",
                category="llm",
                category_profile={"model_name": "claude"},
                data={"output_blocks": [{"type": "text", "text": "The answer is 7."}]},
                data_schema=ds,
            ),
            ScopeEvent(
                scope_category="end",
                uuid="root-001",
                parent_uuid=None,
                timestamp="2026-01-01T00:00:03Z",
                name="agent",
                category="agent",
                data={"response": "done"},
            ),
        ]
        trajectory = convert(events)
        sources = [s.source for s in trajectory.steps]
        assert "user" in sources, f"expected user turn lifted from extractor, got {sources}"
        agent_steps = [s for s in trajectory.steps if s.source == "agent"]
        assert any(s.message == "The answer is 7." for s in agent_steps), (
            f"expected Anthropic output extracted into agent step; got {[s.message for s in agent_steps]}"
        )
    finally:
        LLM_EXTRACTOR_REGISTRY.pop(("anthropic/messages", "1"), None)


# ---------------------------------------------------------------------------
# End-to-end: custom tool extractor unwraps vendor-specific wrapper
# ---------------------------------------------------------------------------


class _MycoToolExtractor:
    """Unwraps ``{"data": {"payload": X}}`` — an acme convention."""

    def extract_tool_result(self, data: Any) -> str | None:
        if isinstance(data, dict):
            inner = data.get("data")
            if isinstance(inner, dict) and "payload" in inner:
                return str(inner["payload"])
        return DEFAULT_TOOL_EXTRACTOR.extract_tool_result(data)


def test_registering_tool_extractor_overrides_default() -> None:
    ds = {"name": "myco/tool-result", "version": "1"}
    register_tool_extractor("myco/tool-result", "1", _MycoToolExtractor())
    try:
        assert resolve_tool_extractor(ds).extract_tool_result(
            {"data": {"payload": "wrapped-answer"}},
        ) == "wrapped-answer"
        # Non-myco events still fall through to the default extractor.
        assert resolve_tool_extractor(None).extract_tool_result({"result": 7}) == "7"
    finally:
        TOOL_EXTRACTOR_REGISTRY.pop(("myco/tool-result", "1"), None)


# ---------------------------------------------------------------------------
# End-to-end: custom mark extractor lifts a different vendor convention
# ---------------------------------------------------------------------------


class _AcmeNotifyExtractor:
    """Lifts marks whose ``data.kind == "user-notify"`` as user steps."""

    def extract_role_and_content(self, data: Any) -> tuple[str, Any] | None:
        if isinstance(data, dict) and data.get("kind") == "user-notify":
            return "user", data.get("text", "")
        return None


def test_registering_mark_extractor_enables_custom_role_lift() -> None:
    ds = {"name": "acme/notify", "version": "1"}
    register_mark_extractor("acme/notify", "1", _AcmeNotifyExtractor())
    try:
        events = [
            ScopeEvent(
                scope_category="start",
                uuid="root-001",
                parent_uuid=None,
                timestamp="2026-01-01T00:00:00Z",
                name="agent",
                category="agent",
                data={"input": "go"},
            ),
            MarkEvent(
                uuid="mark-001",
                parent_uuid="root-001",
                timestamp="2026-01-01T00:00:01Z",
                name="note",
                data={"kind": "user-notify", "text": "please summarize"},
                data_schema=ds,
            ),
            ScopeEvent(
                scope_category="end",
                uuid="root-001",
                parent_uuid=None,
                timestamp="2026-01-01T00:00:02Z",
                name="agent",
                category="agent",
                data={"response": "done"},
            ),
        ]
        trajectory = convert(events)
        user_steps = [s for s in trajectory.steps if s.source == "user"]
        assert any(s.message == "please summarize" for s in user_steps), (
            f"expected custom mark lifted to user step; got {[(s.source, s.message) for s in trajectory.steps]}"
        )
    finally:
        MARK_EXTRACTOR_REGISTRY.pop(("acme/notify", "1"), None)


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
