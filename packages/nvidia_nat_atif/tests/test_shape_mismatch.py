# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ShapeMismatchError tests for the ATOF→ATIF converter.

The reference extractors assume an OpenAI chat-completions shape inside
``event.data``. Any producer that deviates would have its payload silently
dropped; :class:`ShapeMismatchError` converts that into a hard failure.

Runnable either via ``pytest`` or as a script:
    uv run pytest packages/nvidia_nat_atif/tests/test_shape_mismatch.py
    uv run python packages/nvidia_nat_atif/tests/test_shape_mismatch.py
"""

from __future__ import annotations

import pytest

from nat.atof import ScopeEvent
from nat.atof.scripts.atof_to_atif_converter import ShapeMismatchError
from nat.atof.scripts.atof_to_atif_converter import convert


# ---------------------------------------------------------------------------
# Stream builders
# ---------------------------------------------------------------------------


def _openai_shaped_stream() -> list:
    """Well-formed stream that matches the reference extractors."""
    return [
        ScopeEvent(
            scope_category="start",
            uuid="root-001",
            parent_uuid=None,
            timestamp="2026-01-01T00:00:00Z",
            name="calc-agent",
            category="agent",
            data={"input": "3 + 4?"},
        ),
        ScopeEvent(
            scope_category="start",
            uuid="llm-001",
            parent_uuid="root-001",
            timestamp="2026-01-01T00:00:01Z",
            name="gpt-4.1",
            category="llm",
            category_profile={"model_name": "gpt-4.1"},
            data={"messages": [{"role": "user", "content": "3 + 4?"}]},
            data_schema={"name": "openai/chat-completions", "version": "1"},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="llm-001",
            parent_uuid="root-001",
            timestamp="2026-01-01T00:00:02Z",
            name="gpt-4.1",
            category="llm",
            category_profile={"model_name": "gpt-4.1"},
            data={"content": "The answer is 7."},
            data_schema={"name": "openai/chat-completions", "version": "1"},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="root-001",
            parent_uuid=None,
            timestamp="2026-01-01T00:00:03Z",
            name="calc-agent",
            category="agent",
            data={"response": "The answer is 7."},
        ),
    ]


def _anthropic_input_stream() -> list:
    """LLM scope-start payload uses Anthropic ``input``/``system`` fields the
    reference extractor does not understand. Conversion must raise on the
    scope-start event.
    """
    return [
        ScopeEvent(
            scope_category="start",
            uuid="root-001",
            parent_uuid=None,
            timestamp="2026-01-01T00:00:00Z",
            name="agent",
            category="agent",
            data={"input": "go"},
        ),
        ScopeEvent(
            scope_category="start",
            uuid="llm-001",
            parent_uuid="root-001",
            timestamp="2026-01-01T00:00:01Z",
            name="claude",
            category="llm",
            category_profile={"model_name": "claude"},
            data={"system": "be helpful", "input": [{"role": "user", "parts": []}]},
            data_schema={"name": "anthropic/messages", "version": "1"},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="llm-001",
            parent_uuid="root-001",
            timestamp="2026-01-01T00:00:02Z",
            name="claude",
            category="llm",
            category_profile={"model_name": "claude"},
            data={"content": "done"},
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


def _anthropic_output_stream() -> list:
    """LLM scope-end payload uses Anthropic ``output_blocks`` — unknown to
    the extractor. ``data`` is non-empty but produces neither content nor
    tool_calls, so the whole assistant turn would be dropped.
    """
    return [
        ScopeEvent(
            scope_category="start",
            uuid="root-001",
            parent_uuid=None,
            timestamp="2026-01-01T00:00:00Z",
            name="agent",
            category="agent",
            data={"input": "go"},
        ),
        ScopeEvent(
            scope_category="start",
            uuid="llm-001",
            parent_uuid="root-001",
            timestamp="2026-01-01T00:00:01Z",
            name="claude",
            category="llm",
            category_profile={"model_name": "claude"},
            data={"messages": [{"role": "user", "content": "go"}]},
            data_schema={"name": "openai/chat-completions", "version": "1"},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="llm-001",
            parent_uuid="root-001",
            timestamp="2026-01-01T00:00:02Z",
            name="claude",
            category="llm",
            category_profile={"model_name": "claude"},
            data={"stop_reason": "end_turn", "output_blocks": [{"type": "text", "text": "done"}]},
            data_schema={"name": "anthropic/messages", "version": "1"},
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


def _tool_calls_only_stream() -> list:
    """LLM scope-end with ONLY ``tool_calls`` (no ``content``). This is a
    legitimate OpenAI-shape response: the assistant decided to call a tool
    and produced no text. Must NOT raise.
    """
    return [
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
            name="gpt",
            category="llm",
            category_profile={"model_name": "gpt"},
            data={"messages": [{"role": "user", "content": "3 + 4?"}]},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="llm-001",
            parent_uuid="root-001",
            timestamp="2026-01-01T00:00:02Z",
            name="gpt",
            category="llm",
            category_profile={"model_name": "gpt"},
            # No ``content`` key at all; only tool_calls.
            data={"tool_calls": [{"id": "call_1", "name": "add", "arguments": {"a": 3, "b": 4}}]},
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


def _tool_missing_call_id_stream() -> list:
    """Tool event without ``tool_call_id`` — not a data drop, just a
    correlation gap. Must NOT raise (the converter emits an observation
    with ``source_call_id=None``).
    """
    return [
        ScopeEvent(
            scope_category="start",
            uuid="root-001",
            parent_uuid=None,
            timestamp="2026-01-01T00:00:00Z",
            name="agent",
            category="agent",
            data={"input": "go"},
        ),
        ScopeEvent(
            scope_category="start",
            uuid="tool-001",
            parent_uuid="root-001",
            timestamp="2026-01-01T00:00:01Z",
            name="search",
            category="tool",
            category_profile=None,
            data={"query": "q"},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="tool-001",
            parent_uuid="root-001",
            timestamp="2026-01-01T00:00:02Z",
            name="search",
            category="tool",
            category_profile=None,
            data={"result": "answer"},
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


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_openai_shaped_stream_converts_without_error() -> None:
    trajectory = convert(_openai_shaped_stream())
    assert trajectory.steps


def test_tool_calls_only_response_does_not_raise() -> None:
    """Empty ``content`` with non-empty ``tool_calls`` is a legitimate
    assistant turn, not a shape mismatch."""
    trajectory = convert(_tool_calls_only_stream())
    assert trajectory.steps


def test_tool_missing_call_id_does_not_raise() -> None:
    """Tool events are not subject to shape-mismatch detection — their
    ``data`` is handled by the generic ``_serialize_tool_result`` path,
    which never returns empty on a non-empty dict."""
    trajectory = convert(_tool_missing_call_id_stream())
    assert trajectory.steps


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_llm_input_shape_mismatch_raises() -> None:
    with pytest.raises(ShapeMismatchError) as exc_info:
        convert(_anthropic_input_stream())

    exc = exc_info.value
    assert exc.kind == "llm_input"
    assert exc.uuid == "llm-001"
    assert exc.data_schema == {"name": "anthropic/messages", "version": "1"}
    assert set(exc.data_keys) == {"system", "input"}


def test_llm_output_shape_mismatch_raises() -> None:
    with pytest.raises(ShapeMismatchError) as exc_info:
        convert(_anthropic_output_stream())

    exc = exc_info.value
    assert exc.kind == "llm_output"
    assert exc.uuid == "llm-001"
    assert exc.data_schema == {"name": "anthropic/messages", "version": "1"}
    assert set(exc.data_keys) == {"stop_reason", "output_blocks"}


def test_error_message_mentions_uuid_and_keys() -> None:
    """The exception's string representation must carry enough context to
    debug the offending event without re-running the converter."""
    with pytest.raises(ShapeMismatchError) as exc_info:
        convert(_anthropic_output_stream())

    msg = str(exc_info.value)
    assert "llm-001" in msg
    assert "llm_output" in msg
    assert "output_blocks" in msg or "stop_reason" in msg


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
