# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``data_schema`` validation in the ATOF→ATIF converter.

When an event declares a ``data_schema`` registered in
:mod:`nat.atof.schemas`, the converter validates ``event.data`` against it
in a pre-pass and raises :class:`DataSchemaViolationError` on failure.
Unknown schemas log a ``WARNING`` and pass through; events without a
``data_schema`` skip validation entirely.

Runnable either via ``pytest`` or as a script:
    uv run pytest packages/nvidia_nat_atif/tests/test_data_schema_validation.py
    uv run python packages/nvidia_nat_atif/tests/test_data_schema_validation.py
"""

from __future__ import annotations

import logging

import pytest

from nat.atof import ScopeEvent
from nat.atof.schemas import SCHEMA_REGISTRY
from nat.atof.schemas import register_schema
from nat.atof.scripts.atof_to_atif_converter import DataSchemaViolationError
from nat.atof.scripts.atof_to_atif_converter import convert


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

OPENAI_DS = {"name": "openai/chat-completions", "version": "1"}


def _root_agent_start() -> ScopeEvent:
    return ScopeEvent(
        scope_category="start",
        uuid="root-001",
        parent_uuid=None,
        timestamp="2026-01-01T00:00:00Z",
        name="agent",
        category="agent",
        data={"input": "go"},
    )


def _root_agent_end() -> ScopeEvent:
    return ScopeEvent(
        scope_category="end",
        uuid="root-001",
        parent_uuid=None,
        timestamp="2026-01-01T00:00:03Z",
        name="agent",
        category="agent",
        data={"response": "done"},
    )


def _llm_start(*, data: dict, data_schema: dict | None = OPENAI_DS) -> ScopeEvent:
    return ScopeEvent(
        scope_category="start",
        uuid="llm-001",
        parent_uuid="root-001",
        timestamp="2026-01-01T00:00:01Z",
        name="gpt",
        category="llm",
        category_profile={"model_name": "gpt"},
        data=data,
        data_schema=data_schema,
    )


def _llm_end(*, data: dict, data_schema: dict | None = OPENAI_DS) -> ScopeEvent:
    return ScopeEvent(
        scope_category="end",
        uuid="llm-001",
        parent_uuid="root-001",
        timestamp="2026-01-01T00:00:02Z",
        name="gpt",
        category="llm",
        category_profile={"model_name": "gpt"},
        data=data,
        data_schema=data_schema,
    )


# ---------------------------------------------------------------------------
# Happy paths (valid payloads that declare the registered schema)
# ---------------------------------------------------------------------------


def test_openai_input_messages_passes_validation() -> None:
    events = [
        _root_agent_start(),
        _llm_start(data={"messages": [{"role": "user", "content": "hi"}]}),
        _llm_end(data={"content": "hello"}),
        _root_agent_end(),
    ]
    trajectory = convert(events)
    assert trajectory.steps


def test_openai_nested_content_messages_passes() -> None:
    """Input payload with ``content.messages`` nesting (the alternative shape
    ``_unwrap_llm_messages`` accepts)."""
    events = [
        _root_agent_start(),
        _llm_start(data={"content": {"messages": [{"role": "user", "content": "hi"}]}}),
        _llm_end(data={"content": "hello"}),
        _root_agent_end(),
    ]
    convert(events)


def test_openai_tool_calls_only_output_passes() -> None:
    """An assistant turn with only ``tool_calls`` (no ``content``) is a
    valid OpenAI response."""
    events = [
        _root_agent_start(),
        _llm_start(data={"messages": [{"role": "user", "content": "add 3 and 4"}]}),
        _llm_end(
            data={"tool_calls": [{"id": "c1", "name": "add", "arguments": {"a": 3, "b": 4}}]},
        ),
        _root_agent_end(),
    ]
    convert(events)


def test_openai_choices_output_passes() -> None:
    """Nested ``choices[0].message`` output shape passes validation."""
    events = [
        _root_agent_start(),
        _llm_start(data={"messages": [{"role": "user", "content": "hi"}]}),
        _llm_end(
            data={"choices": [{"message": {"content": "hello", "role": "assistant"}}]},
        ),
        _root_agent_end(),
    ]
    convert(events)


# ---------------------------------------------------------------------------
# Missing schema: validation is skipped (legacy producer behavior)
# ---------------------------------------------------------------------------


def test_missing_data_schema_skips_validation() -> None:
    """Events without ``data_schema`` are not validated (spec §2: field
    is optional). An Anthropic-style payload with no schema declaration
    still dies at the shape-mismatch guardrail, but not at this one."""
    events = [
        _root_agent_start(),
        _llm_start(data={"messages": [{"role": "user", "content": "hi"}]}, data_schema=None),
        _llm_end(data={"content": "hello"}, data_schema=None),
        _root_agent_end(),
    ]
    trajectory = convert(events)
    assert trajectory.steps


# ---------------------------------------------------------------------------
# Unknown schema: WARN, don't raise
# ---------------------------------------------------------------------------


def test_unknown_data_schema_logs_warning_and_skips(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If the producer declares a ``data_schema`` we haven't registered,
    validation is skipped with a warning — we cannot validate what we
    don't know about."""
    caplog.set_level(logging.WARNING, logger="nat.atof.scripts.atof_to_atif_converter")
    events = [
        _root_agent_start(),
        _llm_start(
            data={"messages": [{"role": "user", "content": "hi"}]},
            data_schema={"name": "acme/made-up", "version": "99"},
        ),
        _llm_end(
            data={"content": "hi"},
            data_schema={"name": "acme/made-up", "version": "99"},
        ),
        _root_agent_end(),
    ]
    convert(events)
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("acme/made-up" in m for m in messages)
    assert any("unregistered data_schema" in m for m in messages)


# ---------------------------------------------------------------------------
# Validation failures: raise DataSchemaViolationError
# ---------------------------------------------------------------------------


def test_empty_payload_declaring_openai_schema_raises() -> None:
    """An empty ``{}`` payload matches none of the required keys."""
    events = [
        _root_agent_start(),
        _llm_start(data={}),
        _llm_end(data={"content": "hi"}),
        _root_agent_end(),
    ]
    with pytest.raises(DataSchemaViolationError) as exc_info:
        convert(events)

    exc = exc_info.value
    assert exc.uuid == "llm-001"
    assert exc.data_schema == OPENAI_DS


def test_anthropic_shaped_payload_declaring_openai_schema_raises() -> None:
    """Anthropic ``input``/``system`` keys don't satisfy the OpenAI schema
    (which requires ``messages``, ``content``, ``tool_calls``, or ``choices``
    at top level)."""
    events = [
        _root_agent_start(),
        _llm_start(
            data={"system": "be helpful", "input": [{"role": "user", "parts": []}]},
        ),
        _llm_end(data={"content": "hi"}),
        _root_agent_end(),
    ]
    with pytest.raises(DataSchemaViolationError):
        convert(events)


def test_data_schema_violation_error_carries_context() -> None:
    """The exception must expose uuid, declared schema, path, and message
    for debugging without re-running the converter."""
    events = [
        _root_agent_start(),
        _llm_start(
            data={"foo": "bar"},
        ),
        _llm_end(data={"content": "hi"}),
        _root_agent_end(),
    ]
    with pytest.raises(DataSchemaViolationError) as exc_info:
        convert(events)

    exc = exc_info.value
    assert exc.uuid == "llm-001"
    assert exc.data_schema["name"] == "openai/chat-completions"
    assert isinstance(exc.path, list)
    assert exc.message  # non-empty jsonschema message
    assert "llm-001" in str(exc)
    assert "openai/chat-completions" in str(exc)


# ---------------------------------------------------------------------------
# Custom schema registration
# ---------------------------------------------------------------------------


def test_register_custom_schema_enables_validation() -> None:
    """Producers can plug their own schema into the registry and it takes
    effect immediately for subsequent ``convert`` calls."""
    key = ("test/myco-payload", "1")
    register_schema(
        "test/myco-payload",
        "1",
        {
            "type": "object",
            "required": ["myco_field"],
        },
    )
    try:
        # Valid payload passes.
        events = [
            _root_agent_start(),
            _llm_start(
                data={"messages": [{"role": "user", "content": "hi"}], "myco_field": "x"},
                data_schema={"name": "test/myco-payload", "version": "1"},
            ),
            _llm_end(data={"content": "hi"}),
            _root_agent_end(),
        ]
        convert(events)

        # Invalid payload (missing myco_field) raises.
        bad_events = [
            _root_agent_start(),
            _llm_start(
                data={"messages": [{"role": "user", "content": "hi"}]},
                data_schema={"name": "test/myco-payload", "version": "1"},
            ),
            _llm_end(data={"content": "hi"}),
            _root_agent_end(),
        ]
        with pytest.raises(DataSchemaViolationError):
            convert(bad_events)
    finally:
        SCHEMA_REGISTRY.pop(key, None)


def test_register_schema_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        register_schema("", "1", {})
    with pytest.raises(ValueError):
        register_schema("x", "", {})
    with pytest.raises(ValueError):
        register_schema("x", "1", "not-a-dict")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
