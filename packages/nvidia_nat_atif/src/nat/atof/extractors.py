# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pluggable payload extractors for the ATOF→ATIF converter.

The ATOF wire envelope is producer-agnostic, but the *contents* of
``event.data`` are producer-defined. The converter must translate those
contents into ATIF step fields (messages, tool calls, tool results,
mark-lifted sources). This module defines three Protocol interfaces and
three registries that let producers plug in their own extractors,
keyed on the producer-declared ``data_schema = {name, version}``:

- :class:`LlmPayloadExtractor` — for ``category == "llm"`` scope events:
  parses input messages, output text, and assistant tool_calls.
- :class:`ToolPayloadExtractor` — for ``category == "tool"`` scope-end
  events: serializes the tool result to a string.
- :class:`MarkPayloadExtractor` — for mark events whose payload carries
  a ``role`` hint that should lift to an ATIF step source.

Ships one built-in extractor per protocol:

- :class:`OpenAiChatCompletionsLlmExtractor` (registered for
  ``openai/chat-completions@1``). Also the fallback for LLM events
  without a ``data_schema``.
- :class:`GenericToolResultExtractor` — unwraps single-key ``{result}``
  or ``{output}`` wrappers, otherwise serializes the payload as JSON.
  Used when no tool extractor is registered for an event's schema.
- :class:`NatRoleMarkExtractor` — lifts marks whose ``data.role`` is
  one of ``"user"``, ``"system"``, ``"agent"``. Used when no mark
  extractor is registered.

Register new extractors before calling the converter:

    from nat.atof.extractors import register_llm_extractor

    class MyExtractor:
        def extract_input_messages(self, data): ...
        def extract_output_text(self, data): ...
        def extract_tool_calls(self, data): ...

    register_llm_extractor("myco/my-llm", "1", MyExtractor())
"""

from __future__ import annotations

import json
from typing import Any
from typing import Protocol
from typing import runtime_checkable


# ---------------------------------------------------------------------------
# Protocol interfaces
# ---------------------------------------------------------------------------


@runtime_checkable
class LlmPayloadExtractor(Protocol):
    """Extracts ATIF-relevant fields from an ``llm`` scope event's ``data``.

    Implementations MUST be pure functions over ``data`` — no side effects,
    no network, no filesystem access. Return empty collections or strings
    when a field is not present; the converter distinguishes "legitimately
    empty" from "shape mismatch" at the dispatch layer.
    """

    def extract_input_messages(self, data: Any) -> list[dict[str, Any]]:
        """Return the chat history messages from an LLM scope-start payload.

        Each message SHOULD carry ``role`` and ``content`` keys; ``content``
        MAY be a string or a multimodal part list (ATIF v1.6+).
        """
        ...

    def extract_output_text(self, data: Any) -> str:
        """Return the assistant text from an LLM scope-end payload.

        Returns ``""`` when the response carries only tool_calls or has no
        text content.
        """
        ...

    def extract_tool_calls(self, data: Any) -> list[dict[str, Any]]:
        """Return the tool_calls issued by the assistant in this turn.

        Each dict MUST carry ``tool_call_id``, ``function_name``, and
        ``arguments`` (dict). Returns ``[]`` when no tool was called.
        """
        ...


@runtime_checkable
class ToolPayloadExtractor(Protocol):
    """Extracts a serialized result string from a ``tool`` scope-end payload."""

    def extract_tool_result(self, data: Any) -> str | None:
        """Return the tool result as a string, or ``None`` when ``data`` is
        ``None``."""
        ...


@runtime_checkable
class MarkPayloadExtractor(Protocol):
    """Classifies a mark event payload as either a role-lifted step
    (user/system/agent) or an opaque system step."""

    def extract_role_and_content(self, data: Any) -> tuple[str, Any] | None:
        """If the mark should lift to an ATIF step with a specific
        ``source``, return ``(source, content)``. Otherwise return ``None``
        to fall through to the opaque-system-step path.

        ``source`` MUST be one of ``"user"``, ``"system"``, ``"agent"``.
        ``content`` is passed through as-is (string or part list).
        """
        ...


# ---------------------------------------------------------------------------
# Default LLM extractor: OpenAI chat-completions
# ---------------------------------------------------------------------------


class OpenAiChatCompletionsLlmExtractor:
    """Reference LLM extractor accepting both direct and nested OpenAI shapes.

    Input shapes (extract_input_messages):
    - ``{"messages": [...]}``
    - ``{"content": {"messages": [...]}}``

    Output shapes (extract_output_text):
    - ``{"content": "..."}``
    - ``{"choices": [{"message": {"content": "..."}}]}``

    Tool-call shapes (extract_tool_calls):
    - Flat: ``{"tool_calls": [{"id", "name", "arguments"}]}``
    - Nested: ``{"choices": [{"message": {"tool_calls": [...]}}]}``
    - Per-call can use either flat ``{id, name, arguments}`` or the OpenAI
      nested ``{id, function: {name, arguments}}`` form.
    """

    def extract_input_messages(self, data: Any) -> list[dict[str, Any]]:
        if not isinstance(data, dict) or not data:
            return []
        content = data.get("content")
        if isinstance(content, dict):
            messages = content.get("messages", [])
            if messages:
                return messages
        messages = data.get("messages", [])
        if messages:
            return messages
        return []

    def extract_output_text(self, data: Any) -> str:
        if not isinstance(data, dict):
            return ""
        content = data.get("content")
        if isinstance(content, str):
            return content
        try:
            choice_content = data["choices"][0]["message"].get("content", "")
            if isinstance(choice_content, str):
                return choice_content
        except (KeyError, IndexError, TypeError):
            pass
        return ""

    def extract_tool_calls(self, data: Any) -> list[dict[str, Any]]:
        if not isinstance(data, dict) or not data:
            return []
        raw_calls = data.get("tool_calls")
        if not raw_calls:
            try:
                raw_calls = data["choices"][0]["message"].get("tool_calls", [])
            except (KeyError, IndexError, TypeError):
                raw_calls = []

        result: list[dict[str, Any]] = []
        for tc in raw_calls or []:
            if "function" in tc and isinstance(tc["function"], dict):
                inner = tc["function"]
                tool_id = tc["id"]
                name = inner.get("name", "")
                args = inner.get("arguments", {})
            else:
                tool_id = tc["id"]
                name = tc.get("name", "")
                args = tc.get("arguments", {})

            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}

            result.append(
                {
                    "tool_call_id": tool_id,
                    "function_name": name,
                    "arguments": args,
                }
            )
        return result


# ---------------------------------------------------------------------------
# Default tool extractor
# ---------------------------------------------------------------------------


class GenericToolResultExtractor:
    """Unwraps ``{result: X}`` or ``{output: X}`` single-key wrappers into
    a primitive or JSON-serialized string; otherwise serializes the whole
    payload as compact JSON."""

    def extract_tool_result(self, data: Any) -> str | None:
        if data is None:
            return None
        if isinstance(data, dict):
            if len(data) == 1:
                key = next(iter(data))
                if key in ("result", "output"):
                    val = data[key]
                    if isinstance(val, (str, int, float, bool)):
                        return str(val)
                    return json.dumps(val, separators=(",", ":"))
            return json.dumps(data, separators=(",", ":"))
        if isinstance(data, str):
            return data
        return str(data)


# ---------------------------------------------------------------------------
# Default mark extractor
# ---------------------------------------------------------------------------


class NatRoleMarkExtractor:
    """Lifts a mark event to a sourced ATIF step when its payload carries
    ``data.role ∈ {"user", "system", "agent"}``. Content is taken from
    ``data.content`` then ``data.message`` (string fallback ``""``)."""

    _VALID_ROLES = frozenset({"user", "system", "agent"})

    def extract_role_and_content(self, data: Any) -> tuple[str, Any] | None:
        if not isinstance(data, dict):
            return None
        role = data.get("role")
        if not isinstance(role, str) or role not in self._VALID_ROLES:
            return None
        content = data.get("content")
        if content is None:
            content = data.get("message")
        if content is None:
            content = ""
        return role, content


# ---------------------------------------------------------------------------
# Registries and resolvers
# ---------------------------------------------------------------------------


DEFAULT_LLM_EXTRACTOR: LlmPayloadExtractor = OpenAiChatCompletionsLlmExtractor()
DEFAULT_TOOL_EXTRACTOR: ToolPayloadExtractor = GenericToolResultExtractor()
DEFAULT_MARK_EXTRACTOR: MarkPayloadExtractor = NatRoleMarkExtractor()


LLM_EXTRACTOR_REGISTRY: dict[tuple[str, str], LlmPayloadExtractor] = {
    ("openai/chat-completions", "1"): DEFAULT_LLM_EXTRACTOR,
}
TOOL_EXTRACTOR_REGISTRY: dict[tuple[str, str], ToolPayloadExtractor] = {}
MARK_EXTRACTOR_REGISTRY: dict[tuple[str, str], MarkPayloadExtractor] = {}


def _validate_key(name: str, version: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    if not isinstance(version, str) or not version:
        raise ValueError("version must be a non-empty string")


def register_llm_extractor(name: str, version: str, extractor: LlmPayloadExtractor) -> None:
    """Register an LLM payload extractor for ``(name, version)``."""
    _validate_key(name, version)
    if not isinstance(extractor, LlmPayloadExtractor):
        raise TypeError("extractor must implement the LlmPayloadExtractor protocol")
    LLM_EXTRACTOR_REGISTRY[(name, version)] = extractor


def register_tool_extractor(name: str, version: str, extractor: ToolPayloadExtractor) -> None:
    """Register a tool payload extractor for ``(name, version)``."""
    _validate_key(name, version)
    if not isinstance(extractor, ToolPayloadExtractor):
        raise TypeError("extractor must implement the ToolPayloadExtractor protocol")
    TOOL_EXTRACTOR_REGISTRY[(name, version)] = extractor


def register_mark_extractor(name: str, version: str, extractor: MarkPayloadExtractor) -> None:
    """Register a mark payload extractor for ``(name, version)``."""
    _validate_key(name, version)
    if not isinstance(extractor, MarkPayloadExtractor):
        raise TypeError("extractor must implement the MarkPayloadExtractor protocol")
    MARK_EXTRACTOR_REGISTRY[(name, version)] = extractor


def _resolve(
    registry: dict[tuple[str, str], Any],
    data_schema: dict[str, Any] | None,
    default: Any,
) -> Any:
    if not isinstance(data_schema, dict):
        return default
    name = data_schema.get("name")
    version = data_schema.get("version")
    if not isinstance(name, str) or not isinstance(version, str):
        return default
    return registry.get((name, version), default)


def resolve_llm_extractor(data_schema: dict[str, Any] | None) -> LlmPayloadExtractor:
    """Return the LLM extractor registered for ``data_schema``, or the
    built-in OpenAI chat-completions extractor if unregistered/absent."""
    return _resolve(LLM_EXTRACTOR_REGISTRY, data_schema, DEFAULT_LLM_EXTRACTOR)


def resolve_tool_extractor(data_schema: dict[str, Any] | None) -> ToolPayloadExtractor:
    """Return the tool extractor registered for ``data_schema``, or the
    generic result-unwrap extractor if unregistered/absent."""
    return _resolve(TOOL_EXTRACTOR_REGISTRY, data_schema, DEFAULT_TOOL_EXTRACTOR)


def resolve_mark_extractor(data_schema: dict[str, Any] | None) -> MarkPayloadExtractor:
    """Return the mark extractor registered for ``data_schema``, or the
    built-in role-lifting extractor if unregistered/absent."""
    return _resolve(MARK_EXTRACTOR_REGISTRY, data_schema, DEFAULT_MARK_EXTRACTOR)


__all__ = [
    "DEFAULT_LLM_EXTRACTOR",
    "DEFAULT_MARK_EXTRACTOR",
    "DEFAULT_TOOL_EXTRACTOR",
    "GenericToolResultExtractor",
    "LLM_EXTRACTOR_REGISTRY",
    "LlmPayloadExtractor",
    "MARK_EXTRACTOR_REGISTRY",
    "MarkPayloadExtractor",
    "NatRoleMarkExtractor",
    "OpenAiChatCompletionsLlmExtractor",
    "TOOL_EXTRACTOR_REGISTRY",
    "ToolPayloadExtractor",
    "register_llm_extractor",
    "register_mark_extractor",
    "register_tool_extractor",
    "resolve_llm_extractor",
    "resolve_mark_extractor",
    "resolve_tool_extractor",
]
