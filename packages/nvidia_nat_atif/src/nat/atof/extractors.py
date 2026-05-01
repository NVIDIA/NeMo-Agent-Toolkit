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

LLM extractors are produced by combining a declarative :class:`SchemaMap`
with the generic :class:`SchemaMapLlmExtractor` engine. A ``SchemaMap``
captures the per-provider field paths (where input messages live, where
output text lives, where tool calls live) plus three optional hooks for
the irreducible per-provider transforms: polymorphic content unpacking,
output-message decomposition, and tool-call shape adaptation. Most
providers are expressible as pure paths; richer providers (Anthropic
content blocks, Gemini parts) use the hooks.

Ships one built-in extractor per protocol:

- :class:`OpenAiChatCompletionsLlmExtractor` — a :class:`SchemaMapLlmExtractor`
  configured by :data:`OPENAI_CHAT_COMPLETIONS_V1_MAP`. Registered for
  ``openai/chat-completions@1`` and used as the fallback for LLM events
  without a ``data_schema``.
- :class:`GenericToolResultExtractor` — unwraps single-key ``{result}``
  or ``{output}`` wrappers, otherwise serializes the payload as JSON.
  Used when no tool extractor is registered for an event's schema.
- :class:`NatRoleMarkExtractor` — lifts marks whose ``data.role`` is
  one of ``"user"``, ``"system"``, ``"agent"``. Used when no mark
  extractor is registered.

Register new extractors before calling the converter. For an
OpenAI-shaped provider, define a SchemaMap and register it:

    from nat.atof.extractors import (
        SchemaMap, SchemaMapLlmExtractor, register_llm_extractor,
    )

    MYCO_MAP = SchemaMap(
        name="myco/chat", version="1",
        input_messages_paths=("messages",),
        output_text_paths=("response",),
        output_tool_calls_paths=("tool_calls",),
    )
    register_llm_extractor("myco/chat", "1", SchemaMapLlmExtractor(MYCO_MAP))

For richer shapes (Anthropic content blocks, Gemini parts), use the
hook fields on SchemaMap to handle the irreducible transforms.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
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
# Schema-map engine: declarative path resolver + optional hooks
# ---------------------------------------------------------------------------


def _resolve_path(data: Any, path: str) -> Any:
    """Walk a dotted path through nested dicts/lists. Returns ``None`` on miss.

    Path components are segmented on ``"."``. A digit-only segment indexes
    into a list at that position; any other segment is a dict key. Returns
    the value at the final position, or ``None`` if any step fails.

    Examples:
        _resolve_path({"a": {"b": 1}}, "a.b")               # → 1
        _resolve_path({"a": [{"b": 2}]}, "a.0.b")           # → 2
        _resolve_path({"a": 1}, "a.b")                      # → None
        _resolve_path({}, "x")                              # → None
    """
    if not path:
        return data
    current: Any = data
    for part in path.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
        elif isinstance(current, list):
            if not part.isdigit():
                return None
            idx = int(part)
            if idx >= len(current):
                return None
            current = current[idx]
        else:
            return None
    return current


def _resolve_first(data: Any, paths: tuple[str, ...]) -> Any:
    """Try each path in order; return the first non-``None`` value, else ``None``."""
    for p in paths:
        value = _resolve_path(data, p)
        if value is not None:
            return value
    return None


@dataclass(frozen=True)
class SchemaMap:
    """Declarative description of where ATIF-relevant fields live within a
    provider's LLM payload, plus optional hooks for irreducible transforms.

    A ``SchemaMap`` captures three things:

    1. **Field paths** — dotted paths (with numeric list indices) telling
       the engine where to find input messages, output text, and output
       tool calls. Each field accepts a tuple of candidate paths; the
       engine tries them in order and uses the first hit.

    2. **Per-tool-call sub-paths** — for providers whose tool-call shape
       fits the OpenAI flat-or-nested convention. Each tool call is a dict;
       these paths name where ID/name/arguments live within that dict.

    3. **Optional hooks** — escape hatches for the three transforms that
       can't be expressed declaratively:

       - ``normalize_input_messages``: input ``data`` → ATIF-shaped
         message list. Use when content is polymorphic (Anthropic
         string-or-blocks, Gemini parts) and a single field-path can't
         flatten it.
       - ``normalize_output_message``: output ``data`` → ``(text, tool_calls)``
         pair. Use when output text and tool calls coexist in the same
         polymorphic structure (Anthropic ``content`` blocks).
       - ``transform_tool_call``: per-call dict adapter. Use when tool
         calls don't carry an ID (Gemini synthesizes from name+index)
         or use non-OpenAI nesting.

    Hooks always win over paths. If ``normalize_output_message`` is set,
    the engine ignores ``output_text_paths`` and ``output_tool_calls_paths``.

    Pure-paths providers (OpenAI) leave the hooks at ``None``. Mixed
    providers (Anthropic, Gemini) use one or two hooks.

    Args:
        name: Schema name (e.g. ``"openai/chat-completions"``).
        version: Schema version string.
        input_messages_paths: Candidate paths to the input messages array.
        output_text_paths: Candidate paths to the output assistant text.
        output_tool_calls_paths: Candidate paths to the output tool-calls array.
        tool_call_id_paths: Candidate sub-paths for tool-call ID.
        tool_call_name_paths: Candidate sub-paths for tool-call function name.
        tool_call_args_paths: Candidate sub-paths for tool-call arguments.
        tool_call_args_parse_json: When True, parse string arguments as JSON.
        role_aliases: Map of provider role values to canonical role values
            (e.g., ``{"model": "assistant"}`` for Gemini). Applied to messages
            extracted via field paths; hooks bypass this.
        normalize_input_messages: Optional hook overriding path-based input
            extraction. Signature: ``(data) → list[{"role", "content", ...}]``.
        normalize_output_message: Optional hook overriding path-based output
            extraction. Signature: ``(data) → (text, tool_calls)``.
        transform_tool_call: Optional per-call adapter. Signature:
            ``(raw_call_dict, index) → ATIF-shaped {"tool_call_id", "function_name", "arguments"}``.
            When set, replaces the per-tool-call path resolution entirely.
    """

    name: str
    version: str

    input_messages_paths: tuple[str, ...] = ()
    output_text_paths: tuple[str, ...] = ()
    output_tool_calls_paths: tuple[str, ...] = ()

    tool_call_id_paths: tuple[str, ...] = ("id", )
    tool_call_name_paths: tuple[str, ...] = ("name", "function.name")
    tool_call_args_paths: tuple[str, ...] = ("arguments", "function.arguments")
    tool_call_args_parse_json: bool = True

    role_aliases: Mapping[str, str] = field(default_factory=dict)

    normalize_input_messages: Callable[[Any], list[dict[str, Any]]] | None = None
    normalize_output_message: Callable[[Any], tuple[str, list[dict[str, Any]]]] | None = None
    transform_tool_call: Callable[[dict[str, Any], int], dict[str, Any]] | None = None


class SchemaMapLlmExtractor:
    """Generic LLM payload extractor driven by a :class:`SchemaMap`.

    Implements :class:`LlmPayloadExtractor` by routing extraction through
    the map's hooks (when set) or its declarative field paths (otherwise).
    A single instance per ``(name, version)`` is the intended pattern;
    register it with :func:`register_llm_extractor`.
    """

    def __init__(self, schema_map: SchemaMap) -> None:
        self.schema_map = schema_map

    def extract_input_messages(self, data: Any) -> list[dict[str, Any]]:
        if not isinstance(data, dict) or not data:
            return []

        if self.schema_map.normalize_input_messages is not None:
            return self.schema_map.normalize_input_messages(data)

        raw = _resolve_first(data, self.schema_map.input_messages_paths)
        if not isinstance(raw, list):
            return []
        return self._apply_role_aliases(raw)

    def extract_output_text(self, data: Any) -> str:
        if not isinstance(data, dict):
            return ""

        if self.schema_map.normalize_output_message is not None:
            text, _ = self.schema_map.normalize_output_message(data)
            return text

        value = _resolve_first(data, self.schema_map.output_text_paths)
        if isinstance(value, str):
            return value
        return ""

    def extract_tool_calls(self, data: Any) -> list[dict[str, Any]]:
        if not isinstance(data, dict) or not data:
            return []

        if self.schema_map.normalize_output_message is not None:
            _, tool_calls = self.schema_map.normalize_output_message(data)
            return tool_calls

        raw_calls = _resolve_first(data, self.schema_map.output_tool_calls_paths)
        if not isinstance(raw_calls, list):
            return []

        result: list[dict[str, Any]] = []
        for idx, raw in enumerate(raw_calls):
            if not isinstance(raw, dict):
                continue
            if self.schema_map.transform_tool_call is not None:
                result.append(self.schema_map.transform_tool_call(raw, idx))
            else:
                result.append(self._extract_tool_call_fields(raw))
        return result

    def _apply_role_aliases(self, messages: list[Any]) -> list[dict[str, Any]]:
        aliases = self.schema_map.role_aliases
        if not aliases:
            return [m for m in messages if isinstance(m, dict)]
        out: list[dict[str, Any]] = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            if isinstance(role, str) and role in aliases:
                m = {**m, "role": aliases[role]}
            out.append(m)
        return out

    def _extract_tool_call_fields(self, raw: dict[str, Any]) -> dict[str, Any]:
        tool_id = _resolve_first(raw, self.schema_map.tool_call_id_paths)
        name = _resolve_first(raw, self.schema_map.tool_call_name_paths) or ""
        args: Any = _resolve_first(raw, self.schema_map.tool_call_args_paths)
        if args is None:
            args = {}

        if self.schema_map.tool_call_args_parse_json and isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}

        return {
            "tool_call_id": tool_id,
            "function_name": name,
            "arguments": args,
        }


# ---------------------------------------------------------------------------
# Built-in: OpenAI chat-completions schema map (no hooks, pure paths)
# ---------------------------------------------------------------------------

# Order matters: the engine tries paths left-to-right and returns the first
# non-None hit. ``content.messages`` precedes ``messages`` so a payload
# carrying both (rare) prefers the nested form, matching the historical
# precedence of the hand-rolled OpenAI extractor.
OPENAI_CHAT_COMPLETIONS_V1_MAP = SchemaMap(
    name="openai/chat-completions",
    version="1",
    input_messages_paths=("content.messages", "messages"),
    output_text_paths=("content", "choices.0.message.content"),
    output_tool_calls_paths=("tool_calls", "choices.0.message.tool_calls"),
    tool_call_id_paths=("id", ),
    tool_call_name_paths=("name", "function.name"),
    tool_call_args_paths=("arguments", "function.arguments"),
    tool_call_args_parse_json=True,
)


class OpenAiChatCompletionsLlmExtractor(SchemaMapLlmExtractor):
    """Reference LLM extractor accepting both direct and nested OpenAI shapes.

    Thin convenience wrapper around :data:`OPENAI_CHAT_COMPLETIONS_V1_MAP`.
    Behavior is identical to instantiating
    ``SchemaMapLlmExtractor(OPENAI_CHAT_COMPLETIONS_V1_MAP)``.

    Input shapes (extract_input_messages):
    - ``{"messages": [...]}``
    - ``{"content": {"messages": [...]}}``

    Output shapes (extract_output_text):
    - ``{"content": "..."}``
    - ``{"choices": [{"message": {"content": "..."}}]}``

    Tool-call shapes (extract_tool_calls):
    - Flat: ``{"tool_calls": [{"id", "name", "arguments"}]}``
    - Nested: ``{"choices": [{"message": {"tool_calls": [...]}}]}``
    - Per-call: either flat ``{id, name, arguments}`` or the OpenAI
      ``{id, function: {name, arguments}}`` form.
    """

    def __init__(self) -> None:
        super().__init__(OPENAI_CHAT_COMPLETIONS_V1_MAP)


# ---------------------------------------------------------------------------
# Built-in: Anthropic Messages schema map (uses content-block hooks)
# ---------------------------------------------------------------------------
#
# Anthropic's Messages API carries text and tool-uses in the same
# ``content`` field — a polymorphic list of typed blocks (``text``,
# ``tool_use``, ``tool_result``). Path-based extraction can't split that
# list into ATIF's separate text/tool_calls slots, so the SchemaMap uses
# the ``normalize_input_messages`` and ``normalize_output_message`` hooks.
#
# Tool results from prior turns arrive on the wire as ``user``-role
# messages whose content is a list containing ``tool_result`` blocks
# (Anthropic's transport for "here's what the tool returned, keep going").
# In ATIF those results are sourced from the corresponding tool scope-end
# event, not from the LLM input. The input hook deliberately drops
# ``tool_result`` blocks so they don't double-emit as user steps.


def _anthropic_normalize_input_messages(data: Any) -> list[dict[str, Any]]:
    """Flatten Anthropic ``messages`` (with polymorphic content) to
    ``[{"role", "content"}]`` for the converter.

    Per-message rules:
    - String content → emitted unchanged.
    - List content with text blocks → text blocks concatenated into a
      single string (round-trip-clean: each block's text joined with no
      separator, matching Anthropic's own ``response.content[*].text``
      concatenation semantics).
    - List content with only ``tool_use`` / ``tool_result`` blocks →
      message dropped (tool I/O is captured by tool scope events, not
      LLM input messages — see module-level note above).

    Non-dict messages and non-string roles are skipped.
    """
    if not isinstance(data, dict):
        return []
    messages = data.get("messages")
    if not isinstance(messages, list):
        return []

    out: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if not isinstance(role, str):
            continue
        content = msg.get("content")

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text = block.get("text", "")
                    if isinstance(text, str) and text:
                        text_parts.append(text)
            if text_parts:
                out.append({"role": role, "content": "".join(text_parts)})
            # Pure tool_use / tool_result messages: skip — captured elsewhere.

    return out


def _anthropic_normalize_output_message(data: Any, ) -> tuple[str, list[dict[str, Any]]]:
    """Decompose an Anthropic response's top-level ``content`` block list
    into ``(text, tool_calls)``.

    The response shape is ``{"role": "assistant", "content": [<blocks>], ...}``
    where blocks are typed: ``{"type": "text", "text": ...}`` for text,
    ``{"type": "tool_use", "id", "name", "input": {dict}}`` for tool calls.
    Anthropic sends ``input`` already as a dict — no JSON parsing needed.
    """
    if not isinstance(data, dict):
        return "", []
    content = data.get("content")
    if not isinstance(content, list):
        return "", []

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text", "")
            if isinstance(text, str):
                text_parts.append(text)
        elif block_type == "tool_use":
            inp = block.get("input")
            if not isinstance(inp, dict):
                inp = {}
            tool_calls.append({
                "tool_call_id": block.get("id", ""),
                "function_name": block.get("name", ""),
                "arguments": inp,
            })

    return "".join(text_parts), tool_calls


ANTHROPIC_MESSAGES_V1_MAP = SchemaMap(
    name="anthropic/messages",
    version="1",
    normalize_input_messages=_anthropic_normalize_input_messages,
    normalize_output_message=_anthropic_normalize_output_message,
)


def register_anthropic_messages_v1() -> None:
    """Install the Anthropic Messages JSON Schema and LLM extractor.

    Idempotent — safe to call multiple times. Registers
    ``anthropic/messages@1`` in both :data:`SCHEMA_REGISTRY` (validation)
    and :data:`LLM_EXTRACTOR_REGISTRY` (extraction). Call this once at
    process startup before invoking the converter on Anthropic-shaped
    payloads.
    """
    # Lazy import: defer to call site so a SCHEMA_REGISTRY consumer that
    # only wants OpenAI doesn't pay the (tiny) cost at module import.
    from nat.atof.schemas import ANTHROPIC_MESSAGES_V1
    from nat.atof.schemas import register_schema

    register_schema("anthropic/messages", "1", ANTHROPIC_MESSAGES_V1)
    register_llm_extractor(
        "anthropic/messages",
        "1",
        SchemaMapLlmExtractor(ANTHROPIC_MESSAGES_V1_MAP),
    )


# ---------------------------------------------------------------------------
# Built-in: Gemini generateContent schema map (uses parts-list hooks)
# ---------------------------------------------------------------------------
#
# Gemini's generateContent API uses a different polymorphic structure
# than Anthropic. Each turn carries ``parts: [<part>]`` where each part
# is exactly one of ``{text}``, ``{functionCall: {name, args}}``, or
# ``{functionResponse: {name, response}}``. Roles are ``"user"`` or
# ``"model"`` (note the renaming from "assistant"). Tool calls have no
# vendor-supplied IDs — Gemini matches function responses to function
# calls by ``name`` only.
#
# Both input and output share the same parts shape, but live at
# different paths: input at ``contents[].parts``, output at
# ``candidates[0].content.parts``. The hooks handle both; the
# ``role_aliases`` field is unused since hooks don't consult it.


def _gemini_walk_parts_for_text(parts: Any) -> str:
    if not isinstance(parts, list):
        return ""
    chunks: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str):
            chunks.append(text)
    return "".join(chunks)


def _gemini_walk_parts_for_tool_calls(parts: Any) -> list[dict[str, Any]]:
    if not isinstance(parts, list):
        return []
    out: list[dict[str, Any]] = []
    for idx, part in enumerate(parts):
        if not isinstance(part, dict):
            continue
        fc = part.get("functionCall")
        if isinstance(fc, dict):
            name = fc.get("name", "")
            args = fc.get("args")
            if not isinstance(args, dict):
                args = {}
            # Gemini doesn't provide a tool_call_id; synthesize a stable
            # one from name + ordinal so downstream ATIF observation
            # reconciliation has a key. Producers can override by adding
            # a custom ``tool_call_id`` field to the part dict — we
            # honour it if present.
            tool_id = part.get("tool_call_id") or f"{name}__{idx}"
            out.append({
                "tool_call_id": tool_id,
                "function_name": name,
                "arguments": args,
            })
    return out


def _gemini_normalize_input_messages(data: Any) -> list[dict[str, Any]]:
    """Flatten Gemini ``contents[].parts[]`` to ATIF-shaped messages.

    Role aliasing: Gemini uses ``"model"`` for assistant turns —
    normalised to ``"assistant"`` so downstream consumers see a uniform
    vocabulary. Tool call/response parts are dropped from input
    extraction (captured by tool scope events).
    """
    if not isinstance(data, dict):
        return []
    contents = data.get("contents")
    if not isinstance(contents, list):
        return []

    out: list[dict[str, Any]] = []
    for turn in contents:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        if role == "model":
            role = "assistant"
        if not isinstance(role, str):
            continue
        text = _gemini_walk_parts_for_text(turn.get("parts"))
        if text:
            out.append({"role": role, "content": text})

    return out


def _gemini_normalize_output_message(data: Any, ) -> tuple[str, list[dict[str, Any]]]:
    """Decompose a Gemini response's first candidate into ``(text, tool_calls)``.

    Gemini may return multiple candidates — ATIF represents a single
    assistant turn, so we use ``candidates[0]`` (the highest-ranked one)
    and ignore the rest. This matches Gemini's typical usage where
    ``candidate_count`` defaults to 1.
    """
    if not isinstance(data, dict):
        return "", []
    candidate = _resolve_path(data, "candidates.0.content")
    if not isinstance(candidate, dict):
        return "", []
    parts = candidate.get("parts")
    return (
        _gemini_walk_parts_for_text(parts),
        _gemini_walk_parts_for_tool_calls(parts),
    )


GEMINI_GENERATE_CONTENT_V1_MAP = SchemaMap(
    name="gemini/generate-content",
    version="1",
    role_aliases={"model": "assistant"},
    normalize_input_messages=_gemini_normalize_input_messages,
    normalize_output_message=_gemini_normalize_output_message,
)


def register_gemini_generate_content_v1() -> None:
    """Install the Gemini generateContent JSON Schema and LLM extractor.

    Idempotent — safe to call multiple times. Registers
    ``gemini/generate-content@1`` in both :data:`SCHEMA_REGISTRY` and
    :data:`LLM_EXTRACTOR_REGISTRY`. Call this once at process startup
    before invoking the converter on Gemini-shaped payloads.
    """
    from nat.atof.schemas import GEMINI_GENERATE_CONTENT_V1
    from nat.atof.schemas import register_schema

    register_schema("gemini/generate-content", "1", GEMINI_GENERATE_CONTENT_V1)
    register_llm_extractor(
        "gemini/generate-content",
        "1",
        SchemaMapLlmExtractor(GEMINI_GENERATE_CONTENT_V1_MAP),
    )


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
    "ANTHROPIC_MESSAGES_V1_MAP",
    "DEFAULT_LLM_EXTRACTOR",
    "DEFAULT_MARK_EXTRACTOR",
    "DEFAULT_TOOL_EXTRACTOR",
    "GEMINI_GENERATE_CONTENT_V1_MAP",
    "GenericToolResultExtractor",
    "LLM_EXTRACTOR_REGISTRY",
    "LlmPayloadExtractor",
    "MARK_EXTRACTOR_REGISTRY",
    "MarkPayloadExtractor",
    "NatRoleMarkExtractor",
    "OPENAI_CHAT_COMPLETIONS_V1_MAP",
    "OpenAiChatCompletionsLlmExtractor",
    "SchemaMap",
    "SchemaMapLlmExtractor",
    "TOOL_EXTRACTOR_REGISTRY",
    "ToolPayloadExtractor",
    "register_anthropic_messages_v1",
    "register_gemini_generate_content_v1",
    "register_llm_extractor",
    "register_mark_extractor",
    "register_tool_extractor",
    "resolve_llm_extractor",
    "resolve_mark_extractor",
    "resolve_tool_extractor",
]
