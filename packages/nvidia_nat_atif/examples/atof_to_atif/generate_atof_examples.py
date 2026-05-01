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
"""Generate ATOF v0.1 example JSONL files.

- **EXMP-01 — tier-1 raw pass-through**: A calculator-shaped workflow where the
  producer can't classify any scope. Every scope carries ``category:
  "unknown"``, ``category_profile: null``, with opaque raw JSON in ``data``.
  Demonstrates the floor — a valid ATOF stream carrying only timing + raw
  payloads. Converts to a sequence of opaque ATIF system steps (each scope-end
  becomes one system step).

- **EXMP-02 — tier-2 semantic-tagged**: Same calculator workflow as EXMP-01 but
  with every scope classified (``category: "agent"/"llm"/"tool"``) and
  ``category_profile`` populated (``category_profile.model_name`` for llm
  events, ``category_profile.tool_call_id`` for tool events). Demonstrates
  ``attributes: ["remote"]`` on the tool scope and ``data_schema`` on the llm
  scopes. Converts to a 5-step rich ATIF trajectory (user → agent → system →
  user → agent).

- **EXMP-03 — mark events**: A short chat agent bracketed by ``mark`` events
  (session-start / session-end checkpoints). Demonstrates the second event
  kind — ``mark`` — including the generic checkpoint form (``category: null``,
  just a named timestamp with associated data).

- **EXMP-04 — Anthropic Messages**: A document-summarization workflow where
  Claude calls a ``read_file`` tool, then formulates a summary. LLM payloads
  use Anthropic's Messages API shape — ``messages[].content`` polymorphic
  string-or-block-list on input, ``content[]`` typed blocks on output
  (``text`` + ``tool_use``). Demonstrates that the schema-map-driven
  extractor handles polymorphic content via the ``normalize_*`` hooks.

- **EXMP-05 — Gemini generateContent**: A timezone lookup workflow where
  Gemini calls a ``get_current_time`` function, then answers. LLM payloads
  use Gemini's ``contents[].parts[]`` request shape and ``candidates[0].content.parts[]``
  response shape. Demonstrates ``role_aliases`` (Gemini's ``"model"`` →
  ``"assistant"``) and synthesized tool_call_ids (Gemini doesn't supply
  IDs — extractor synthesizes ``name__index``).

- **EXMP-06 — Heterogeneous router**: A real plausible orchestrator that
  receives a multi-part request, routes pieces to specialist LLMs from
  different providers, and combines the responses. One stream contains
  three LLM scope events whose ``data_schema`` declares OpenAI, Anthropic,
  and Gemini in turn — the strongest end-to-end evidence that the
  converter dispatches per-event by schema, not per-stream.

Usage:
    python generate_atof_examples.py [--output-dir DIR]

See ATOF spec §1.1 (two enrichment tiers), §3 (event kinds), §4 (category
vocabulary).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nat.atof import Event
from nat.atof import MarkEvent
from nat.atof import ScopeEvent
from nat.atof import write_jsonl

OUTPUT_DIR = Path(__file__).parent / "output"

# Schema identifiers reused across the LLM turns in each example.
_OPENAI_CHAT_SCHEMA = {"name": "openai/chat-completions", "version": "1"}
_ANTHROPIC_MESSAGES_SCHEMA = {"name": "anthropic/messages", "version": "1"}
_GEMINI_GENERATE_CONTENT_SCHEMA = {"name": "gemini/generate-content", "version": "1"}

# ---------------------------------------------------------------------------
# Shared timestamps (deterministic for diff-able output)
# ---------------------------------------------------------------------------


def _ts(scenario: int, second: int) -> str:
    """RFC 3339 timestamp helper. Maps scenario index to a deterministic day
    in January 2026 so each example's events are sorted and diff-able.
    """
    return f"2026-01-{scenario:02d}T00:00:{second:02d}Z"


# ---------------------------------------------------------------------------
# EXMP-01: Raw pass-through — tier-1 (all scopes opaque / category=unknown)
# ---------------------------------------------------------------------------


def generate_exmp01() -> list[Event]:
    """A calculator-shaped workflow where the producer can't classify any
    scope. Every scope carries ``category: "unknown"``, ``category_profile:
    None``, and opaque raw JSON in ``data``. Eight events total (4 paired
    scope events).

    Demonstrates the tier-1 floor: a valid ATOF stream capturing only timing +
    raw payloads. Converts to a 4-step ATIF trajectory of opaque system steps
    via the reference converter's generic scope-end fall-through
    (see README → Conversion reference).
    """
    events: list[Event] = [
        # Outer wrapper scope — opaque, no semantic class
        ScopeEvent(
            scope_category="start",
            uuid="root-001",
            parent_uuid=None,
            timestamp=_ts(1, 0),
            name="opaque_workflow",
            attributes=[],
            category="unknown",
            data={"raw_query": "What is 3 + 4?"},
        ),
        # Inner callback 1 — opaque
        ScopeEvent(
            scope_category="start",
            uuid="inner-001",
            parent_uuid="root-001",
            timestamp=_ts(1, 1),
            name="provider_callback_1",
            attributes=[],
            category="unknown",
            data={"raw_payload": "<provider request 1>"},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="inner-001",
            parent_uuid="root-001",
            timestamp=_ts(1, 2),
            name="provider_callback_1",
            attributes=[],
            category="unknown",
            data={"raw_payload": "<provider response 1: tool invocation>"},
        ),
        # Inner callback 2 — opaque
        ScopeEvent(
            scope_category="start",
            uuid="inner-002",
            parent_uuid="root-001",
            timestamp=_ts(1, 3),
            name="provider_callback_2",
            attributes=[],
            category="unknown",
            data={"raw_payload": "<provider request 2>"},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="inner-002",
            parent_uuid="root-001",
            timestamp=_ts(1, 4),
            name="provider_callback_2",
            attributes=[],
            category="unknown",
            data={"raw_payload": "<provider response 2: tool result>"},
        ),
        # Inner callback 3 — opaque
        ScopeEvent(
            scope_category="start",
            uuid="inner-003",
            parent_uuid="root-001",
            timestamp=_ts(1, 5),
            name="provider_callback_3",
            attributes=[],
            category="unknown",
            data={"raw_payload": "<provider request 3>"},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="inner-003",
            parent_uuid="root-001",
            timestamp=_ts(1, 6),
            name="provider_callback_3",
            attributes=[],
            category="unknown",
            data={"raw_payload": "<provider response 3: final answer>"},
        ),
        # Outer wrapper ends
        ScopeEvent(
            scope_category="end",
            uuid="root-001",
            parent_uuid=None,
            timestamp=_ts(1, 7),
            name="opaque_workflow",
            attributes=[],
            category="unknown",
            data={"raw_result": "3 + 4 = 7"},
        ),
    ]
    return events


# ---------------------------------------------------------------------------
# EXMP-02: Simple Tool Call — tier-2 semantic-tagged
# ---------------------------------------------------------------------------


def generate_exmp02() -> list[Event]:
    """A single calculator tool call. Eight events (4 paired scope events).

    Workflow: agent → llm (decides to call calculator__add) → tool runs →
    llm (formulates final answer) → agent done.

    Demonstrates:
    - ``category`` + ``category_profile`` for ``agent`` / ``llm`` / ``tool``.
    - ``attributes: ["remote"]`` on the tool scope (spec §2.1) — the tool
      executes out-of-process (HTTP / MCP / subprocess).
    - ``data_schema`` on llm scopes pointing at a canonical schema identifier
      (``openai/chat-completions.v1``) that a consumer can validate ``data``
      against (spec §2, §3).
    """
    events: list[Event] = [
        ScopeEvent(
            scope_category="start",
            uuid="agent-001",
            parent_uuid=None,
            timestamp=_ts(2, 0),
            name="calculator_agent",
            attributes=[],
            category="agent",
            data={"input": "What is 3 + 4?"},
        ),
        ScopeEvent(
            scope_category="start",
            uuid="llm-001",
            parent_uuid="agent-001",
            timestamp=_ts(2, 1),
            name="gpt-4.1",
            attributes=[],
            category="llm",
            category_profile={"model_name": "gpt-4.1"},
            data={"messages": [{
                "role": "user", "content": "What is 3 + 4?"
            }]},
            data_schema=_OPENAI_CHAT_SCHEMA,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="llm-001",
            parent_uuid="agent-001",
            timestamp=_ts(2, 2),
            name="gpt-4.1",
            attributes=[],
            category="llm",
            category_profile={"model_name": "gpt-4.1"},
            data={
                "content": "",
                "tool_calls": [{
                    "id": "call_abc", "name": "calculator__add", "arguments": {
                        "a": 3, "b": 4
                    }
                }, ],
            },
            data_schema=_OPENAI_CHAT_SCHEMA,
        ),
        ScopeEvent(
            scope_category="start",
            uuid="tool-001",
            parent_uuid="agent-001",
            timestamp=_ts(2, 3),
            name="calculator__add",
            attributes=["remote"],
            category="tool",
            category_profile={"tool_call_id": "call_abc"},
            data={
                "a": 3, "b": 4
            },
        ),
        ScopeEvent(
            scope_category="end",
            uuid="tool-001",
            parent_uuid="agent-001",
            timestamp=_ts(2, 4),
            name="calculator__add",
            attributes=["remote"],
            category="tool",
            category_profile={"tool_call_id": "call_abc"},
            data={"result": 7},
        ),
        ScopeEvent(
            scope_category="start",
            uuid="llm-002",
            parent_uuid="agent-001",
            timestamp=_ts(2, 5),
            name="gpt-4.1",
            attributes=[],
            category="llm",
            category_profile={"model_name": "gpt-4.1"},
            data={
                "messages": [
                    {
                        "role": "user", "content": "What is 3 + 4?"
                    },
                    {
                        "role":
                            "assistant",
                        "tool_calls": [{
                            "id": "call_abc",
                            "name": "calculator__add",
                            "arguments": {
                                "a": 3, "b": 4
                            },
                        }, ],
                    },
                    {
                        "role": "tool", "tool_call_id": "call_abc", "content": {
                            "result": 7
                        }
                    },
                ]
            },
            data_schema=_OPENAI_CHAT_SCHEMA,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="llm-002",
            parent_uuid="agent-001",
            timestamp=_ts(2, 6),
            name="gpt-4.1",
            attributes=[],
            category="llm",
            category_profile={"model_name": "gpt-4.1"},
            data={"content": "3 + 4 = 7"},
            data_schema=_OPENAI_CHAT_SCHEMA,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="agent-001",
            parent_uuid=None,
            timestamp=_ts(2, 7),
            name="calculator_agent",
            attributes=[],
            category="agent",
            data={"response": "3 + 4 = 7"},
        ),
    ]
    return events


# ---------------------------------------------------------------------------
# EXMP-03: Chat agent with session-boundary mark events
# ---------------------------------------------------------------------------


def generate_exmp03() -> list[Event]:
    """A short chat agent bracketed by two ``mark`` events.

    Demonstrates the ``mark`` event kind (spec §3.2):
    - Generic checkpoint form — ``category`` and ``category_profile`` are both
      absent / null; the mark is just a named timestamp with associated data.
    - Unpaired — marks are single-shot, no start/end semantics.

    Workflow:
        session_start mark → agent turn (one llm call, no tools) → session_end mark.

    Six events total (2 marks + 2 paired scope events for agent + 2 paired
    scope events for the single llm turn). Converts to a 4-step ATIF
    trajectory (system → user → agent → system) with the two marks
    materialising as ``source: "system"`` steps carrying the mark's ``data``
    as the message.
    """
    events: list[Event] = [
        MarkEvent(
            uuid="mark-start-001",
            parent_uuid=None,
            timestamp=_ts(3, 0),
            name="session_start",
            data={
                "session_id": "sess-003", "user_id": "user-042"
            },
        ),
        ScopeEvent(
            scope_category="start",
            uuid="agent-003",
            parent_uuid=None,
            timestamp=_ts(3, 1),
            name="chat_agent",
            attributes=[],
            category="agent",
            data={"input": "What's the capital of France?"},
        ),
        ScopeEvent(
            scope_category="start",
            uuid="llm-003",
            parent_uuid="agent-003",
            timestamp=_ts(3, 2),
            name="gpt-4.1",
            attributes=[],
            category="llm",
            category_profile={"model_name": "gpt-4.1"},
            data={"messages": [{
                "role": "user", "content": "What's the capital of France?"
            }]},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="llm-003",
            parent_uuid="agent-003",
            timestamp=_ts(3, 3),
            name="gpt-4.1",
            attributes=[],
            category="llm",
            category_profile={"model_name": "gpt-4.1"},
            data={"content": "The capital of France is Paris."},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="agent-003",
            parent_uuid=None,
            timestamp=_ts(3, 4),
            name="chat_agent",
            attributes=[],
            category="agent",
            data={"response": "The capital of France is Paris."},
        ),
        MarkEvent(
            uuid="mark-end-001",
            parent_uuid=None,
            timestamp=_ts(3, 5),
            name="session_end",
            data={
                "session_id": "sess-003", "message_count": 1
            },
        ),
    ]
    return events


# ---------------------------------------------------------------------------
# EXMP-04: Anthropic Messages — document-summarize with tool_use
# ---------------------------------------------------------------------------


def generate_exmp04() -> list[Event]:
    """Claude summarizes a document via a ``read_file`` tool call.

    Demonstrates Anthropic Messages API payload shapes:
    - **Input string content** (turn 1 user message): ``content`` is a
      plain string (the simple form).
    - **Output typed blocks** (turn 1 assistant): ``content`` is a list
      of typed blocks containing ``text`` + ``tool_use``.
    - **Mixed input on turn 2**: ``messages`` includes the assistant's
      prior turn (with ``tool_use`` blocks) and a fresh user turn with
      ``tool_result`` blocks (Anthropic's transport for tool returns).
      The Anthropic extractor's ``_anthropic_normalize_input_messages``
      hook drops both — assistant turns aren't user-facing, and tool
      returns are captured by the tool scope events.

    Eight events: one paired agent + two paired LLM turns + one paired
    tool. The tool's ``category_profile.tool_call_id`` matches the
    Anthropic ``tool_use.id`` so observation reconciliation works.
    """
    tu_id = "toolu_01abc"
    file_content = "Title: Intro\nThis project is an end-to-end demo."

    events: list[Event] = [
        ScopeEvent(
            scope_category="start",
            uuid="agent-004",
            parent_uuid=None,
            timestamp=_ts(4, 0),
            name="claude_summarizer",
            attributes=[],
            category="agent",
            data={"input": "Summarize the document at /docs/intro.md"},
        ),
        ScopeEvent(
            scope_category="start",
            uuid="llm-004-1",
            parent_uuid="agent-004",
            timestamp=_ts(4, 1),
            name="claude-3-5-sonnet",
            attributes=[],
            category="llm",
            category_profile={"model_name": "claude-3-5-sonnet"},
            data={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{
                    "role": "user", "content": "Summarize the document at /docs/intro.md"
                }, ],
            },
            data_schema=_ANTHROPIC_MESSAGES_SCHEMA,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="llm-004-1",
            parent_uuid="agent-004",
            timestamp=_ts(4, 2),
            name="claude-3-5-sonnet",
            attributes=[],
            category="llm",
            category_profile={"model_name": "claude-3-5-sonnet"},
            data={
                "id": "msg_01xyz",
                "role": "assistant",
                "model": "claude-3-5-sonnet-20241022",
                "content": [
                    {
                        "type": "text", "text": "Let me read that file for you."
                    },
                    {
                        "type": "tool_use",
                        "id": tu_id,
                        "name": "read_file",
                        "input": {
                            "path": "/docs/intro.md"
                        },
                    },
                ],
                "stop_reason": "tool_use",
            },
            data_schema=_ANTHROPIC_MESSAGES_SCHEMA,
        ),
        ScopeEvent(
            scope_category="start",
            uuid="tool-004",
            parent_uuid="agent-004",
            timestamp=_ts(4, 3),
            name="read_file",
            attributes=["remote"],
            category="tool",
            category_profile={"tool_call_id": tu_id},
            data={"path": "/docs/intro.md"},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="tool-004",
            parent_uuid="agent-004",
            timestamp=_ts(4, 4),
            name="read_file",
            attributes=["remote"],
            category="tool",
            category_profile={"tool_call_id": tu_id},
            data={"result": file_content},
        ),
        ScopeEvent(
            scope_category="start",
            uuid="llm-004-2",
            parent_uuid="agent-004",
            timestamp=_ts(4, 5),
            name="claude-3-5-sonnet",
            attributes=[],
            category="llm",
            category_profile={"model_name": "claude-3-5-sonnet"},
            data={
                "model":
                    "claude-3-5-sonnet-20241022",
                "messages": [
                    {
                        "role": "user", "content": "Summarize the document at /docs/intro.md"
                    },
                    {
                        "role":
                            "assistant",
                        "content": [
                            {
                                "type": "text", "text": "Let me read that file for you."
                            },
                            {
                                "type": "tool_use",
                                "id": tu_id,
                                "name": "read_file",
                                "input": {
                                    "path": "/docs/intro.md"
                                },
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tu_id,
                            "content": file_content,
                        }, ],
                    },
                ],
            },
            data_schema=_ANTHROPIC_MESSAGES_SCHEMA,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="llm-004-2",
            parent_uuid="agent-004",
            timestamp=_ts(4, 6),
            name="claude-3-5-sonnet",
            attributes=[],
            category="llm",
            category_profile={"model_name": "claude-3-5-sonnet"},
            data={
                "id": "msg_02xyz",
                "role": "assistant",
                "model": "claude-3-5-sonnet-20241022",
                "content": [{
                    "type": "text",
                    "text": "The document is the introduction page for an end-to-end demo project.",
                }, ],
                "stop_reason": "end_turn",
            },
            data_schema=_ANTHROPIC_MESSAGES_SCHEMA,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="agent-004",
            parent_uuid=None,
            timestamp=_ts(4, 7),
            name="claude_summarizer",
            attributes=[],
            category="agent",
            data={
                "response": "The document is the introduction page for an end-to-end demo project.",
            },
        ),
    ]
    return events


# ---------------------------------------------------------------------------
# EXMP-05: Gemini generateContent — timezone lookup with functionCall
# ---------------------------------------------------------------------------


def generate_exmp05() -> list[Event]:
    """Gemini answers a timezone question via a ``get_current_time`` function call.

    Demonstrates Gemini generateContent payload shapes:
    - **Input parts list** (turn 1): ``contents[].parts[]`` with a single
      ``{text}`` part for the user's question.
    - **Output candidate** (turn 1 response): ``candidates[0].content.parts[]``
      mixing a ``{text}`` part and a ``{functionCall}`` part.
    - **Multi-turn input** (turn 2 request): ``contents`` includes prior
      ``role: "model"`` turn with the functionCall, and a ``role: "user"``
      turn with a ``functionResponse`` part. The Gemini extractor's
      ``_gemini_normalize_input_messages`` hook drops both (no text →
      no user step emitted).
    - **Role aliasing**: Gemini uses ``"model"`` rather than
      ``"assistant"``; the extractor's ``role_aliases`` map normalises it.
    - **Synthesized tool_call_id**: Gemini doesn't provide an ID with
      ``functionCall``; the extractor synthesizes ``name__index``
      (e.g. ``get_current_time__1``) for ATIF observation reconciliation.

    Eight events.
    """
    synthesized_tc_id = "get_current_time__1"
    iso_now = "2026-04-30T15:30:00+09:00"

    events: list[Event] = [
        ScopeEvent(
            scope_category="start",
            uuid="agent-005",
            parent_uuid=None,
            timestamp=_ts(5, 0),
            name="gemini_assistant",
            attributes=[],
            category="agent",
            data={"input": "What time is it in Tokyo right now?"},
        ),
        ScopeEvent(
            scope_category="start",
            uuid="llm-005-1",
            parent_uuid="agent-005",
            timestamp=_ts(5, 1),
            name="gemini-2.0-flash",
            attributes=[],
            category="llm",
            category_profile={"model_name": "gemini-2.0-flash"},
            data={
                "contents": [{
                    "role": "user", "parts": [{
                        "text": "What time is it in Tokyo right now?"
                    }]
                }, ],
            },
            data_schema=_GEMINI_GENERATE_CONTENT_SCHEMA,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="llm-005-1",
            parent_uuid="agent-005",
            timestamp=_ts(5, 2),
            name="gemini-2.0-flash",
            attributes=[],
            category="llm",
            category_profile={"model_name": "gemini-2.0-flash"},
            data={
                "candidates": [{
                    "content": {
                        "role":
                            "model",
                        "parts": [
                            {
                                "text": "Let me check the current time in Tokyo. "
                            },
                            {
                                "functionCall": {
                                    "name": "get_current_time",
                                    "args": {
                                        "timezone": "Asia/Tokyo"
                                    },
                                },
                            },
                        ],
                    },
                    "finishReason": "STOP",
                }, ],
            },
            data_schema=_GEMINI_GENERATE_CONTENT_SCHEMA,
        ),
        ScopeEvent(
            scope_category="start",
            uuid="tool-005",
            parent_uuid="agent-005",
            timestamp=_ts(5, 3),
            name="get_current_time",
            attributes=[],
            category="tool",
            category_profile={"tool_call_id": synthesized_tc_id},
            data={"timezone": "Asia/Tokyo"},
        ),
        ScopeEvent(
            scope_category="end",
            uuid="tool-005",
            parent_uuid="agent-005",
            timestamp=_ts(5, 4),
            name="get_current_time",
            attributes=[],
            category="tool",
            category_profile={"tool_call_id": synthesized_tc_id},
            data={"result": iso_now},
        ),
        ScopeEvent(
            scope_category="start",
            uuid="llm-005-2",
            parent_uuid="agent-005",
            timestamp=_ts(5, 5),
            name="gemini-2.0-flash",
            attributes=[],
            category="llm",
            category_profile={"model_name": "gemini-2.0-flash"},
            data={
                "contents": [
                    {
                        "role": "user", "parts": [{
                            "text": "What time is it in Tokyo right now?"
                        }]
                    },
                    {
                        "role":
                            "model",
                        "parts": [{
                            "functionCall": {
                                "name": "get_current_time",
                                "args": {
                                    "timezone": "Asia/Tokyo"
                                },
                            },
                        }, ],
                    },
                    {
                        "role":
                            "user",
                        "parts": [{
                            "functionResponse": {
                                "name": "get_current_time",
                                "response": {
                                    "result": iso_now
                                },
                            },
                        }, ],
                    },
                ],
            },
            data_schema=_GEMINI_GENERATE_CONTENT_SCHEMA,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="llm-005-2",
            parent_uuid="agent-005",
            timestamp=_ts(5, 6),
            name="gemini-2.0-flash",
            attributes=[],
            category="llm",
            category_profile={"model_name": "gemini-2.0-flash"},
            data={
                "candidates": [{
                    "content": {
                        "role":
                            "model",
                        "parts": [{
                            "text": "It's currently 3:30 PM on April 30, 2026 in Tokyo (Japan Standard Time).",
                        }, ],
                    },
                    "finishReason": "STOP",
                }, ],
            },
            data_schema=_GEMINI_GENERATE_CONTENT_SCHEMA,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="agent-005",
            parent_uuid=None,
            timestamp=_ts(5, 7),
            name="gemini_assistant",
            attributes=[],
            category="agent",
            data={
                "response": "It's currently 3:30 PM on April 30, 2026 in Tokyo (Japan Standard Time).",
            },
        ),
    ]
    return events


# ---------------------------------------------------------------------------
# EXMP-06: Heterogeneous router — three LLM providers in one trajectory
# ---------------------------------------------------------------------------


def generate_exmp06() -> list[Event]:
    """An orchestrator that routes a multi-part request to specialist LLMs
    from three different providers in one trajectory.

    User query has two parts: (a) write a Python factorial function,
    (b) compute 2^32. The orchestrator dispatches:

    1. **OpenAI gpt-4o (router)** — receives the full query, decides the
       routing plan (plain-text reasoning, no tool calls).
    2. **Anthropic claude-3-5-sonnet (code specialist)** — receives just
       the code task, returns code (text only).
    3. **Gemini gemini-2.0-flash (math specialist)** — receives just the
       math task, returns the answer (text only).

    The ATOF stream contains three LLM scope events whose ``data_schema``
    declares ``openai/chat-completions@1``, ``anthropic/messages@1``, and
    ``gemini/generate-content@1`` respectively. The converter dispatches
    per-event based on this declaration — the strongest end-to-end
    evidence that the schema-map architecture handles heterogeneous
    streams without producer-side coordination.

    Eight events: paired orchestrator agent + three paired LLM turns.
    No tool scopes (each specialist returns a plain-text response).
    """
    user_query = "Two things: (1) write a Python function for factorial, and (2) tell me what 2^32 equals."
    code_answer = "```python\ndef factorial(n: int) -> int:\n    return 1 if n <= 1 else n * factorial(n - 1)\n```"
    math_answer = "2^32 = 4,294,967,296"

    events: list[Event] = [
        ScopeEvent(
            scope_category="start",
            uuid="orchestrator-006",
            parent_uuid=None,
            timestamp=_ts(6, 0),
            name="multi_provider_router",
            attributes=[],
            category="agent",
            data={"input": user_query},
        ),
        # --- 1. OpenAI router -----------------------------------------------
        ScopeEvent(
            scope_category="start",
            uuid="llm-006-router",
            parent_uuid="orchestrator-006",
            timestamp=_ts(6, 1),
            name="gpt-4o-router",
            attributes=[],
            category="llm",
            category_profile={"model_name": "gpt-4o"},
            data={
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a router. Decide which specialist handles each part of the user's request.",
                    },
                    {
                        "role": "user", "content": user_query
                    },
                ],
            },
            data_schema=_OPENAI_CHAT_SCHEMA,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="llm-006-router",
            parent_uuid="orchestrator-006",
            timestamp=_ts(6, 2),
            name="gpt-4o-router",
            attributes=[],
            category="llm",
            category_profile={"model_name": "gpt-4o"},
            data={
                "content": ("Plan: send the factorial-code task to claude-3-5-sonnet "
                            "(strong code synthesis), and the 2^32 arithmetic to "
                            "gemini-2.0-flash. I'll combine the responses.")
            },
            data_schema=_OPENAI_CHAT_SCHEMA,
        ),
        # --- 2. Anthropic code specialist -----------------------------------
        ScopeEvent(
            scope_category="start",
            uuid="llm-006-code",
            parent_uuid="orchestrator-006",
            timestamp=_ts(6, 3),
            name="claude-3-5-sonnet",
            attributes=[],
            category="llm",
            category_profile={"model_name": "claude-3-5-sonnet"},
            data={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{
                    "role": "user",
                    "content": "Write a Python function for factorial.",
                }, ],
            },
            data_schema=_ANTHROPIC_MESSAGES_SCHEMA,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="llm-006-code",
            parent_uuid="orchestrator-006",
            timestamp=_ts(6, 4),
            name="claude-3-5-sonnet",
            attributes=[],
            category="llm",
            category_profile={"model_name": "claude-3-5-sonnet"},
            data={
                "id": "msg_006code",
                "role": "assistant",
                "model": "claude-3-5-sonnet-20241022",
                "content": [{
                    "type": "text", "text": code_answer
                }],
                "stop_reason": "end_turn",
            },
            data_schema=_ANTHROPIC_MESSAGES_SCHEMA,
        ),
        # --- 3. Gemini math specialist --------------------------------------
        ScopeEvent(
            scope_category="start",
            uuid="llm-006-math",
            parent_uuid="orchestrator-006",
            timestamp=_ts(6, 5),
            name="gemini-2.0-flash",
            attributes=[],
            category="llm",
            category_profile={"model_name": "gemini-2.0-flash"},
            data={
                "contents": [{
                    "role": "user", "parts": [{
                        "text": "What is 2^32?"
                    }]
                }, ],
            },
            data_schema=_GEMINI_GENERATE_CONTENT_SCHEMA,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="llm-006-math",
            parent_uuid="orchestrator-006",
            timestamp=_ts(6, 6),
            name="gemini-2.0-flash",
            attributes=[],
            category="llm",
            category_profile={"model_name": "gemini-2.0-flash"},
            data={
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{
                            "text": math_answer
                        }],
                    },
                    "finishReason": "STOP",
                }, ],
            },
            data_schema=_GEMINI_GENERATE_CONTENT_SCHEMA,
        ),
        ScopeEvent(
            scope_category="end",
            uuid="orchestrator-006",
            parent_uuid=None,
            timestamp=_ts(6, 7),
            name="multi_provider_router",
            attributes=[],
            category="agent",
            data={
                "response": (f"Here's both:\n\n{code_answer}\n\nAnd: {math_answer}"),
            },
        ),
    ]
    return events


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for the generated JSONL files (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    scenarios = [
        ("exmp01_atof.jsonl", "tier-1 raw pass-through", generate_exmp01),
        ("exmp02_atof.jsonl", "tier-2 semantic-tagged", generate_exmp02),
        ("exmp03_atof.jsonl", "mark events", generate_exmp03),
        ("exmp04_atof.jsonl", "Anthropic Messages — tool_use", generate_exmp04),
        ("exmp05_atof.jsonl", "Gemini generateContent — functionCall", generate_exmp05),
        ("exmp06_atof.jsonl", "heterogeneous router (3 providers)", generate_exmp06),
    ]

    for filename, label, generator in scenarios:
        events = generator()
        path = args.output_dir / filename
        write_jsonl(events, path)
        print(f"Wrote {len(events)} events ({label}) to {path}")


if __name__ == "__main__":
    main()
