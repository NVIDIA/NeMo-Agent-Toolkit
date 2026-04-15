# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for the Agentic Trajectory Observability Format (ATOF).

ATOF is a JSON-Lines wire format for agent runtime event streams. These
models define the three event kinds (``ScopeStartEvent``, ``ScopeEndEvent``,
``MarkEvent``), the structured error payload (``ErrorInfo``), the behavioral
flag enum (``Flags``), the canonical ``scope_type`` vocabulary (``ScopeType``),
and the codec-annotated LLM request/response types (``AnnotatedLLMRequest``,
``AnnotatedLLMResponse`` and their components).

See ``atof-event-format.md`` for the core wire format and
``atof-codec-profiles.md`` for the codec-annotation layer.
"""

from nat.atof.codec import AnnotatedLLMRequest
from nat.atof.codec import AnnotatedLLMResponse
from nat.atof.codec import ContentPart as CodecContentPart
from nat.atof.codec import GenerationParams
from nat.atof.codec import Message
from nat.atof.codec import RequestToolCall
from nat.atof.codec import ResponseToolCall
from nat.atof.codec import ToolDefinition
from nat.atof.codec import Usage
from nat.atof.events import ErrorInfo
from nat.atof.events import Event
from nat.atof.events import MarkEvent
from nat.atof.events import ScopeEndEvent
from nat.atof.events import ScopeStartEvent
from nat.atof.flags import Flags
from nat.atof.io import read_jsonl
from nat.atof.io import write_jsonl
from nat.atof.scope_type import ScopeType

__all__ = [
    "AnnotatedLLMRequest",
    "AnnotatedLLMResponse",
    "CodecContentPart",
    "ErrorInfo",
    "Event",
    "Flags",
    "GenerationParams",
    "MarkEvent",
    "Message",
    "RequestToolCall",
    "ResponseToolCall",
    "ScopeEndEvent",
    "ScopeStartEvent",
    "ScopeType",
    "ToolDefinition",
    "Usage",
    "read_jsonl",
    "write_jsonl",
]
