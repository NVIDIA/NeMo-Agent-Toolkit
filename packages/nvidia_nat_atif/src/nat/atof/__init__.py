# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for the Agentic Trajectory Observability Format (ATOF).

ATOF is a JSON-Lines wire format for agent runtime event streams. These
models define the four event kinds (``ScopeStartEvent``, ``ScopeEndEvent``,
``MarkEvent``, ``StreamHeaderEvent``), the structured error payload
(``ErrorInfo``), the behavioral flag enum (``Flags``), the canonical
``scope_type`` vocabulary (``ScopeType``), and the schema-annotated LLM
request/response types (``AnnotatedLLMRequest``, ``AnnotatedLLMResponse``
and their components).

See ``atof-event-format.md`` for the core wire format and
``atof-schema-profiles.md`` for the schema-annotation layer + 4-priority
schema resolution protocol (§6).
"""

from nat.atof.annotations import AnnotatedLLMRequest
from nat.atof.annotations import AnnotatedLLMResponse
from nat.atof.annotations import ContentPart
from nat.atof.annotations import GenerationParams
from nat.atof.annotations import Message
from nat.atof.annotations import RequestToolCall
from nat.atof.annotations import ResponseToolCall
from nat.atof.annotations import ToolDefinition
from nat.atof.annotations import Usage
from nat.atof.events import ErrorInfo
from nat.atof.events import Event
from nat.atof.events import MarkEvent
from nat.atof.events import ScopeEndEvent
from nat.atof.events import ScopeStartEvent
from nat.atof.events import StreamHeaderEvent
from nat.atof.flags import Flags
from nat.atof.io import read_jsonl
from nat.atof.io import write_jsonl
from nat.atof.scope_type import ScopeType

__all__ = [
    "AnnotatedLLMRequest",
    "AnnotatedLLMResponse",
    "ContentPart",
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
    "StreamHeaderEvent",
    "ToolDefinition",
    "Usage",
    "read_jsonl",
    "write_jsonl",
]
