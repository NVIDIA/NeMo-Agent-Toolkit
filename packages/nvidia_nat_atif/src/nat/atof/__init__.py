# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for the Agentic Trajectory Observability Format (ATOF).

ATOF is a JSON-Lines wire format for agent runtime event streams.
These models define the 7 event types, attribute flags, and codec schemas
as standalone Pydantic models.

See ``atof-event-format.md`` for the full specification.
"""

from nat.atof.attributes import LLMAttributes
from nat.atof.attributes import ScopeAttributes
from nat.atof.attributes import ToolAttributes
from nat.atof.codec import AnnotatedLLMRequest
from nat.atof.codec import AnnotatedLLMResponse
from nat.atof.codec import ContentPart as CodecContentPart
from nat.atof.codec import GenerationParams
from nat.atof.codec import Message
from nat.atof.codec import RequestToolCall
from nat.atof.codec import ResponseToolCall
from nat.atof.codec import ToolDefinition
from nat.atof.codec import Usage
from nat.atof.events import Event
from nat.atof.events import LLMEndEvent
from nat.atof.events import LLMStartEvent
from nat.atof.events import MarkEvent
from nat.atof.events import ScopeEndEvent
from nat.atof.events import ScopeStartEvent
from nat.atof.events import ToolEndEvent
from nat.atof.events import ToolStartEvent
from nat.atof.io import read_jsonl
from nat.atof.io import write_jsonl
from nat.atof.scope_type import ScopeType

__all__ = [
    "AnnotatedLLMRequest",
    "AnnotatedLLMResponse",
    "CodecContentPart",
    "Event",
    "GenerationParams",
    "LLMAttributes",
    "LLMEndEvent",
    "LLMStartEvent",
    "MarkEvent",
    "Message",
    "RequestToolCall",
    "ResponseToolCall",
    "ScopeAttributes",
    "ScopeEndEvent",
    "ScopeStartEvent",
    "ScopeType",
    "ToolAttributes",
    "ToolDefinition",
    "ToolEndEvent",
    "ToolStartEvent",
    "Usage",
    "read_jsonl",
    "write_jsonl",
]
