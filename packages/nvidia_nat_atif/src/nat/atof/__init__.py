# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for the Agentic Trajectory Observability Format (ATOF).

ATOF is a JSON-Lines wire format for agent runtime event streams.
These models define the 3 event types (ScopeStart, ScopeEnd, Mark), the
shared behavioral ``Flags`` enum, the typed ``ScopeType``-specific profile
classes, and the codec schemas as standalone Pydantic models.

See ``atof-event-format.md`` v0.1 for the full specification.
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
from nat.atof.profiles import CustomProfile
from nat.atof.profiles import LLMProfile
from nat.atof.profiles import ToolProfile
from nat.atof.scope_type import ScopeType

__all__ = [
    "AnnotatedLLMRequest",
    "AnnotatedLLMResponse",
    "CodecContentPart",
    "CustomProfile",
    "ErrorInfo",
    "Event",
    "Flags",
    "GenerationParams",
    "LLMProfile",
    "MarkEvent",
    "Message",
    "RequestToolCall",
    "ResponseToolCall",
    "ScopeEndEvent",
    "ScopeStartEvent",
    "ScopeType",
    "ToolDefinition",
    "ToolProfile",
    "Usage",
    "read_jsonl",
    "write_jsonl",
]
