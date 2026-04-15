# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for the Agentic Trajectory Observability Format (ATOF).

ATOF is the JSON-Lines wire format for NeMo-Flow subscriber callbacks.
These models mirror the NeMo-Flow Rust event types as standalone Pydantic
models with no NeMo-Flow runtime dependency.

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
    # Event types
    "Event",
    "ScopeStartEvent",
    "ScopeEndEvent",
    "LLMStartEvent",
    "LLMEndEvent",
    "ToolStartEvent",
    "ToolEndEvent",
    "MarkEvent",  # Enums and flags
    "ScopeType",
    "ScopeAttributes",
    "LLMAttributes",
    "ToolAttributes",  # Codec types
    "AnnotatedLLMRequest",
    "AnnotatedLLMResponse",
    "CodecContentPart",
    "GenerationParams",
    "Message",
    "RequestToolCall",
    "ResponseToolCall",
    "ToolDefinition",
    "Usage",  # I/O
    "read_jsonl",
    "write_jsonl",
]
