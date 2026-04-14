# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for the Agentic Trajectory Observability Format (ATOF).

ATOF is a JSON-Lines wire format for agent runtime event streams. These models
define the 4 event types (ScopeStart, ScopeEnd, Mark, StreamHeaderEvent), the
shared behavioral ``Flags`` enum, the :class:`ProfileContract` base, and the
two reference profile implementations (``DefaultLlmV1``, ``DefaultToolV1``).

See ``atof-event-format.md`` v0.2 for the full specification.

NOTE: this file is in a transient state during phase 08 (ATOF v0.2 refactor).
Plan 08-03 removes the v0.1 typed profile re-exports (``ToolProfile``,
``LLMProfile``, ``CustomProfile``) and the ``ScopeType`` enum re-export. Plan
08-04 will expand this module to export the full v0.2 public API including
``StreamHeaderEvent``, ``ProfileContract``, ``DefaultLlmV1``, ``DefaultToolV1``,
and the ``validate_profile`` helper.
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
    "ToolDefinition",
    "Usage",
    "read_jsonl",
    "write_jsonl",
]
