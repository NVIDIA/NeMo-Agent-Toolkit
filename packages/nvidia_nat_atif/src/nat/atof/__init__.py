# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for the Agentic Trajectory Observability Format (ATOF).

ATOF is a JSON-Lines wire format for agent runtime event streams. These models
define the four event types — the 4-kind event model of ``ScopeStartEvent``,
``ScopeEndEvent``, ``MarkEvent``, and ``StreamHeaderEvent`` — along with the
profile-contract base (``ProfileContract``) and its two reference
implementations (``DefaultLlmV1`` for ``default/llm.v1`` and ``DefaultToolV1``
for ``default/tool.v1``), the ``validate_profile`` Draft 2020-12 JSON Schema
helper, and the shared behavioral ``Flags`` enum.

The ``ScopeType`` name is retained as a documentation-only type alias
(``ScopeType = str``); ``scope_type`` on events is an open-vocabulary string
(spec §3.1). The v0.1 typed profile classes (``ToolProfile``, ``LLMProfile``,
``CustomProfile``) were removed in phase 8 — importing them from this package
raises ``ImportError`` (pure break per D-21, no shims).

See ``atof-event-format.md`` v0.2 for the full specification, in particular
§§4–6 (Profile Contract Protocol, Stream Header Event, Reference Profile
Implementations).
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
from nat.atof.events import StreamHeaderEvent
from nat.atof.flags import Flags
from nat.atof.io import read_jsonl
from nat.atof.io import write_jsonl
from nat.atof.profile_contract import ProfileContract
from nat.atof.profiles import DefaultLlmV1
from nat.atof.profiles import DefaultToolV1
from nat.atof.scope_type import ScopeType
from nat.atof.validation import validate_profile

__all__ = [
    "AnnotatedLLMRequest",
    "AnnotatedLLMResponse",
    "CodecContentPart",
    "DefaultLlmV1",
    "DefaultToolV1",
    "ErrorInfo",
    "Event",
    "Flags",
    "GenerationParams",
    "MarkEvent",
    "Message",
    "ProfileContract",
    "RequestToolCall",
    "ResponseToolCall",
    "ScopeEndEvent",
    "ScopeStartEvent",
    "ScopeType",
    "StreamHeaderEvent",
    "ToolDefinition",
    "Usage",
    "read_jsonl",
    "validate_profile",
    "write_jsonl",
]
