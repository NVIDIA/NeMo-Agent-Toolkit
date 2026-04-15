# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ATOF event models for the 7 NeMo-Flow event types.

Mirrors the Rust event structs in ``crates/core/src/types/event.rs`` as
standalone Pydantic models. The ``Event`` type is a discriminated union
keyed on the ``kind`` field.

See ATOF spec Sections 2–3.
"""

from __future__ import annotations

from typing import Annotated
from typing import Any
from typing import Literal
from typing import Union

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Discriminator
from pydantic import Field
from pydantic import Tag

from nat.atof.attributes import LLMAttributes
from nat.atof.attributes import ScopeAttributes
from nat.atof.attributes import ToolAttributes
from nat.atof.codec import AnnotatedLLMRequest
from nat.atof.codec import AnnotatedLLMResponse
from nat.atof.scope_type import ScopeType


# ---------------------------------------------------------------------------
# Base fields shared by all event types (Section 2)
# ---------------------------------------------------------------------------


class _EventBase(BaseModel):
    """Common fields shared by all 7 ATOF event types."""

    uuid: str = Field(description="Unique handle identifier (v7 UUID)")
    parent_uuid: str | None = Field(default=None, description="UUID of parent scope")
    timestamp: str = Field(description="Wall-clock time (RFC 3339)")
    name: str = Field(description="Human-readable label for this handle")
    data: Any | None = Field(default=None, description="Application-specific payload")
    metadata: dict[str, Any] | None = Field(default=None, description="Tracing metadata")

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Event types (Section 3)
# ---------------------------------------------------------------------------


class ScopeStartEvent(_EventBase):
    """Emitted when a scope is pushed onto the scope stack (Section 3.1)."""

    kind: Literal["ScopeStart"] = "ScopeStart"
    attributes: int = Field(default=0, description="ScopeAttributes bitflags")
    scope_type: str = Field(description="Scope type enum value")


class ScopeEndEvent(_EventBase):
    """Emitted when a scope is popped from the scope stack (Section 3.2)."""

    kind: Literal["ScopeEnd"] = "ScopeEnd"
    attributes: int = Field(default=0, description="ScopeAttributes bitflags")
    scope_type: str = Field(description="Same as matching ScopeStartEvent")


class LLMStartEvent(_EventBase):
    """Emitted when an LLM call begins (Section 3.3)."""

    kind: Literal["LLMStart"] = "LLMStart"
    attributes: int = Field(default=0, description="LLMAttributes bitflags")
    input: Any | None = Field(default=None, description="Post-sanitize LLM request payload")
    model_name: str | None = Field(default=None, description="Model identifier")
    annotated_request: AnnotatedLLMRequest | None = Field(
        default=None, description="Codec-decoded request (if codec registered)"
    )


class LLMEndEvent(_EventBase):
    """Emitted when an LLM call completes (Section 3.4)."""

    kind: Literal["LLMEnd"] = "LLMEnd"
    attributes: int = Field(default=0, description="Same flags as matching LLMStartEvent")
    output: Any | None = Field(default=None, description="Post-sanitize LLM response payload")
    model_name: str | None = Field(default=None, description="Model identifier")
    annotated_response: AnnotatedLLMResponse | None = Field(
        default=None, description="Codec-decoded response (if codec registered)"
    )


class ToolStartEvent(_EventBase):
    """Emitted when a tool invocation begins (Section 3.5)."""

    kind: Literal["ToolStart"] = "ToolStart"
    attributes: int = Field(default=0, description="ToolAttributes bitflags")
    input: Any | None = Field(default=None, description="Post-sanitize tool input arguments")
    tool_call_id: str | None = Field(
        default=None, description="Correlation ID from LLM tool-call response"
    )


class ToolEndEvent(_EventBase):
    """Emitted when a tool invocation completes (Section 3.6)."""

    kind: Literal["ToolEnd"] = "ToolEnd"
    attributes: int = Field(default=0, description="Same flags as matching ToolStartEvent")
    output: Any | None = Field(default=None, description="Post-sanitize tool result")
    tool_call_id: str | None = Field(
        default=None, description="Same as matching ToolStartEvent"
    )


class MarkEvent(_EventBase):
    """Emitted for a named checkpoint in the event stream (Section 3.7)."""

    kind: Literal["Mark"] = "Mark"


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------


def _get_event_kind(v: Any) -> str:
    """Extract discriminator value from raw dict or model instance."""
    if isinstance(v, dict):
        return v.get("kind", "")
    return getattr(v, "kind", "")


Event = Annotated[
    Union[
        Annotated[ScopeStartEvent, Tag("ScopeStart")],
        Annotated[ScopeEndEvent, Tag("ScopeEnd")],
        Annotated[LLMStartEvent, Tag("LLMStart")],
        Annotated[LLMEndEvent, Tag("LLMEnd")],
        Annotated[ToolStartEvent, Tag("ToolStart")],
        Annotated[ToolEndEvent, Tag("ToolEnd")],
        Annotated[MarkEvent, Tag("Mark")],
    ],
    Discriminator(_get_event_kind),
]
"""Discriminated union of all 7 ATOF event types, keyed on ``kind``."""
