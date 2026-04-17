# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ATOF event models for the 7 agent runtime event types.

Standalone Pydantic models for each event kind. The ``Event`` type is a
discriminated union keyed on the ``kind`` field.

See ATOF spec Sections 2-3.
"""

from __future__ import annotations

from typing import Annotated
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Discriminator
from pydantic import Field
from pydantic import Tag
from pydantic import field_validator

from nat.atof.codec import AnnotatedLLMRequest
from nat.atof.codec import AnnotatedLLMResponse


def _canonicalize_attributes(v: Any) -> list[str]:
    """Normalize an ``attributes`` field to a sorted, deduplicated list of strings.

    Accepts either a list of strings or :class:`enum.StrEnum` members. Unknown
    flag names are preserved — the spec requires consumers to round-trip them.
    """
    if v is None:
        return []
    if not isinstance(v, (list, tuple, set)):
        raise TypeError(f"attributes must be a list of strings, got {type(v).__name__}")
    normalized: set[str] = set()
    for item in v:
        if not isinstance(item, str):
            raise TypeError(f"attributes entries must be strings, got {type(item).__name__}")
        normalized.add(str(item))
    return sorted(normalized)


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
    attributes: list[str] = Field(default_factory=list,
                                  description="Canonical ScopeAttributes flag names (sorted, deduped)")
    scope_type: str = Field(description="Scope type enum value")
    input: Any | None = Field(default=None, description="Post-sanitize scope input payload (optional)")

    _normalize_attributes = field_validator("attributes", mode="before")(_canonicalize_attributes)


class ScopeEndEvent(_EventBase):
    """Emitted when a scope is popped from the scope stack (Section 3.2)."""

    kind: Literal["ScopeEnd"] = "ScopeEnd"
    attributes: list[str] = Field(default_factory=list, description="Same flags as matching ScopeStartEvent")
    scope_type: str = Field(description="Same as matching ScopeStartEvent")
    output: Any | None = Field(default=None, description="Post-sanitize scope output payload (optional)")

    _normalize_attributes = field_validator("attributes", mode="before")(_canonicalize_attributes)


class LLMStartEvent(_EventBase):
    """Emitted when an LLM call begins (Section 3.3)."""

    kind: Literal["LLMStart"] = "LLMStart"
    attributes: list[str] = Field(default_factory=list,
                                  description="Canonical LLMAttributes flag names (sorted, deduped)")
    input: Any | None = Field(default=None, description="Post-sanitize LLM request payload")
    model_name: str | None = Field(default=None, description="Model identifier")
    annotated_request: AnnotatedLLMRequest | None = Field(default=None,
                                                          description="Codec-decoded request (if codec registered)")

    _normalize_attributes = field_validator("attributes", mode="before")(_canonicalize_attributes)


class LLMEndEvent(_EventBase):
    """Emitted when an LLM call completes (Section 3.4)."""

    kind: Literal["LLMEnd"] = "LLMEnd"
    attributes: list[str] = Field(default_factory=list, description="Same flags as matching LLMStartEvent")
    output: Any | None = Field(default=None, description="Post-sanitize LLM response payload")
    model_name: str | None = Field(default=None, description="Model identifier")
    annotated_response: AnnotatedLLMResponse | None = Field(default=None,
                                                            description="Codec-decoded response (if codec registered)")

    _normalize_attributes = field_validator("attributes", mode="before")(_canonicalize_attributes)


class ToolStartEvent(_EventBase):
    """Emitted when a tool invocation begins (Section 3.5)."""

    kind: Literal["ToolStart"] = "ToolStart"
    attributes: list[str] = Field(default_factory=list,
                                  description="Canonical ToolAttributes flag names (sorted, deduped)")
    input: Any | None = Field(default=None, description="Post-sanitize tool input arguments")
    tool_call_id: str | None = Field(default=None, description="Correlation ID from LLM tool-call response")

    _normalize_attributes = field_validator("attributes", mode="before")(_canonicalize_attributes)


class ToolEndEvent(_EventBase):
    """Emitted when a tool invocation completes (Section 3.6)."""

    kind: Literal["ToolEnd"] = "ToolEnd"
    attributes: list[str] = Field(default_factory=list, description="Same flags as matching ToolStartEvent")
    output: Any | None = Field(default=None, description="Post-sanitize tool result")
    tool_call_id: str | None = Field(default=None, description="Same as matching ToolStartEvent")

    _normalize_attributes = field_validator("attributes", mode="before")(_canonicalize_attributes)


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
    Annotated[ScopeStartEvent, Tag("ScopeStart")] | Annotated[ScopeEndEvent, Tag("ScopeEnd")]
    | Annotated[LLMStartEvent, Tag("LLMStart")] | Annotated[LLMEndEvent, Tag("LLMEnd")]
    | Annotated[ToolStartEvent, Tag("ToolStart")] | Annotated[ToolEndEvent, Tag("ToolEnd")]
    | Annotated[MarkEvent, Tag("Mark")],
    Discriminator(_get_event_kind),
]
"""Discriminated union of all 7 ATOF event types, keyed on ``kind``."""
