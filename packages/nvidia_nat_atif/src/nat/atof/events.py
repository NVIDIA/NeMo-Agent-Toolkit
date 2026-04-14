# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ATOF event models for the 3 event types (ScopeStart, ScopeEnd, Mark) per spec v0.1.

Standalone Pydantic models for each event kind. The ``Event`` type is a
discriminated union keyed on the ``kind`` field.

Spec v0.1 collapses the seven-kind event model (ScopeStart/End, LLMStart/End,
ToolStart/End, Mark) into three kinds (ScopeStart, ScopeEnd, Mark) with
subject-specific fields carried in a typed ``profile`` sub-schema discriminated
by ``scope_type``. See ATOF spec §2 (common fields), §3 (event kinds), §4
(scope profiles), §5.1 (timestamp), §5.6 (status), §5.7 (schema_version).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Discriminator
from pydantic import Field
from pydantic import Tag
from pydantic import computed_field
from pydantic import field_validator
from pydantic import model_validator

from nat.atof.flags import Flags  # noqa: F401  (re-exported for convenience)
from nat.atof.profiles import CustomProfile
from nat.atof.profiles import LLMProfile
from nat.atof.profiles import ToolProfile
from nat.atof.scope_type import ScopeType

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_SCHEMA_VERSION_PATTERN = re.compile(r"^0\.\d+$")

_NULL_PROFILE_SCOPE_TYPES: frozenset[ScopeType] = frozenset({
    ScopeType.AGENT,
    ScopeType.FUNCTION,
    ScopeType.RETRIEVER,
    ScopeType.EMBEDDER,
    ScopeType.RERANKER,
    ScopeType.GUARDRAIL,
    ScopeType.EVALUATOR,
})

# ---------------------------------------------------------------------------
# Flag canonicalization helper
# ---------------------------------------------------------------------------


def _canonicalize_flags(v: Any) -> list[str]:
    """Normalize a ``flags`` field to a sorted, deduplicated list of strings.

    Accepts either a list of strings or :class:`enum.StrEnum` members. Unknown
    flag names are preserved -- the spec requires consumers to round-trip them.
    """
    if v is None:
        return []
    if not isinstance(v, (list, tuple, set)):
        raise TypeError(f"flags must be a list of strings, got {type(v).__name__}")
    normalized: set[str] = set()
    for item in v:
        if not isinstance(item, str):
            raise TypeError(f"flags entries must be strings, got {type(item).__name__}")
        normalized.add(str(item))
    return sorted(normalized)


# ---------------------------------------------------------------------------
# Shared profile/scope_type cross-field validator
# ---------------------------------------------------------------------------


def _check_profile_scope_type(event: Any) -> Any:
    """Cross-field validator: profile type MUST match scope_type (spec §4).

    Enforces the following rules:
      * scope_type='unknown' forces profile=None (spec §4.11).
      * Null-profile scope types (agent/function/retriever/embedder/reranker/
        guardrail/evaluator) forbid any non-null profile (spec §4.2).
      * ToolProfile is valid only with scope_type='tool'.
      * LLMProfile is valid only with scope_type='llm'.
      * CustomProfile is valid only with scope_type='custom'.
      * scope_type='tool' requires ToolProfile or None.
      * scope_type='llm' requires LLMProfile or None.
      * scope_type='custom' requires CustomProfile or None.
    """
    st = event.scope_type
    p = event.profile
    # Rule A: scope_type="unknown" forces profile=None (D-07 / spec §4.11)
    if st == ScopeType.UNKNOWN and p is not None:
        raise ValueError(f"scope_type='unknown' requires profile=None, got {type(p).__name__}")
    # Rule B: null-profile scope types forbid any profile (D-08)
    if st in _NULL_PROFILE_SCOPE_TYPES and p is not None:
        raise ValueError(f"scope_type='{st.value}' requires profile=None, got {type(p).__name__}")
    # Rule C: ToolProfile only with scope_type='tool'
    if isinstance(p, ToolProfile) and st != ScopeType.TOOL:
        raise ValueError(f"ToolProfile is only valid when scope_type='tool', got '{st.value}'")
    # Rule D: LLMProfile only with scope_type='llm'
    if isinstance(p, LLMProfile) and st != ScopeType.LLM:
        raise ValueError(f"LLMProfile is only valid when scope_type='llm', got '{st.value}'")
    # Rule E: CustomProfile only with scope_type='custom'
    if isinstance(p, CustomProfile) and st != ScopeType.CUSTOM:
        raise ValueError(f"CustomProfile is only valid when scope_type='custom', got '{st.value}'")
    # Rule F: scope_type='tool' with non-None non-ToolProfile profile
    if st == ScopeType.TOOL and p is not None and not isinstance(p, ToolProfile):
        raise ValueError(f"scope_type='tool' requires ToolProfile or None, got {type(p).__name__}")
    # Rule G: scope_type='llm' with non-None non-LLMProfile profile
    if st == ScopeType.LLM and p is not None and not isinstance(p, LLMProfile):
        raise ValueError(f"scope_type='llm' requires LLMProfile or None, got {type(p).__name__}")
    # Rule H: scope_type='custom' with non-None non-CustomProfile profile
    if st == ScopeType.CUSTOM and p is not None and not isinstance(p, CustomProfile):
        raise ValueError(f"scope_type='custom' requires CustomProfile or None, got {type(p).__name__}")
    return event


# ---------------------------------------------------------------------------
# Base fields shared by all event types (spec §2)
# ---------------------------------------------------------------------------


class _EventBase(BaseModel):
    """Common fields shared by all ATOF event types (spec §2)."""

    schema_version: str = Field(default="0.1", description="ATOF wire-format version (spec §5.7)")
    uuid: str = Field(description="Unique handle identifier (v7 UUID)")
    parent_uuid: str | None = Field(default=None, description="UUID of parent scope")
    timestamp: str | int = Field(description="Wall-clock time: RFC 3339 string OR int epoch microseconds (spec §5.1)")
    name: str = Field(description="Human-readable label for this handle")
    data: Any | None = Field(default=None, description="Application-specific payload")
    metadata: dict[str, Any] | None = Field(default=None, description="Tracing metadata")

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def _reject_attributes_alias(cls, values: Any) -> Any:
        """Reject any wire payload carrying the old ``attributes`` field name.

        The ``attributes`` field was renamed to ``flags`` in spec v0.1
        (per D-03); no backward-compat alias is provided. A caller who sends
        ``attributes=[]`` should get a loud failure rather than a silent
        accept-and-ignore.
        """
        if isinstance(values, dict) and "attributes" in values:
            raise ValueError("Field 'attributes' was renamed to 'flags' in spec v0.1 "
                             "(see CHANGELOG / atof-event-format.md §4.1). "
                             "No backward-compat alias is provided.")
        return values

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, v: str) -> str:
        """Enforce ``0.Y`` schema_version pattern on read (spec §5.7)."""
        if not isinstance(v, str) or not _SCHEMA_VERSION_PATTERN.match(v):
            raise ValueError(f"schema_version must match '0.Y' (spec §5.7), got {v!r}")
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ts_micros(self) -> int:
        """Normalize polymorphic timestamp to int microseconds UTC (spec §5.1)."""
        ts = self.timestamp
        if isinstance(ts, int):
            return ts
        if isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError as e:
                raise ValueError(f"Invalid RFC 3339 timestamp: {ts!r}") from e
            return int(dt.timestamp() * 1_000_000)
        raise ValueError(f"timestamp must be str or int, got {type(ts).__name__}")


# ---------------------------------------------------------------------------
# ErrorInfo (spec §3.2 status/error payload)
# ---------------------------------------------------------------------------


class ErrorInfo(BaseModel):
    """Structured error payload for failed scopes (spec §3.2)."""

    type: str = Field(description="Error class/kind identifier")
    message: str = Field(description="Human-readable error message")
    stack: str | None = Field(default=None, description="Optional stack trace")

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Event types (spec §3)
# ---------------------------------------------------------------------------


class ScopeStartEvent(_EventBase):
    """Emitted when a scope is pushed onto the scope stack (spec §3.1)."""

    kind: Literal["ScopeStart"] = "ScopeStart"
    scope_type: ScopeType = Field(description="Scope type enum value (spec §4)")
    flags: list[str] = Field(default_factory=list, description="Canonical Flags names (sorted, deduped)")
    profile: ToolProfile | LLMProfile | CustomProfile | None = Field(
        default=None, description="Scope-type-specific profile (spec §4)")
    input: Any | None = Field(default=None, description="Post-sanitize scope input payload (optional)")

    _normalize_flags = field_validator("flags", mode="before")(_canonicalize_flags)

    @model_validator(mode="after")
    def _validate_profile_matches_scope_type(self) -> ScopeStartEvent:
        return _check_profile_scope_type(self)


class ScopeEndEvent(_EventBase):
    """Emitted when a scope is popped from the scope stack (spec §3.2)."""

    kind: Literal["ScopeEnd"] = "ScopeEnd"
    scope_type: ScopeType = Field(description="Same as matching ScopeStartEvent")
    flags: list[str] = Field(default_factory=list, description="Same flags as matching ScopeStartEvent")
    profile: ToolProfile | LLMProfile | CustomProfile | None = Field(
        default=None, description="Same profile as matching ScopeStartEvent")
    output: Any | None = Field(default=None, description="Post-sanitize scope output payload (optional)")
    status: Literal["ok", "error", "cancelled"] = Field(description="Scope completion status (spec §5.6)")
    error: ErrorInfo | None = Field(default=None, description="Structured error payload (required when status='error')")

    _normalize_flags = field_validator("flags", mode="before")(_canonicalize_flags)

    @model_validator(mode="after")
    def _validate_profile_matches_scope_type(self) -> ScopeEndEvent:
        return _check_profile_scope_type(self)

    @model_validator(mode="after")
    def _validate_status_error_coherence(self) -> ScopeEndEvent:
        """Enforce status/error coherence (spec §5.6).

        * ``status='error'`` requires a non-null ``error`` payload.
        * ``status='ok'`` forbids a non-null ``error`` payload.
        * ``status='cancelled'`` allows either (cancellation may or may not
          carry a structured error).
        """
        if self.status == "error" and self.error is None:
            raise ValueError("ScopeEndEvent with status='error' requires a non-null 'error' "
                             "payload (spec §5.6)")
        if self.status == "ok" and self.error is not None:
            raise ValueError("ScopeEndEvent with status='ok' must not carry an 'error' "
                             "payload (spec §5.6)")
        return self


class MarkEvent(_EventBase):
    """Emitted for a named checkpoint in the event stream (spec §3.3)."""

    kind: Literal["Mark"] = "Mark"


# ---------------------------------------------------------------------------
# Discriminated union (spec §3)
# ---------------------------------------------------------------------------


def _get_event_kind(v: Any) -> str:
    """Extract discriminator value from raw dict or model instance."""
    if isinstance(v, dict):
        return v.get("kind", "")
    return getattr(v, "kind", "")


Event = Annotated[
    Annotated[ScopeStartEvent, Tag("ScopeStart")]
    | Annotated[ScopeEndEvent, Tag("ScopeEnd")]
    | Annotated[MarkEvent, Tag("Mark")],
    Discriminator(_get_event_kind),
]
"""Discriminated union of all 3 ATOF event types, keyed on ``kind`` (spec §3)."""
