# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ATOF event models for the 4 event kinds (ScopeStart, ScopeEnd, Mark, StreamHeader) per spec v0.2.

Standalone Pydantic models for each event kind. The ``Event`` type is a
discriminated union keyed on the ``kind`` field.

Spec v0.2 refactors the event model from a closed scope-type catalog (v0.1)
into a profile-contract protocol. ``scope_type`` is an open-vocabulary string;
``profile`` is typed against the :class:`ProfileContract` base (D-22 natural
rejection of v0.1 streams lacking ``$schema``). A new ``StreamHeaderEvent``
declares stream-wide profile schemas and the default profile mode. See ATOF
spec §2 (common fields), §3 (event kinds), §4 (Profile Contract Protocol),
§5 (Stream Header Event), §6 (Reference Profile Implementations).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Self

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Discriminator
from pydantic import Field
from pydantic import Tag
from pydantic import computed_field
from pydantic import field_validator
from pydantic import model_validator

from nat.atof.flags import Flags  # noqa: F401  (re-exported for convenience)
from nat.atof.profile_contract import ProfileContract

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_SCHEMA_VERSION_PATTERN = re.compile(r"^0\.\d+$")

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
# Base fields shared by all event types (spec §2)
# ---------------------------------------------------------------------------


class _EventBase(BaseModel):
    """Common fields shared by all ATOF event types (spec §2)."""

    schema_version: str = Field(default="0.2", description="ATOF wire-format version (spec §2; D-20)")
    uuid: str = Field(description="Unique handle identifier (v7 UUID)")
    parent_uuid: str | None = Field(default=None, description="UUID of parent scope")
    timestamp: str | int = Field(description="Wall-clock time: RFC 3339 string OR int epoch microseconds (spec §2)")
    name: str = Field(description="Human-readable label for this handle")
    data: Any | None = Field(default=None, description="Application-specific payload")
    metadata: dict[str, Any] | None = Field(default=None, description="Tracing metadata")

    # `populate_by_name=True` (WR-04 from Phase 8 review): none of the current
    # `_EventBase` fields declare aliases, but ``ProfileContract`` intentionally
    # sets this same flag to accept its ``$``-prefixed wire keys. Mirroring the
    # setting here pre-empts breakage when a future subclass or spec revision
    # adds an aliased event field (e.g., another ``$``-prefixed wire meta-key).
    # Strictly widens accepted input; no effect on current wire shape.
    model_config = ConfigDict(extra="allow", populate_by_name=True)

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
            raise ValueError(
                "Field 'attributes' was renamed to 'flags' in spec v0.1 "
                "(see CHANGELOG / atof-event-format.md §4.1). "
                "No backward-compat alias is provided."
            )
        return values

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, v: str) -> str:
        """Enforce ``0.Y`` schema_version pattern on read (spec §2)."""
        if not isinstance(v, str) or not _SCHEMA_VERSION_PATTERN.match(v):
            raise ValueError(f"schema_version must match '0.Y' (spec §2), got {v!r}")
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ts_micros(self) -> int:
        """Normalize polymorphic timestamp to int microseconds UTC (spec §2)."""
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

    # WR-04 (Phase 8 review): mirror ``_EventBase``'s ``populate_by_name=True`` so a
    # future aliased field on ``ErrorInfo`` (none today) can be constructed either
    # by Python name or wire alias without breaking existing callers.
    model_config = ConfigDict(extra="allow", populate_by_name=True)


# ---------------------------------------------------------------------------
# Event types (spec §3)
# ---------------------------------------------------------------------------


class ScopeStartEvent(_EventBase):
    """Scope start marker for ATOF v0.2 (spec §3.1)."""

    kind: Literal["ScopeStart"] = "ScopeStart"
    scope_type: str = Field(
        min_length=1,
        description=(
            "Open-vocabulary scope type string (spec §3.1; conventions: agent, function, "
            "tool, llm, retriever, embedder, reranker, guardrail, evaluator, custom, unknown)."
        ),
    )
    flags: list[str] = Field(
        default_factory=list,
        description="Canonical Flags names (sorted, deduped; spec §2).",
    )
    profile: ProfileContract | None = Field(
        default=None,
        description="Profile contract (spec §4); base class typing is the D-22 natural-rejection anchor.",
    )
    input: Any | None = Field(
        default=None,
        description="Post-sanitize scope input payload (optional).",
    )

    _normalize_flags = field_validator("flags", mode="before")(_canonicalize_flags)
    # v0.1 closed-enum profile/scope_type cross-field validator removed per D-10.


class ScopeEndEvent(_EventBase):
    """Scope end marker for ATOF v0.2 (spec §3.2)."""

    kind: Literal["ScopeEnd"] = "ScopeEnd"
    scope_type: str = Field(min_length=1, description="Open-vocabulary scope type (spec §3.2).")
    flags: list[str] = Field(default_factory=list, description="Canonical Flags names (spec §2).")
    profile: ProfileContract | None = Field(default=None, description="Profile contract (spec §4).")
    status: Literal["ok", "error", "cancelled"] = Field(
        default="ok",
        description="Scope completion status (spec §3.2).",
    )
    error: ErrorInfo | None = Field(default=None, description="Error details when status == 'error' (spec §3.2).")
    output: Any | None = Field(default=None, description="Post-sanitize scope output payload (optional).")

    _normalize_flags = field_validator("flags", mode="before")(_canonicalize_flags)

    @model_validator(mode="after")
    def _validate_status_error_coherence(self) -> Self:
        """Enforce status/error coherence — 3-branch semantics preserved from v0.1 (spec §3.2).

        * ``status='error'`` requires a non-null ``error`` payload.
        * ``status='ok'`` forbids a non-null ``error`` payload.
        * ``status='cancelled'`` allows either (cancellation may or may not
          carry a structured error).
        """
        if self.status == "error" and self.error is None:
            raise ValueError("ScopeEndEvent with status='error' requires a non-null 'error' payload (spec §3.2)")
        if self.status == "ok" and self.error is not None:
            raise ValueError("ScopeEndEvent with status='ok' must not carry an 'error' payload (spec §3.2)")
        # status == "cancelled" → either form allowed; fall through.
        return self


class MarkEvent(_EventBase):
    """Emitted for a named checkpoint in the event stream (spec §3.3)."""

    kind: Literal["Mark"] = "Mark"


class StreamHeaderEvent(_EventBase):
    """Declares stream-wide profile schemas + default mode (spec §5)."""

    kind: Literal["StreamHeader"] = "StreamHeader"
    profile_mode_default: Literal["header", "inline", "opaque"] = Field(
        description="Default profile mode for the stream (spec §5.3).",
    )
    schemas: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Registry mapping schema_id → JSON Schema body (spec §5.4).",
    )

    @model_validator(mode="after")
    def _validate_schemas_have_consistent_ids(self) -> Self:
        """Enforce schemas[id]['$id'] == id when $id is present (spec §5.4)."""
        for schema_id, schema_body in self.schemas.items():
            declared_id = schema_body.get("$id")
            if declared_id is not None and declared_id != schema_id:
                raise ValueError(f"schemas[{schema_id!r}].$id mismatches dict key (got {declared_id!r})")
        return self


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
    | Annotated[MarkEvent, Tag("Mark")]
    | Annotated[StreamHeaderEvent, Tag("StreamHeader")],
    Discriminator(_get_event_kind),
]
"""Discriminated union of all 4 ATOF event types, keyed on ``kind`` (spec §3)."""
