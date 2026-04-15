# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ATOF event models for the 3 event kinds per spec v0.1.

Standalone Pydantic models for each event kind. The ``Event`` type is a
discriminated union keyed on the ``kind`` field.

Three event kinds:
- ``ScopeStartEvent`` — a scope was opened
- ``ScopeEndEvent``   — a scope was closed (carries terminal ``status``)
- ``MarkEvent``       — a point-in-time checkpoint

What kind of work a scope represents is carried by the ``scope_type`` field
on ``ScopeStart`` / ``ScopeEnd``. Kind-specific typed fields (``model_name``,
``tool_call_id``, ``codec`` annotations) live directly on these events and
are null for scope types that don't need them.

See ATOF spec:
- §2 (common envelope), §2.1 (attributes)
- §3 (event kinds)
- §4 (scope_type vocabulary)
- §5 (status and error semantics)
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

from nat.atof.codec import AnnotatedLLMRequest
from nat.atof.codec import AnnotatedLLMResponse
from nat.atof.flags import Flags  # noqa: F401  (re-exported for convenience)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_SCHEMA_VERSION_PATTERN = re.compile(r"^0\.\d+$")

_CANONICAL_SCOPE_TYPES: frozenset[str] = frozenset(
    {
        "agent",
        "function",
        "llm",
        "tool",
        "retriever",
        "embedder",
        "reranker",
        "guardrail",
        "evaluator",
        "custom",
        "unknown",
    }
)

# ---------------------------------------------------------------------------
# Attributes canonicalization helper (spec §2.1)
# ---------------------------------------------------------------------------


def _canonicalize_attributes(v: Any) -> list[str]:
    """Normalize an ``attributes`` field to a sorted, deduplicated list of strings.

    Accepts either a list of strings or :class:`Flags` StrEnum members. Unknown
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
# Structured error info (spec §5.1)
# ---------------------------------------------------------------------------


class ErrorInfo(BaseModel):
    """Structured error payload on ScopeEndEvent when status == 'error' (spec §5.1)."""

    message: str = Field(description="Human-readable error description")
    type: str | None = Field(default=None, description="Error class or category")
    traceback: str | None = Field(default=None, description="Optional stack trace or debug trace")

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Base fields shared by all event types (spec §2)
# ---------------------------------------------------------------------------


class _EventBase(BaseModel):
    """Common fields shared by all ATOF event types (spec §2)."""

    schema_version: str = Field(default="0.1", description="ATOF wire-format version (spec §2, §6.6)")
    uuid: str = Field(description="Unique span identifier (v7 UUID recommended)")
    parent_uuid: str | None = Field(default=None, description="UUID of parent scope")
    timestamp: str | int = Field(description="Wall-clock time: RFC 3339 string OR int epoch microseconds (spec §6.1)")
    name: str = Field(description="Human-readable label")
    data: Any | None = Field(default=None, description="Application-defined payload; opaque to ATOF")
    metadata: dict[str, Any] | None = Field(default=None, description="Tracing/correlation envelope")

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, v: str) -> str:
        if not _SCHEMA_VERSION_PATTERN.match(v):
            raise ValueError(f"schema_version must match '0.MINOR' (e.g., '0.1'), got '{v}'")
        return v

    @field_validator("uuid", "parent_uuid")
    @classmethod
    def _validate_uuid_non_empty(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not isinstance(v, str) or not v:
            raise ValueError("uuid / parent_uuid must be a non-empty string when set")
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ts_micros(self) -> int:
        """Timestamp normalized to int epoch microseconds (spec §6.1).

        Not emitted on the wire (excluded by ``io.write_jsonl``). For
        in-memory sorting and consumer-side comparison only.
        """
        if isinstance(self.timestamp, int):
            return self.timestamp
        dt = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1_000_000)


# ---------------------------------------------------------------------------
# Common scope fields shared by ScopeStartEvent and ScopeEndEvent (spec §3.1, §3.2)
# ---------------------------------------------------------------------------


class _ScopeEventBase(_EventBase):
    """Shared fields between ScopeStart and ScopeEnd (spec §3.1, §3.2)."""

    attributes: list[str] = Field(
        default_factory=list,
        description="Canonical lowercase flag array, sorted and deduplicated (spec §2.1)",
    )
    scope_type: str = Field(description="Semantic category of the scope (spec §4)")
    subtype: str | None = Field(
        default=None,
        description="Free-form vendor name. REQUIRED when scope_type=='custom'; SHOULD be absent otherwise (spec §4.2)",
    )
    model_name: str | None = Field(
        default=None,
        description="LLM model identifier. Populated when scope_type=='llm' (spec §1.2)",
    )
    tool_call_id: str | None = Field(
        default=None,
        description="Tool-call correlation ID. Populated when scope_type=='tool' (spec §1.2, §6.5)",
    )
    codec: dict[str, Any] | None = Field(
        default=None,
        description="Codec identifier {name, version} declaring the annotated_* shape (see atof-codec-profiles.md)",
    )

    @field_validator("attributes", mode="before")
    @classmethod
    def _canonicalize_attributes_field(cls, v: Any) -> list[str]:
        return _canonicalize_attributes(v)

    @field_validator("scope_type")
    @classmethod
    def _validate_scope_type(cls, v: str) -> str:
        if not isinstance(v, str) or not v:
            raise ValueError("scope_type must be a non-empty string")
        # Canonical vocabulary is enforced at the spec level; consumers MUST NOT
        # reject unknown values (spec §4.3). We warn via model_validator below
        # instead of rejecting here.
        return v

    @model_validator(mode="after")
    def _validate_scope_type_subtype_coherence(self) -> Self:
        """Enforce §4.2 subtype rules."""
        if self.scope_type == "custom":
            if self.subtype is None or self.subtype == "":
                raise ValueError("subtype is REQUIRED when scope_type == 'custom' (spec §4.2)")
        return self


# ---------------------------------------------------------------------------
# Event kinds (spec §3)
# ---------------------------------------------------------------------------


class ScopeStartEvent(_ScopeEventBase):
    """Emitted when a scope is pushed onto the active scope stack (spec §3.1)."""

    kind: Literal["ScopeStart"] = "ScopeStart"
    input: Any | None = Field(default=None, description="Raw input payload (post-guardrail); opaque to ATOF")
    annotated_request: AnnotatedLLMRequest | None = Field(
        default=None,
        description="Structured codec-decoded request (see atof-codec-profiles.md)",
    )


class ScopeEndEvent(_ScopeEventBase):
    """Emitted when a scope is popped from the active scope stack (spec §3.2).

    Paired 1:1 with ScopeStart by ``uuid``. Carries required terminal ``status``.
    """

    kind: Literal["ScopeEnd"] = "ScopeEnd"
    output: Any | None = Field(default=None, description="Raw output payload (post-guardrail); opaque to ATOF")
    annotated_response: AnnotatedLLMResponse | None = Field(
        default=None,
        description="Structured codec-decoded response (see atof-codec-profiles.md)",
    )
    status: Literal["ok", "error", "cancelled"] = Field(
        description="Terminal outcome of the scope (spec §5.1)",
    )
    error: ErrorInfo | None = Field(
        default=None,
        description="Structured error info when status=='error' (spec §5.1)",
    )

    @model_validator(mode="after")
    def _validate_status_error_coherence(self) -> Self:
        """Enforce §5.1: when status != 'error', error SHOULD be absent/null."""
        if self.status != "error" and self.error is not None:
            raise ValueError(
                f"error field populated but status=='{self.status}' "
                "(spec §5.1: error SHOULD be absent when status != 'error')"
            )
        return self


class MarkEvent(_EventBase):
    """Point-in-time checkpoint (spec §3.3).

    Unpaired (no Start/End semantics). Carries only envelope + optional data/metadata.
    Does NOT carry attributes, scope_type, status, input, or output.
    """

    kind: Literal["Mark"] = "Mark"


# ---------------------------------------------------------------------------
# Discriminated union (spec §3)
# ---------------------------------------------------------------------------


def _get_event_kind(v: Any) -> str:
    """Extract the discriminator value from a raw dict or model instance."""
    if isinstance(v, dict):
        return v.get("kind", "")
    return getattr(v, "kind", "")


Event = Annotated[
    Annotated[ScopeStartEvent, Tag("ScopeStart")]
    | Annotated[ScopeEndEvent, Tag("ScopeEnd")]
    | Annotated[MarkEvent, Tag("Mark")],
    Discriminator(_get_event_kind),
]
"""Discriminated union of the 3 ATOF event kinds, keyed on ``kind`` (spec §3)."""
