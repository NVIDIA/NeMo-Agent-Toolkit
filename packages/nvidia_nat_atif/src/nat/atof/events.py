# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ATOF event models for the 2 event kinds per spec v0.1.

Standalone Pydantic models for each event kind. The ``Event`` type is a
discriminated union keyed on the ``kind`` field.

Two event kinds:
- ``ScopeEvent`` — a scope lifecycle event (start or end, distinguished by
                   ``scope_category``). A start/end pair shares the same
                   ``uuid`` (spec §5.3).
- ``MarkEvent`` — a point-in-time checkpoint, unpaired.

What kind of work an event represents is carried by the ``category`` field.
Category-specific typed fields are packaged into a single optional
``category_profile`` sub-object (spec §4.4) — ``model_name`` for ``llm``,
``tool_call_id`` for ``tool``, ``subtype`` for ``custom``, with additional
keys reserved for future categories. ``category`` is REQUIRED on
``ScopeEvent`` and OPTIONAL on ``MarkEvent``.

See ATOF spec:
- §2 (common envelope), §2.1 (attributes)
- §3 (event kinds)
- §4 (category vocabulary)
- §5 (event stream semantics)
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

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_ATOF_VERSION_PATTERN = re.compile(r"^0\.\d+$")

_CANONICAL_CATEGORIES: frozenset[str] = frozenset(
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
# Helpers
# ---------------------------------------------------------------------------


def _canonicalize_attributes(v: Any) -> list[str]:
    """Normalize an ``attributes`` field to a sorted, deduplicated list of strings.

    Accepts either a list of strings or :class:`Flags` StrEnum members. Unknown
    flag names are preserved — the spec requires consumers to round-trip them.
    """
    if v is None:
        return []
    if not isinstance(v, (list, tuple, set)):
        raise ValueError(f"attributes must be a list of strings, got {type(v).__name__}")
    normalized: set[str] = set()
    for item in v:
        if not isinstance(item, str):
            raise ValueError(f"attributes entries must be strings, got {type(item).__name__}")
        normalized.add(str(item))
    return sorted(normalized)


def _require_subtype_when_custom(category: str | None, category_profile: dict[str, Any] | None) -> None:
    """Enforce §4.2: when ``category == "custom"``, ``category_profile.subtype`` is REQUIRED."""
    if category == "custom":
        subtype = (category_profile or {}).get("subtype")
        if not isinstance(subtype, str) or not subtype:
            raise ValueError(
                "category_profile.subtype is REQUIRED and must be a non-empty string "
                "when category == 'custom' (spec §4.2)"
            )


# ---------------------------------------------------------------------------
# Base fields shared by all event types (spec §2)
# ---------------------------------------------------------------------------


class _EventBase(BaseModel):
    """Common fields shared by all ATOF event types (spec §2)."""

    atof_version: str = Field(default="0.1", description="ATOF wire-format version (spec §2, §5.6)")
    uuid: str = Field(description="Unique span identifier (v7 UUID recommended)")
    parent_uuid: str | None = Field(default=None, description="UUID of parent scope")
    timestamp: str | int = Field(description="Wall-clock time: RFC 3339 string OR int epoch microseconds (spec §5.1)")
    name: str = Field(description="Human-readable label")
    data: Any | None = Field(default=None, description="Application-defined payload; opaque to ATOF")
    data_schema: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Schema identifier {name, version} describing the shape of ``data``. "
            "Opaque to ATOF core; validation against the named schema is the "
            "consumer's responsibility (spec §2, §3)."
        ),
    )
    metadata: dict[str, Any] | None = Field(default=None, description="Tracing/correlation envelope")

    model_config = ConfigDict(extra="allow")

    @field_validator("atof_version")
    @classmethod
    def _validate_atof_version(cls, v: str) -> str:
        if not _ATOF_VERSION_PATTERN.match(v):
            raise ValueError(f"atof_version must match '0.MINOR' (e.g., '0.1'), got '{v}'")
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
        """Timestamp normalized to int epoch microseconds (spec §5.1).

        Not emitted on the wire (excluded by ``io.write_jsonl``). For
        in-memory sorting and consumer-side comparison only.
        """
        if isinstance(self.timestamp, int):
            return self.timestamp
        dt = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1_000_000)


# ---------------------------------------------------------------------------
# Event kinds (spec §3)
# ---------------------------------------------------------------------------


class ScopeEvent(_EventBase):
    """Scope lifecycle event (spec §3.1).

    A single scope span produces two ``ScopeEvent`` instances sharing the
    same ``uuid``: one with ``scope_category: "start"`` when the scope is
    pushed onto the active scope stack, and one with ``scope_category: "end"``
    when the scope is popped.
    """

    kind: Literal["scope"] = "scope"
    scope_category: Literal["start", "end"] = Field(
        description="Lifecycle phase of the scope event (spec §3.1)",
    )
    attributes: list[str] = Field(
        default_factory=list,
        description="Canonical lowercase flag array, sorted and deduplicated (spec §2.1)",
    )
    category: str = Field(description="Semantic category of the scope (spec §4)")
    category_profile: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Category-specific typed fields (spec §4.4). Keys: "
            "'model_name' for llm, 'tool_call_id' for tool, 'subtype' for custom. "
            "Null for tier-1 opaque events and categories with no defined keys."
        ),
    )

    @field_validator("attributes", mode="before")
    @classmethod
    def _canonicalize_attributes_field(cls, v: Any) -> list[str]:
        return _canonicalize_attributes(v)

    @field_validator("category")
    @classmethod
    def _validate_category(cls, v: str) -> str:
        if not isinstance(v, str) or not v:
            raise ValueError("category must be a non-empty string")
        # Canonical vocabulary is enforced at the spec level; consumers MUST NOT
        # reject unknown values (spec §4.3).
        return v

    @model_validator(mode="after")
    def _validate_category_subtype_coherence(self) -> Self:
        _require_subtype_when_custom(self.category, self.category_profile)
        return self


class MarkEvent(_EventBase):
    """Point-in-time checkpoint (spec §3.2).

    Unpaired (no start/end semantics). MAY carry ``category`` +
    ``category_profile`` to indicate the kind of work the checkpoint relates
    to; when both are absent, the mark is a generic named timestamp.
    Does NOT carry ``scope_category`` or ``attributes``.
    """

    kind: Literal["mark"] = "mark"
    category: str | None = Field(
        default=None,
        description="Semantic category (spec §4). Null or absent means the mark is a generic checkpoint.",
    )
    category_profile: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Category-specific typed fields (spec §4.4). REQUIRED when "
            "category == 'custom' (must carry category_profile.subtype)."
        ),
    )

    @model_validator(mode="after")
    def _validate_category_subtype_coherence(self) -> Self:
        _require_subtype_when_custom(self.category, self.category_profile)
        return self


# ---------------------------------------------------------------------------
# Discriminated union (spec §3)
# ---------------------------------------------------------------------------


def _get_event_kind(v: Any) -> str:
    """Extract the discriminator value from a raw dict or model instance."""
    if isinstance(v, dict):
        return v.get("kind", "")
    return getattr(v, "kind", "")


Event = Annotated[
    Annotated[ScopeEvent, Tag("scope")] | Annotated[MarkEvent, Tag("mark")],
    Discriminator(_get_event_kind),
]
"""Discriminated union of the 2 ATOF event kinds, keyed on ``kind`` (spec §3)."""
