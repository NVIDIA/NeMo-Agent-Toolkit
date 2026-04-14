# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Profile contract base class for ATOF v0.2 typed scopes.

Declares the ``$schema`` / ``$version`` / ``$mode`` wire-format meta-fields
for all profile payloads and runs producer-side JSON Schema validation at
construction time (spec §4). Subclasses (``DefaultLlmV1``, ``DefaultToolV1``,
vendor profiles) override ``schema_id`` / ``version`` defaults and declare
their canonical JSON Schema body in ``JSON_SCHEMA: ClassVar[dict]``.

**D-22 natural rejection:** This base class has REQUIRED ``$schema`` and
``$version`` aliases (NO defaults). Typing ``ScopeStartEvent.profile`` as
``ProfileContract | None`` makes v0.1 profile payloads (which lack
``$schema``) fail validation naturally — no special detection logic
required. Subclasses MAY supply defaults for their own schema ID/version.

See ATOF spec Section 4 (Profile Contract Protocol) and Section 6
(Reference Profile Implementations).
"""

from __future__ import annotations

from typing import Any
from typing import ClassVar
from typing import Literal
from typing import Self

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from nat.atof.validation import validate_profile


class ProfileContract(BaseModel):
    """Base class for all ATOF profile contracts (spec §4).

    The ``$schema``, ``$version``, and ``$mode`` wire-format meta-fields are
    mapped to Pythonic attribute names via Pydantic aliases. Producer-side
    validation runs at construction: vendor fields are validated against the
    declared JSON Schema (either the inline dict passed as ``$schema``, or
    the ``JSON_SCHEMA`` class attribute declared by a subclass). Validation
    failures raise ``pydantic.ValidationError``.
    """

    schema_id: str | dict = Field(
        alias="$schema",
        description="Schema ID string (e.g., 'default/llm.v1') OR inline JSON Schema dict with $id (spec §4).",
    )
    version: str = Field(
        alias="$version",
        description="Schema publisher's version string (NOT the ATOF protocol version; spec §4).",
    )
    mode: Literal["header", "inline", "opaque"] | None = Field(
        default=None,
        alias="$mode",
        description="Per-event override of StreamHeaderEvent.profile_mode_default (spec §4).",
    )

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    # Subclasses override with their canonical JSON Schema body.
    JSON_SCHEMA: ClassVar[dict[str, Any]] = {}

    @model_validator(mode="after")
    def _run_producer_validation(self) -> Self:
        """Validate vendor-fields against the declared $schema (spec §4, D-15)."""
        target = self._resolve_schema()
        if target is None:
            # String ID + empty JSON_SCHEMA → opaque passthrough (D-16 consumer
            # preservation; no validator available at producer-side).
            return self
        # NOTE: Pydantic v2 `exclude` takes Python field names, NOT aliases —
        # even when `by_alias=True` is passed. Use `schema_id`/`version`/`mode`
        # (Python names) here, not `$schema`/`$version`/`$mode` (aliases).
        vendor_fields = self.model_dump(
            by_alias=True,
            exclude={"schema_id", "version", "mode"},
            exclude_none=True,
        )
        try:
            validate_profile(vendor_fields, target)
        except Exception as e:  # jsonschema.ValidationError or swapped-lib analog
            raise ValueError(f"Profile failed $schema validation: {e}") from e
        return self

    def _resolve_schema(self) -> dict[str, Any] | None:
        """Resolve the target JSON Schema for producer validation.

        - Inline dict form of ``$schema`` → use directly.
        - String ID + non-empty subclass ``JSON_SCHEMA`` → use class-level.
        - String ID + empty subclass ``JSON_SCHEMA`` → None (opaque passthrough).
        """
        if isinstance(self.schema_id, dict):
            return self.schema_id
        if type(self).JSON_SCHEMA:
            return type(self).JSON_SCHEMA
        return None
