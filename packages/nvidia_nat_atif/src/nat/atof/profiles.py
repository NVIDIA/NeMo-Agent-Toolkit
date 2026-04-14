# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference profile implementations for ATOF v0.2.

This module defines the two spec-defined reference profiles (spec §6):

- ``DefaultLlmV1`` — ``default/llm.v1`` — minimal LLM profile carrying
  ``model_name`` (preserves the v0.1 LLM-profile ``model_name`` field).
- ``DefaultToolV1`` — ``default/tool.v1`` — minimal tool profile carrying
  ``tool_call_id`` (preserves the v0.1 tool-profile ``tool_call_id`` field).

Both are intentionally minimal. Vendors publishing richer profiles (e.g.,
``openai/llm.v1`` with usage/temperature/provider fields, or
``nvidia/guardrail.v1``) subclass ``ProfileContract`` directly and declare
their own ``JSON_SCHEMA``.

See ATOF spec Section 6 (Reference Profile Implementations).
"""

from __future__ import annotations

from typing import Any
from typing import ClassVar

from pydantic import Field

from nat.atof.profile_contract import ProfileContract


class DefaultLlmV1(ProfileContract):
    """Reference profile for ``scope_type='llm'`` (spec §6.1)."""

    # Permissive extra-field policy is inherited from the ProfileContract base (no
    # model_config override here): forward-compat for vendors that subclass DefaultLlmV1
    # to add fields like `provider`, `usage`, etc., without losing those fields on
    # round-trip. Aligned with JSON_SCHEMA additionalProperties: True. Producer-side
    # validation still catches type errors on declared fields (e.g., model_name=123).
    # (W8: explicit discretion override of RESEARCH.md Open Question 1's strict-extra
    # recommendation — see 08-PLAN.md and 08-03-PLAN.md for rationale.)

    # B2 fix: schema_id annotation MUST be `str | dict` (union) to support inline-mode
    # construction via DefaultLlmV1.model_validate({"$schema": <dict>, ...}).
    # Only the DEFAULT is a string; the TYPE preserves the base class union.
    schema_id: str | dict = Field(default="default/llm.v1", alias="$schema")
    version: str = Field(default="1.0", alias="$version")
    model_name: str | None = Field(default=None, description="Model identifier (optional).")

    JSON_SCHEMA: ClassVar[dict[str, Any]] = {
        "$id": "default/llm.v1",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "model_name": {"type": ["string", "null"]},
        },
        "required": [],
        "additionalProperties": True,
    }


class DefaultToolV1(ProfileContract):
    """Reference profile for ``scope_type='tool'`` (spec §6.2)."""

    # Permissive extra-field policy inherited from ProfileContract base
    # (W8: see DefaultLlmV1 rationale above; no model_config override).

    # B2 fix: schema_id annotation preserves union `str | dict` for inline-mode support.
    schema_id: str | dict = Field(default="default/tool.v1", alias="$schema")
    version: str = Field(default="1.0", alias="$version")
    tool_call_id: str | None = Field(default=None, description="LLM tool-call correlation ID (optional).")

    JSON_SCHEMA: ClassVar[dict[str, Any]] = {
        "$id": "default/tool.v1",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "tool_call_id": {"type": ["string", "null"]},
        },
        "required": [],
        "additionalProperties": True,
    }
