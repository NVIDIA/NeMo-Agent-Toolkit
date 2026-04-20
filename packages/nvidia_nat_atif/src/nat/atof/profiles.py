# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scope profile models for ATOF ScopeStart/ScopeEnd events.

Each profile is a Pydantic ``BaseModel`` that carries the subject-specific
fields for a given ``scope_type``. The ``ScopeStartEvent.profile`` /
``ScopeEndEvent.profile`` fields use a typed union of these classes,
discriminated at validation time by the event's ``scope_type`` value.

See ATOF spec Sections 4.3 (tool), 4.4 (llm), and 4.10 (custom).
"""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class ToolProfile(BaseModel):
    """Profile payload for ``scope_type="tool"`` (spec §4.3)."""

    tool_call_id: str | None = Field(default=None,
                                     description="Correlation ID from LLM tool-call response")

    model_config = ConfigDict(extra="forbid")


class LLMProfile(BaseModel):
    """Profile payload for ``scope_type="llm"`` (spec §4.4)."""

    model_name: str | None = Field(default=None, description="Model identifier")

    model_config = ConfigDict(extra="forbid")


class CustomProfile(BaseModel):
    """Profile payload for ``scope_type="custom"`` (spec §4.10).

    Accepts arbitrary vendor-namespaced keys — consumers MUST preserve
    unknown fields when re-emitting per spec §5.7 forward-compat.
    """

    model_config = ConfigDict(extra="allow")
