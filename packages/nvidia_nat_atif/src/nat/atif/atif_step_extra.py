# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Typed models for NAT-owned ancestry / invocation metadata embedded in
ATIF ``extra`` dicts.

The ATIF v1.7 spec does not define a typed ancestry model — producers
embed ancestry-shaped data inside the optional ``extra`` field on
records where it is meaningful. NAT's convention places:

- step-level ancestry at ``Step.extra["ancestry"]``
  (:class:`AtifAncestry` shape)
- per-tool-call ancestry at ``ToolCall.extra["ancestry"]``
  (:class:`AtifAncestry` shape)
- step-level invocation timing at ``Step.extra["invocation"]``
  (:class:`AtifInvocationInfo` shape)
- per-tool-call invocation timing at ``ToolCall.extra["invocation"]``
  (:class:`AtifInvocationInfo` shape)

These models are validated representations of those payloads — producers
MAY round-trip them through ``model_dump()`` before placing them in
``extra``; consumers MAY parse them back with ``model_validate()`` to
revalidate the shape. Direct dict use is also supported. The spec
treats ``extra`` as a loosely-typed dict, so consumers MUST tolerate
absence and missing keys.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator


class AtifAncestry(BaseModel):
    """Validated ancestry metadata embedded in ATIF ``extra`` dicts.

    Used in two locations under the NAT convention:

    - ``Step.extra["ancestry"]`` — step-level: which callable produced
      this step (e.g. an LLM node under a parent agent).
    - ``ToolCall.extra["ancestry"]`` — per-tool-call: which callable
      issued this tool invocation.

    The model enforces a parent-pair invariant: ``parent_name`` MAY only
    be set when ``parent_id`` is also set. The inverse — ``parent_id``
    set, ``parent_name`` unset — is allowed (the converter emits this
    when a parent's UUID isn't in the local name map).
    """

    model_config = ConfigDict(extra="forbid")

    function_id: str = Field(
        ...,
        description="Unique identifier for the callable node, stable across invocations.",
    )
    function_name: str = Field(
        ...,
        description="Human-readable name of the callable node.",
    )
    parent_id: str | None = Field(
        default=None,
        description="Unique identifier of the parent callable; null at the root.",
    )
    parent_name: str | None = Field(
        default=None,
        description="Human-readable name of the parent callable; null when parent_id is null.",
    )

    @model_validator(mode="after")
    def _validate_parent_pair(self) -> Self:
        if self.parent_id is None and self.parent_name is not None:
            raise ValueError("parent_name may only be set when parent_id is present")
        return self


class AtifInvocationInfo(BaseModel):
    """Invocation timing metadata embedded in ATIF ``extra`` dicts.

    Used at ``Step.extra["invocation"]`` for step-level timing and at
    ``ToolCall.extra["invocation"]`` for per-tool timing.
    """

    model_config = ConfigDict(extra="forbid")

    start_timestamp: float | None = Field(
        default=None,
        description="Invocation start timestamp in epoch seconds.",
    )
    end_timestamp: float | None = Field(
        default=None,
        description="Invocation end timestamp in epoch seconds.",
    )
    invocation_id: str | None = Field(
        default=None,
        description=(
            "Optional stable invocation identifier for correlation (for example, `tool_call_id` for tool invocations)."
        ),
    )
    status: str | None = Field(
        default=None,
        description="Optional terminal status for the invocation (for example, `completed`, `error`).",
    )
    framework: str | None = Field(
        default=None,
        description="Optional LLM framework identifier (for example, `langchain`).",
    )

    @model_validator(mode="after")
    def validate_timestamp_pairing(self) -> Self:
        has_start = self.start_timestamp is not None
        has_end = self.end_timestamp is not None
        if has_start != has_end:
            raise ValueError("start_timestamp and end_timestamp must both be set, or both be null")
        return self


class AtifStepExtra(BaseModel):
    """Validated structure for NAT-owned ATIF ``Step.extra`` payload.

    NAT writes the following keys into ``Step.extra`` under this
    convention:

    - ``ancestry`` (required by NAT's converter) — :class:`AtifAncestry`
      shape: which callable produced this step.
    - ``invocation`` (optional) — :class:`AtifInvocationInfo` shape:
      step-level timing.
    - ``data_schema`` (optional, opaque dict) — the producer-declared
      ATOF data_schema preserved for downstream validation.

    Per-tool-call ancestry and timing live on ``ToolCall.extra``, NOT
    here — they're co-located with the tool_call they describe rather
    than aligned by index on the parent step. This is the v1.7-aligned
    layout (the spec adds ``extra`` to ``ToolCall`` for this purpose).

    ``model_config = ConfigDict(extra="allow")`` so callers MAY add
    additional keys. The required ``ancestry`` field documents NAT's
    own convention but does not preclude other producers from emitting
    different ``Step.extra`` shapes.
    """

    model_config = ConfigDict(extra="allow")

    ancestry: AtifAncestry = Field(
        ...,
        description="Step-level ancestry metadata — which callable produced this step.",
    )
    invocation: AtifInvocationInfo | None = Field(
        default=None,
        description="Optional step-level invocation timing metadata.",
    )


class AtifToolCallExtra(BaseModel):
    """Validated structure for NAT-owned ATIF ``ToolCall.extra`` payload.

    NAT writes the following keys into ``ToolCall.extra`` under this
    convention:

    - ``ancestry`` (optional) — :class:`AtifAncestry` shape: which
      callable issued this tool invocation.
    - ``invocation`` (optional) — :class:`AtifInvocationInfo` shape:
      per-tool-call timing.

    ``model_config = ConfigDict(extra="allow")`` so callers MAY add
    additional keys. ``extra="allow"`` also means a ``ToolCall.extra``
    that lacks both keys still validates — neither is required by the
    NAT convention.
    """

    model_config = ConfigDict(extra="allow")

    ancestry: AtifAncestry | None = Field(
        default=None,
        description="Per-tool-call ancestry — which callable issued this tool invocation.",
    )
    invocation: AtifInvocationInfo | None = Field(
        default=None,
        description="Optional per-tool-call invocation timing metadata.",
    )
