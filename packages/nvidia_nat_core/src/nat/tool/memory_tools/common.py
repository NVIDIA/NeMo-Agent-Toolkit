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

import typing

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

from nat.builder.context import Context
from nat.data_models.component_ref import MemoryRef
from nat.data_models.function import FunctionBaseConfig


class MemoryToolConfigBase(FunctionBaseConfig):
    """Shared configuration for tools that operate on a memory backend."""

    memory: MemoryRef = Field(default=MemoryRef("saas_memory"),
                              description=("Instance name of the memory client instance from the workflow "
                                           "configuration object."))
    user_id: str | None = Field(
        default=None,
        description=("Optional fixed user identity for this memory tool. When omitted, the identity is read "
                     "from the current NVIDIA NeMo Agent Toolkit context."),
    )

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, value: str | None) -> str | None:
        if value is None:
            return None

        value = value.strip()
        if not value:
            raise ValueError("user_id must not be empty")
        return value


def resolve_memory_user_id(configured_user_id: str | None) -> str:
    """Return the configured identity or the identity bound to this invocation."""
    if configured_user_id is not None:
        return configured_user_id

    context_user_id = Context.get().user_id
    if context_user_id and context_user_id.strip():
        return context_user_id.strip()

    raise ValueError(
        "No user identity is available for this memory operation. Configure user_id or set Context.user_id."
    )


class AddMemoryInput(BaseModel):
    """LLM-visible input for adding a memory; identity is bound by the runtime."""

    model_config = ConfigDict(extra="forbid")

    conversation: list[dict[str, str]] | None = Field(
        default=None,
        description=("List of conversation messages. Each message must have a role and content key."),
    )
    tags: list[str] = Field(default_factory=list, description="List of tags applied to the item.")
    metadata: dict[str, typing.Any] = Field(default_factory=dict, description="Metadata about the memory item.")
    memory: str | None = Field(default=None, description="Optional memory text.")


class GetMemoryInput(BaseModel):
    """LLM-visible input for retrieving memories; identity is bound by the runtime."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="Search query for which to retrieve memory.")
    top_k: int = Field(description="Maximum number of memories to return")


class DeleteMemoryInput(BaseModel):
    """LLM-visible input for deleting the current user's memories."""

    model_config = ConfigDict(extra="forbid")
