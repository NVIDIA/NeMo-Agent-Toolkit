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

import inspect
import typing
from collections.abc import Awaitable
from collections.abc import Callable

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import ImportString
from pydantic import field_validator
from pydantic import model_validator

from nat.data_models.component_ref import MemoryRef
from nat.data_models.function import FunctionBaseConfig


UserIdResolver = Callable[[], str | Awaitable[str]]


class MemoryToolConfigBase(FunctionBaseConfig):
    """Shared configuration for memory tools."""

    memory: MemoryRef = Field(
        default=MemoryRef("saas_memory"),
        description="Instance name of the memory client from the workflow configuration.",
    )
    user_id: str | None = Field(
        default=None,
        description=(
            "Optional fixed user identity for all memory operations. Configure either user_id or user_id_resolver. "
            "This value is never exposed to the LLM."),
    )
    user_id_resolver: ImportString[UserIdResolver] | None = Field(
        default=None,
        description=(
            "Import path to a trusted zero-argument callable that returns the authenticated user's ID. The callable "
            "is invoked for every memory operation and may be synchronous or asynchronous. Configure either "
            "user_id or user_id_resolver. This value is never exposed to the LLM."),
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

    @model_validator(mode="after")
    def validate_user_id_source(self) -> typing.Self:
        if self.user_id is not None and self.user_id_resolver is not None:
            raise ValueError("Configure only one of user_id or user_id_resolver")

        return self


async def resolve_memory_user_id(config: MemoryToolConfigBase) -> str:
    """Resolve a memory identity from trusted configuration or application code."""
    if config.user_id is not None:
        return config.user_id

    if config.user_id_resolver is None:
        raise ValueError("No user identity is available. Configure user_id or user_id_resolver.")

    resolved_user_id = config.user_id_resolver()
    if inspect.isawaitable(resolved_user_id):
        resolved_user_id = await resolved_user_id

    if not isinstance(resolved_user_id, str) or not resolved_user_id.strip():
        raise ValueError("The configured user_id_resolver must return a non-empty string.")

    return resolved_user_id.strip()


class AddMemoryInput(BaseModel):
    """LLM-controlled input for adding a memory."""

    model_config = ConfigDict(extra="forbid")

    conversation: list[dict[str, str]] | None = Field(
        default=None,
        description=(
            "List of conversation messages. Each message must have a role key (user or assistant) and a content key."),
    )
    tags: list[str] = Field(default_factory=list, description="List of tags applied to the item.")
    metadata: dict[str, typing.Any] = Field(default_factory=dict, description="Metadata about the memory item.")
    memory: str | None = Field(default=None, description="A memory to store.")


class GetMemoryInput(BaseModel):
    """LLM-controlled input for retrieving memories."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="Search query for which to retrieve memory.")
    top_k: int = Field(description="Maximum number of memories to return.", gt=0)


class DeleteMemoryInput(BaseModel):
    """LLM-controlled input for deleting a user's memories."""

    model_config = ConfigDict(extra="forbid")
