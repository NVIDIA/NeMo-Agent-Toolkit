# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import logging
from collections.abc import AsyncGenerator

from pydantic import BaseModel
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import MemoryRef
from aiq.data_models.function import FunctionBaseConfig
from aiq.memory.models import MemoryItem

logger = logging.getLogger(__name__)


class MemoryAddInput(BaseModel):
    content: str = Field(..., description="The content to store in memory")
    user_id: str = Field(description="The ID of the user")
    tags: list[str] = Field(default_factory=list, description="Tags for the memory")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class MemorySearchInput(BaseModel):
    query: str = Field(..., description="The search query")
    user_id: str = Field(..., description="The ID of the user")
    top_k: int = Field(5, description="Number of results to return")


class MemorySearchToolConfig(FunctionBaseConfig, name="memory_search"):
    """Configuration for the memory search function."""
    description: str = Field(default="Search through the agent's memory using semantic search",
                             description="The description of this function's use for tool calling agents.")
    memory: MemoryRef = Field(default="redis_memory",
                              description=("Instance name of the memory client instance from the workflow "
                                           "configuration object."))


class MemoryAddToolConfig(FunctionBaseConfig, name="memory_add"):
    """Configuration for the memory addition function."""
    description: str = Field(default="Add new memory to the agent's knowledge base",
                             description="The description of this function's use for tool calling agents.")
    memory: MemoryRef = Field(default="redis_memory",
                              description=("Instance name of the memory client instance from the workflow "
                                           "configuration object."))


@register_function(config_type=MemorySearchToolConfig)
async def search_memory(config: MemorySearchToolConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    """Search through the agent's memory."""
    memory = builder.get_memory_client(config.memory)

    async def _search_memory(input_data: MemorySearchInput) -> str:
        # Ensure query is a string
        query = str(input_data.query)

        memories = await memory.search(query=query, user_id=input_data.user_id, top_k=input_data.top_k)

        memory_str = f"Memories as a JSON: \n{json.dumps([mem.model_dump(mode='json') for mem in memories])}"
        return memory_str

    yield FunctionInfo.from_fn(_search_memory, description=config.description)


@register_function(config_type=MemoryAddToolConfig)
async def add_memory(config: MemoryAddToolConfig, builder: Builder):
    """Add new memory to the agent's knowledge base."""
    memory = builder.get_memory_client(config.memory)

    async def _add_memory(input_data: MemoryAddInput) -> str:
        # Create a properly formatted conversation entry
        conversation = [{"role": "user", "content": input_data.content}]

        # Ensure memory is a string
        memory_text = str(input_data.content)

        user_id = "redis"  # TODO: figure out how to handle user_id, for now just set as `redis`
        tags = input_data.tags or []
        metadata = input_data.metadata or {}

        memory_item = MemoryItem(
            conversation=conversation,
            user_id=user_id,
            memory=memory_text,  # Use the string version of content
            tags=tags,
            metadata=metadata)

        await memory.add_items([memory_item])
        return f"Memory added: {memory_text}"

    yield FunctionInfo.from_fn(_add_memory, description=config.description)
