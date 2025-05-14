# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import MemoryRef
from aiq.data_models.function import FunctionBaseConfig
from aiq.memory.models import MemoryItem

logger = logging.getLogger(__name__)


class AddToolConfig(FunctionBaseConfig, name="add_memory"):
    """Function to add memory to a hosted memory platform."""

    description: str = Field(default=("Tool to add memory about a user's interactions to a system "
                                      "for retrieval later."),
                             description="The description of this function's use for tool calling agents.")
    memory: MemoryRef = Field(default="saas_memory",
                              description=("Instance name of the memory client instance from the workflow "
                                           "configuration object."))


@register_function(config_type=AddToolConfig)
async def add_memory_tool(config: AddToolConfig, builder: Builder):
    """
    Function to add memory to a hosted memory platform.
    """
    from langchain_core.tools import ToolException

    # First, retrieve the memory client
    memory_editor = builder.get_memory_client(config.memory)

    async def _arun(item: MemoryItem) -> str:
        """
        Asynchronous execution of addition of memories.

        Args:
            item (MemoryItem): The memory item to add. Must include:
                - conversation: List of dicts with "role" and "content" keys
                - user_id: String identifier for the user
                - metadata: Dict of metadata (can be empty)
                - tags: Optional list of tags
                - memory: Optional memory string

        Note: If conversation is not provided, it will be created from the memory field
        if available, otherwise an error will be raised.
        """
        try:
            # If conversation is not provided but memory is, create a conversation
            if not item.conversation and item.memory:
                item.conversation = [{"role": "user", "content": item.memory}]
            elif not item.conversation:
                raise ToolException("Either conversation or memory must be provided")

            await memory_editor.add_items([item])
            return "Memory added successfully. You can continue. Please respond to the user."

        except Exception as e:
            raise ToolException(f"Error adding memory: {e}") from e

    yield FunctionInfo.from_fn(_arun, description=config.description)
