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

from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import MemoryRef
from nat.data_models.function import FunctionBaseConfig
from nat.memory.models import SearchMemoryInput

logger = logging.getLogger(__name__)


class GetToolConfig(FunctionBaseConfig, name="get_memory"):
    """Function to get memory to a hosted memory platform."""

    description: str = Field(default=("Tool to retrieve memory about a user's "
                                      "interactions to help answer questions in a personalized way."),
                             description="The description of this function's use for tool calling agents.")
    memory: MemoryRef = Field(default="saas_memory",
                              description=("Instance name of the memory client instance from the workflow "
                                           "configuration object."))


@register_function(config_type=GetToolConfig)
async def get_memory_tool(config: GetToolConfig, builder: Builder):
    """
    Function to get memory to a hosted memory platform.
    """
    from langchain_core.tools import ToolException

    # First, retrieve the memory client
    memory_editor = await builder.get_memory_client(config.memory)

    async def _arun(search_input: SearchMemoryInput) -> str:
        """
        Asynchronous execution of memory retrieval.

        Retrieves formatted memory from the memory provider, optimized for LLM consumption.
        Each provider formats memory according to its specific capabilities (e.g., Zep includes
        timestamps and knowledge graph structure, Redis includes similarity scores).

        Note: user_id is automatically retrieved from Context and passed to the memory editor.
        The LLM does not have access to user_id, ensuring security.
        """
        try:
            # Get user_id from Context (not from LLM input)
            user_id = Context.get().user_id or "default_NAT_user"

            # Retrieve formatted memory from memory provider
            memory_string = await memory_editor.retrieve_memory(
                query=search_input.query,
                user_id=user_id,
                top_k=search_input.top_k,
            )

            # Return memory directly or indicate if no memories found
            return memory_string if memory_string else "No relevant memories found."

        except Exception as e:
            raise ToolException(f"Error retrieving memory: {e}") from e

    yield FunctionInfo.from_fn(_arun, description=config.description)
