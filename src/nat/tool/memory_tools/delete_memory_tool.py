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
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import MemoryRef
from nat.data_models.function import FunctionBaseConfig
from nat.memory.models import DeleteMemoryInput

logger = logging.getLogger(__name__)


class DeleteToolConfig(FunctionBaseConfig, name="delete_memory"):
    """Function to delete memory from a hosted memory platform."""

    description: str = Field(default=("Tool to retrieve memory about a user's "
                                      "interactions to help answer questions in a personalized way."),
                             description="The description of this function's use for tool calling agents.")
    memory: MemoryRef = Field(default="saas_memory",
                              description=("Instance name of the memory client instance from the workflow "
                                           "configuration object."))


@register_function(config_type=DeleteToolConfig)
async def delete_memory_tool(config: DeleteToolConfig, builder: Builder):
    """
    Function to delete memory from a hosted memory platform.
    """

    from langchain_core.tools import ToolException

    # First, retrieve the memory client
    memory_editor = builder.get_memory_client(config.memory)

    async def _arun(user_id: str) -> str:
        """
        Asynchronous execution of deletion of memories.
        """

        try:

            await memory_editor.remove_items(user_id=user_id, )

            return "Memories deleted!"

        except Exception as e:

            raise ToolException(f"Error deleting memory: {e}") from e

    yield FunctionInfo.from_fn(_arun, description=config.description, input_schema=DeleteMemoryInput)
