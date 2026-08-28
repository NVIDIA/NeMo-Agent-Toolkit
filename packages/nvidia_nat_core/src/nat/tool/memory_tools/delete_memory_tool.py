# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from .common import DeleteMemoryInput
from .common import MemoryToolConfigBase
from .common import resolve_memory_user_id

logger = logging.getLogger(__name__)


class DeleteToolConfig(MemoryToolConfigBase, name="delete_memory"):
    """Function to delete memory from a hosted memory platform."""

    description: str = Field(default="Tool to delete a memory from a hosted memory platform.",
                             description="The description of this function's use for tool calling agents.")


@register_function(config_type=DeleteToolConfig)
async def delete_memory_tool(config: DeleteToolConfig, builder: Builder):
    """
    Function to delete memory from a hosted memory platform.
    """

    from langchain_core.tools import ToolException

    # First, retrieve the memory client
    memory_editor = await builder.get_memory_client(config.memory)

    async def _arun(delete_input: DeleteMemoryInput) -> str:
        """
        Asynchronous execution of deletion of memories.
        """

        try:
            del delete_input

            await memory_editor.remove_items(user_id=resolve_memory_user_id(config.user_id))

            return "Memories deleted!"

        except Exception as e:

            raise ToolException(f"Error deleting memory: {e}") from e

    yield FunctionInfo.from_fn(_arun, description=config.description, input_schema=DeleteMemoryInput)
