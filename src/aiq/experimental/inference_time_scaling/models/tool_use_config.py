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

from pydantic import BaseModel
from pydantic import Field


class ToolUseInputSchema(BaseModel):
    """
    Input schema for the tool use function.
    """
    tool_name: str = Field(description="The name of the tool to use. Must be registered in the system.", )
    task_description: str = Field(description="The description of the task to perform with the tool.", )
    motivation: str | None = Field(
        default=None,
        description="An optional motivation for the tool use, providing additional context or reasoning.",
    )
    output: str | None = Field(
        default=None,
        description="The output of the tool use. This can be used to store the result of the tool execution.",
    )


class ToolUselist(BaseModel):
    """
    A list of tools to use.
    """
    tools: list[ToolUseInputSchema] = Field(
        description="A list of tool use inputs, each containing the tool name and task description.", )
