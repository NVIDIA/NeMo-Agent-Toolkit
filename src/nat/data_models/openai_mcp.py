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

from enum import Enum

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class MCPApprovalRequiredEnum(str, Enum):
    """
    Enum to specify if approval is required for tool usage in the OpenAI MCP schema.
    """
    NEVER = "never"
    ALWAYS = "always"
    AUTO = "auto"


class OpenAIMCPSchemaTool(BaseModel):
    """
    Represents a tool in the OpenAI MCP schema.
    """
    type: str = "mcp"
    server_label: str = Field(description="Label for the server where the tool is hosted.")
    server_url: str = Field(description="URL of the server hosting the tool.")
    allowed_tools: list[str] | None = Field(default=None,
                                            description="List of allowed tool names that can be used by the agent.")
    require_approval: MCPApprovalRequiredEnum = Field(default=MCPApprovalRequiredEnum.NEVER,
                                                      description="Specifies if approval is required for tool usage.")
    headers: dict[str, str] | None = Field(default=None,
                                           description="Optional headers to include in requests to the tool server.")

    model_config = ConfigDict(use_enum_values=True)
