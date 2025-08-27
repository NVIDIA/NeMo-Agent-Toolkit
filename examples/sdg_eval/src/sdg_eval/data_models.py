# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION &
# AFFILIATES. All rights reserved.
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

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class ArgumentInfo(BaseModel):
    """Information about a single function argument"""

    name: str = Field(description="Name of the argument")
    type: str = Field(description="Data type of the argument")
    title: str = Field(description="Human-readable title")
    description: str = Field(
        default="", description="Description of the argument"
    )
    required: bool = Field(
        default=False, description="Whether the argument is required"
    )


class FunctionSchema(BaseModel):
    """Schema information for a single function/tool"""

    name: str = Field(description="Name of the function")
    description: str = Field(description="Description of the function")
    input_args: List[ArgumentInfo] = Field(
        description="List of input arguments with types and details"
    )
    output_args: List[ArgumentInfo] = Field(
        description="List of output arguments with types and details"
    )


class AgentToolDetails(BaseModel):
    """Complete workflow function details with all tools"""

    config_source: str = Field(
        description="Path to the NAT agent config file"
    )
    workflow_description: str = Field(
        description="Description of the workflow and its purpose"
    )
    tools_count: int = Field(
        description="Number of tools in the workflow"
    )
    tools: List[FunctionSchema] = Field(
        description="List of function schemas for all tools"
    )


class ScenarioDescription(BaseModel):
    """A description of the scenario for which a synthetic example must be
    generated"""

    scenario_desc: str = Field(
        ...,
        description=(
            "A description of the scenario for which a synthetic "
            "example must be generated"
        )
    )


class UserQuery(BaseModel):
    """A simulated user query to the agent"""

    user_query: str = Field(..., description="The user query to the agent")


class ToolCall(BaseModel):
    """Represents a single tool call"""
    type: str = Field(default="function", description="Type of tool call")
    function: Dict[str, Any] = Field(
        description="Function details including name and arguments"
    )


class AgentResponse(BaseModel):
    """A synthetic response from the agent following OpenAI message format"""

    content: Optional[str] = Field(
        default=None,
        description=(
            "Text content of the response (null if tool calls are made)"
        )
    )
    tool_calls: Optional[List[ToolCall]] = Field(
        default=None,
        description="List of tool calls (null if text response)"
    )

    @model_validator(mode='after')
    def validate_response_type(self):
        """Ensure exactly one of content or tool_calls is populated"""
        if self.content is not None and self.tool_calls is not None:
            raise ValueError("Cannot have both content and tool_calls")
        if self.content is None and self.tool_calls is None:
            raise ValueError("Must have either content or tool_calls")
        return self
