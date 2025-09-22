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

from pydantic import Field
from pydantic import PositiveInt

from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig


class AgentBaseConfig(FunctionBaseConfig):
    """Base configuration class for all NAT agents with common fields."""

    workflow_alias: str | None = Field(
        default=None,
        description=("The alias of the workflow. Useful when the agent is configured as a workflow "
                     "and needs to expose a customized name as a tool."))
    llm_name: LLMRef = Field(description="The LLM model to use with the agent.")
    verbose: bool = Field(default=False, description="Set the verbosity of the agent's logging.")
    description: str = Field(description="The description of this function's use.")
    log_response_max_chars: PositiveInt = Field(
        default=1000, description="Maximum number of characters to display in logs when logging responses.")
