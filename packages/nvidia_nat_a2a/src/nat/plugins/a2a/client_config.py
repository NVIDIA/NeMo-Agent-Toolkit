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
"""Configuration models for A2A client."""

from datetime import timedelta
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import HttpUrl

from nat.data_models.component_ref import AuthenticationRef
from nat.data_models.function import FunctionGroupBaseConfig



class A2AAgentConfig(BaseModel):
    """Configuration for connecting to a remote A2A agent.

    Attributes:
        url: The base URL of the A2A agent (e.g., https://agent.example.com)
        task_timeout: Maximum time to wait for task completion (default: 300 seconds)
        retry_max_attempts: Maximum number of retry attempts for failed requests (default: 3)
        retry_backoff: Exponential backoff multiplier for retries (default: 1.5)
        auth_provider: Optional name of NAT auth provider for authentication
    """

    url: HttpUrl = Field(
        ...,
        description="Base URL of the A2A agent",
    )

    agent_card_path: str = Field(
        default='/.well-known/agent-card.json',
        description="Path to the agent card",
    )

    task_timeout: timedelta = Field(
        default=timedelta(seconds=300),
        description="Maximum time to wait for task completion",
    )

    # Authentication configuration
    auth_provider: str | AuthenticationRef | None = Field(default=None,
                                                          description="Reference to authentication provider")



class A2AClientConfig(FunctionGroupBaseConfig, name="a2a_client"):
    """Configuration for A2A client function group.

    This configuration enables NAT workflows to connect to remote A2A agents
    and expose their skills as NAT functions.

    Attributes:
        agent: Configuration for the remote A2A agent
    """

    agent: A2AAgentConfig = Field(
        ...,
        description="Configuration for the remote A2A agent",
    )
