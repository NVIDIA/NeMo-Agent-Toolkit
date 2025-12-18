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

from datetime import timedelta

from pydantic import Field
from pydantic import HttpUrl
from pydantic import model_validator

from nat.data_models.component_ref import AuthenticationRef
from nat.data_models.function import FunctionGroupBaseConfig


class A2AClientConfig(FunctionGroupBaseConfig, name="a2a_client"):
    """Configuration for A2A client function group.

    This configuration enables NAT workflows to connect to remote A2A agents
    and publish the primary agent function and helper functions.

    Attributes:
        url: The base URL of the A2A agent (e.g., https://agent.example.com)
        agent_card_path: Path to the agent card (default: /.well-known/agent-card.json)
        task_timeout: Maximum time to wait for task completion (default: 300 seconds)
        include_skills_in_description: Include skill details in high-level function description (default: True)
        streaming: Whether to enable streaming support for the A2A client (default: False)
        auth_provider: Optional reference to NAT auth provider for authentication
        default_user_id: Default user identifier for authentication when no session context is available
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

    include_skills_in_description: bool = Field(
        default=True,
        description="Include skill details in the high-level agent function description. "
        "Set to False for shorter descriptions (useful for token optimization). "
        "Skills are always available via get_skills() helper.",
    )

    # streaming is disabled by default because of AIQ-2496
    streaming: bool = Field(
        default=False,
        description="Whether to enable streaming support for the A2A client",
    )

    auth_provider: str | AuthenticationRef | None = Field(
        default=None,
        description="Reference to NAT authentication provider for authenticating with the A2A agent. "
        "Supports OAuth2, API Key, HTTP Basic, and other NAT auth providers.",
    )

    default_user_id: str | None = Field(
        default=None,
        description="Default user ID for authentication. Defaults to agent URL if not provided. "
        "Used for CLI workflows and token storage cache key.",
    )

    @model_validator(mode="after")
    def _set_default_user_id(self) -> "A2AClientConfig":
        """Set default_user_id to agent URL if not provided."""
        if not self.default_user_id:
            self.default_user_id = str(self.url)
        return self
