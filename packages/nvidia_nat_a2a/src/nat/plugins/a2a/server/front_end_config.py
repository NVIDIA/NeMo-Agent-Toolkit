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

import logging

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from nat.authentication.oauth2.oauth2_resource_server_config import OAuth2ResourceServerConfig
from nat.data_models.front_end import FrontEndBaseConfig

logger = logging.getLogger(__name__)


class A2ACapabilitiesConfig(BaseModel):
    """A2A agent capabilities configuration."""

    streaming: bool = Field(
        default=True,
        description="Enable streaming responses (default: True)",
    )
    push_notifications: bool = Field(
        default=False,
        description="Enable push notifications (default: False)",
    )


class A2AFrontEndConfig(FrontEndBaseConfig, name="a2a"):
    """A2A front end configuration.

    A front end that exposes NeMo Agent toolkit workflows as A2A-compliant remote agents.
    """

    # Server settings
    host: str = Field(
        default="localhost",
        description="Host to bind the server to (default: localhost)",
    )
    port: int = Field(
        default=10000,
        description="Port to bind the server to (default: 10000)",
        ge=0,
        le=65535,
    )
    version: str = Field(
        default="1.0.0",
        description="Version of the agent (default: 1.0.0)",
    )
    log_level: str = Field(
        default="INFO",
        description="Log level for the A2A server (default: INFO)",
    )

    # Agent metadata
    name: str = Field(
        default="NeMo Agent Toolkit A2A Agent",
        description="Name of the A2A agent (default: NeMo Agent Toolkit A2A Agent)",
    )
    description: str = Field(
        default="An AI agent powered by NeMo Agent Toolkit exposed via A2A protocol",
        description="Description of what the agent does (default: generic description)",
    )

    # A2A capabilities
    capabilities: A2ACapabilitiesConfig = Field(
        default_factory=A2ACapabilitiesConfig,
        description="Agent capabilities configuration",
    )

    # Concurrency control
    max_concurrency: int = Field(
        default=8,
        description="Maximum number of concurrent workflow executions (default: 8). "
        "Controls how many A2A requests can execute workflows simultaneously. "
        "Set to 0 or -1 for unlimited concurrency.",
        ge=-1,
    )

    # Content modes
    default_input_modes: list[str] = Field(
        default_factory=lambda: ["text", "text/plain"],
        description="Supported input content types (default: text, text/plain)",
    )
    default_output_modes: list[str] = Field(
        default_factory=lambda: ["text", "text/plain"],
        description="Supported output content types (default: text, text/plain)",
    )

    # Optional customization
    runner_class: str | None = Field(
        default=None,
        description="Custom worker class for handling A2A routes (default: built-in worker)",
    )

    # OAuth2 Resource Server (for protecting this A2A agent)
    server_auth: OAuth2ResourceServerConfig | None = Field(
        default=None,
        description=("OAuth 2.0 Resource Server configuration for token verification. "
                     "When configured, the A2A server will validate OAuth2 Bearer tokens on all requests "
                     "except public agent card discovery. Supports both JWT validation (via JWKS) and "
                     "opaque token validation (via RFC 7662 introspection)."),
    )

    @model_validator(mode="after")
    def validate_security_configuration(self):
        """Validate security configuration to prevent accidental misconfigurations."""
        # Check if server is bound to a non-localhost interface without authentication
        localhost_hosts = {"localhost", "127.0.0.1", "::1"}
        if self.host not in localhost_hosts and self.server_auth is None:
            logger.warning(
                "A2A server is configured to bind to '%s' without authentication. "
                "This may expose your server to unauthorized access. "
                "Consider either: (1) binding to localhost for local-only access, "
                "or (2) configuring server_auth for production deployments on public interfaces.",
                self.host,
            )

        return self
