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

from nat.authentication.interfaces import AuthProviderBaseConfig
from nat.data_models.common import OptionalSecretStr
from nat.data_models.common import SerializableSecretStr


class MCPServiceAccountProviderConfig(AuthProviderBaseConfig, name="mcp_service_account"):
    """
    Configuration for MCP service account authentication using OAuth2 client credentials.

    Generic implementation supporting any OAuth2 client credentials flow with
    custom header formatting for service account patterns.

    Common use cases:
    - Headless/automated MCP workflows
    - CI/CD pipelines
    - Backend services without user interaction

    All values must be provided via configuration. Use ${ENV_VAR} syntax in YAML
    configs for environment variable substitution.
    """

    # Required: OAuth2 client credentials
    client_id: str = Field(description="OAuth2 client identifier")

    client_secret: SerializableSecretStr = Field(description="OAuth2 client secret")

    # Required: Token endpoint URL
    token_url: str = Field(description="OAuth2 token endpoint URL")

    # Required: OAuth2 scopes
    scopes: str = Field(description="Space-separated OAuth2 scopes")

    # Optional: Custom token prefix for Authorization header
    token_prefix: str = Field(
        default="service_account",
        description="Token prefix for Authorization header (default: 'service_account'). "
        "Use empty string for standard 'Bearer <token>' format.",
    )

    # Optional: Additional service-specific service account token for two-header authentication patterns
    service_token: OptionalSecretStr = Field(
        default=None,
        description="Optional service account token for two-header authentication patterns",
    )

    # Optional: Custom header name for service token
    service_token_header: str = Field(
        default="Service-Account-Token",
        description="Header name for service account token (default: 'Service-Account-Token')",
    )

    # Token caching configuration
    token_cache_buffer_seconds: int = Field(default=300,
                                            description="Seconds before token expiry to refresh (default: 300s/5min)")
