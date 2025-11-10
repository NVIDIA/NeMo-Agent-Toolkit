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

import os

from pydantic import Field
from pydantic import SecretStr
from pydantic import field_validator

from nat.authentication.interfaces import AuthProviderConfigBase


class MCPServiceAccountProviderConfig(AuthProviderConfigBase, name="mcp_service_account"):
    """
    Configuration for MCP service account authentication using OAuth2 client credentials.

    Generic implementation supporting any OAuth2 client credentials flow with
    custom header formatting for service account patterns.

    Common use cases:
    - Headless/automated MCP workflows
    - CI/CD pipelines
    - Backend services without user interaction

    All values must be provided via configuration or environment variables.
    No deployment-specific defaults are included.
    """

    # Required: OAuth2 client credentials
    client_id: str = Field(
        default_factory=lambda: os.getenv("SERVICE_ACCOUNT_CLIENT_ID", ""),
        description="OAuth2 client identifier (env: SERVICE_ACCOUNT_CLIENT_ID)",
    )

    client_secret: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("SERVICE_ACCOUNT_CLIENT_SECRET", "")),
        description="OAuth2 client secret (env: SERVICE_ACCOUNT_CLIENT_SECRET)",
    )

    # Required: Token endpoint URL
    token_url: str = Field(
        default_factory=lambda: os.getenv("SERVICE_ACCOUNT_TOKEN_URL", ""),
        description="OAuth2 token endpoint URL (env: SERVICE_ACCOUNT_TOKEN_URL)",
    )

    # Required: OAuth2 scopes
    scopes: str = Field(
        default_factory=lambda: os.getenv("SERVICE_ACCOUNT_SCOPES", ""),
        description="Space-separated OAuth2 scopes (env: SERVICE_ACCOUNT_SCOPES)",
    )

    # Optional: Custom token prefix for Authorization header
    token_prefix: str = Field(
        default="service_account",
        description="Token prefix for Authorization header (default: 'service_account'). "
        "Use empty string for standard 'Bearer <token>' format.",
    )

    # Optional: Additional service-specific token
    service_token: SecretStr | None = Field(
        default_factory=lambda: (SecretStr(token) if (token := os.getenv("SERVICE_ACCOUNT_SERVICE_TOKEN")) else None),
        description="Optional service-specific token (env: SERVICE_ACCOUNT_SERVICE_TOKEN)",
    )

    # Optional: Custom header name for service token
    service_token_header: str = Field(default="X-Service-Token",
                                      description="Header name for service token (default: 'X-Service-Token')")

    # Token caching configuration
    token_cache_buffer_seconds: int = Field(default=300,
                                            description="Seconds before token expiry to refresh (default: 300s/5min)")

    @field_validator('client_id', 'client_secret', 'token_url', 'scopes')
    @classmethod
    def validate_required_fields(cls, v, info):
        """Validate that required fields are not empty."""
        field_name = info.field_name

        # Handle SecretStr
        if isinstance(v, SecretStr):
            value = v.get_secret_value()
        else:
            value = v

        if not value or not value.strip():
            env_var_map = {
                'client_id': 'SERVICE_ACCOUNT_CLIENT_ID',
                'client_secret': 'SERVICE_ACCOUNT_CLIENT_SECRET',
                'token_url': 'SERVICE_ACCOUNT_TOKEN_URL',
                'scopes': 'SERVICE_ACCOUNT_SCOPES',
            }
            env_var = env_var_map.get(field_name, field_name.upper())
            raise ValueError(f"{field_name} is required. Set it in config or via ${env_var} environment variable")
        return v
