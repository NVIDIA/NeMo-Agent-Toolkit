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

import typing

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from nat.authentication.interfaces import AuthProviderBaseConfig
from nat.data_models.common import OptionalSecretStr
from nat.data_models.common import SerializableSecretStr


class ServiceTokenConfig(BaseModel):
    """
    Configuration for service-specific token in dual authentication patterns.

    Supports two modes:

    1. Static token: Provide token and header directly
    2. Dynamic function: Provide function path and optional kwargs

    The function will be called on every request and should have signature::

        async def get_service_token(**kwargs) -> str | tuple[str, str]

    If function returns ``tuple[str, str]``, it's interpreted as (header_name, token).
    If function returns ``str``, it's the token and header field is used for header name.

    The function can access runtime context via AIQContext.get() if needed.
    """

    # Static token approach
    token: OptionalSecretStr = Field(
        default=None,
        description="Static service token value (mutually exclusive with function)",
    )

    header: str = Field(
        default="X-Service-Account-Token",
        description="HTTP header name for service token (default: 'X-Service-Account-Token')",
    )

    # Dynamic function approach
    function: str | None = Field(
        default=None,
        description=("Python function path that returns service token dynamically (mutually exclusive with token). "
                     "Function signature: async def func(\\**kwargs) -> str | tuple[str, str]. "
                     "Access runtime context via AIQContext.get() if needed."),
    )

    kwargs: dict[str, typing.Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments to pass to the custom function",
    )

    @model_validator(mode="after")
    def validate_token_or_function(self):
        """Ensure either token or function is provided, but not both."""
        has_token = self.token is not None
        has_function = self.function is not None

        if not has_token and not has_function:
            raise ValueError("Either 'token' or 'function' must be provided in service_token config")

        if has_token and has_function:
            raise ValueError("Cannot specify both 'token' and 'function' in service_token config. Choose one.")

        return self


class MCPServiceAccountProviderConfig(AuthProviderBaseConfig, name="mcp_service_account"):
    """
    Configuration for MCP service account authentication using OAuth2 client credentials.

    Generic implementation supporting any OAuth2 client credentials flow.

    Supports two authentication patterns:
    1. Single authentication: OAuth2 service account token only
    2. Dual authentication: OAuth2 service account token + service-specific token

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
    scopes: list[str] = Field(description="List of OAuth2 scopes (will be joined with spaces for OAuth2 request)")

    # Optional: Service-specific token configuration for dual authentication patterns
    service_token: ServiceTokenConfig | None = Field(
        default=None,
        description="Optional service token configuration for dual authentication patterns. "
        "Provide either a static token or a dynamic function that returns the token at runtime.",
    )

    # Token caching configuration
    token_cache_buffer_seconds: int = Field(default=300,
                                            description="Seconds before token expiry to refresh (default: 300s/5min)")

    @field_validator("scopes", mode="before")
    @classmethod
    def validate_scopes(cls, v):
        """
        Accept both list[str] and space-delimited string formats for scopes.
        Converts string to list for consistency.
        """
        if isinstance(v, str):
            # Split space-delimited string into list
            return [scope.strip() for scope in v.split() if scope.strip()]
        return v
