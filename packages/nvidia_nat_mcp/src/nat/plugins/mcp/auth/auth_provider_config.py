# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pydantic import HttpUrl
from pydantic import model_validator

from nat.authentication.interfaces import AuthProviderBaseConfig
from nat.data_models.common import OptionalSecretStr


class MCPOAuth2ProviderConfig(AuthProviderBaseConfig, name="mcp_oauth2"):
    """
    MCP OAuth2 provider with endpoints discovery, optional DCR, and authentication flow via the OAuth2AuthCodeFlow
    provider.

    Supported modes:
      - Endpoints discovery + Dynamic Client Registration (DCR) (enable_dynamic_registration=True, no client_id)
      - Endpoints discovery + Manual Client Registration (client_id with optional client_secret)

    Precedence:
      - If client_id is provided, manual registration mode is used even when
        enable_dynamic_registration is True.
    """
    server_url: HttpUrl = Field(
        ...,
        description=
        "URL of the MCP server. This is the MCP server that provides tools, NOT the OAuth2 authorization server.")

    # Client registration (manual registration vs DCR)
    client_id: str | None = Field(default=None, description="OAuth2 client ID for pre-registered clients")
    client_secret: OptionalSecretStr = Field(default=None,
                                             description="OAuth2 client secret for pre-registered clients")
    enable_dynamic_registration: bool = Field(
        default=True,
        description="Enable OAuth2 Dynamic Client Registration (RFC 7591). Ignored when client_id is provided.")
    client_name: str = Field(default="NAT MCP Client", description="OAuth2 client name for dynamic registration")

    # OAuth2 flow configuration
    redirect_uri: HttpUrl = Field(..., description="OAuth2 redirect URI.")
    token_endpoint_auth_method: str = Field(default="client_secret_post",
                                            description="The authentication method for the token endpoint.")
    scopes: list[str] = Field(default_factory=list,
                              description="OAuth2 scopes, discovered from MCP server if not provided")
    # Advanced options
    use_pkce: bool = Field(default=True, description="Use PKCE for authorization code flow")
    auth_resource: str | None = Field(
        default=None,
        description="Override the resource URI sent in the OAuth2 authorization request (RFC 8707). "
        "When set, takes precedence over the resource discovered from protected resource metadata."
    )

    # These fields are only used for shared workflow (not per-user workflows)
    default_user_id: str | None = Field(default=None, description="Default user ID for authentication")
    allow_default_user_id_for_tool_calls: bool = Field(default=True, description="Allow default user ID for tool calls")

    # OAuth client credential caching
    oauth_client_ttl: float = Field(default=270.0,
                                    ge=0.0,
                                    description="Amount of time, in seconds, to cache oauth client credentials. "
                                    "Setting this to 0 disables caching.")

    # Token storage configuration
    token_storage_object_store: str | None = Field(
        default=None,
        description="Reference to object store for secure token storage. If None, uses in-memory storage.")

    @model_validator(mode="after")
    def validate_auth_config(self):
        """Validate authentication configuration for MCP-specific options."""

        # if default_user_id is not provided, use the server_url as the default user id
        if not self.default_user_id:
            self.default_user_id = str(self.server_url)
        # Manual registration + MCP discovery (public and confidential clients).
        # NOTE: client_id takes precedence over enable_dynamic_registration.
        if self.client_id:
            # Has pre-registered client ID; client_secret is optional.
            pass
        # Dynamic registration + MCP discovery
        elif self.enable_dynamic_registration:
            # Pure dynamic registration - no explicit credentials needed
            pass

        # Invalid configuration
        else:
            raise ValueError("Must provide either: "
                             "1) enable_dynamic_registration=True without client_id (dynamic), or "
                             "2) client_id with optional client_secret (manual)")

        return self
