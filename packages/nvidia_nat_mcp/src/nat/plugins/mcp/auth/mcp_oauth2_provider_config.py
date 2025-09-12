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
from pydantic import HttpUrl
from pydantic import model_validator

from nat.authentication.interfaces import AuthProviderBaseConfig


class MCPOAuth2ProviderConfig(AuthProviderBaseConfig, name="mcp_oauth2"):
    """MCP OAuth2 authentication provider configuration for MCP-specific auth flows.

    This provider handles MCP server discovery and dynamic registration, which are not
    supported by the standard oauth2_auth_code_flow provider.

    Option 3: Dynamic registration + MCP discovery (enable_dynamic_registration=True)
    Option 4: Manual registration + MCP discovery (client_id + client_secret provided)
    """
    server_url: HttpUrl | None = Field(
        default=None,
        description=
        "URL of the MCP server (this is the MCP server that provides tools, NOT the OAuth2 authorization server)")

    # OAuth2 client credentials (Option 4 only)
    client_id: str | None = Field(default=None, description="OAuth2 client ID (for pre-registered clients)")
    client_secret: str | None = Field(default=None, description="OAuth2 client secret (for pre-registered clients)")

    # OAuth2 flow configuration
    scopes: list[str] | None = Field(default=None,
                                     description="OAuth2 scopes (discovered from MCP server if not provided)")
    redirect_uri: str | None = Field(default=None,
                                     description="OAuth2 redirect URI (defaults to localhost with random port)")

    # Advanced options
    enable_dynamic_registration: bool = Field(default=True,
                                              description="Enable OAuth2 Dynamic Client Registration (RFC 7591)")
    use_pkce: bool = Field(default=True, description="Use PKCE for authorization code flow")
    client_name: str = Field(default="NAT MCP Client", description="OAuth2 client name for dynamic registration")

    @model_validator(mode="after")
    def validate_auth_config(self):
        """Validate authentication configuration for MCP-specific options."""

        # Option 3: Dynamic registration + MCP discovery
        if self.enable_dynamic_registration and not self.client_id:
            # Pure dynamic registration - no explicit credentials needed
            pass

        # Option 4: Manual registration + MCP discovery
        elif self.client_id and self.client_secret:
            # Has credentials but will discover URLs from MCP server
            pass

        # Invalid configuration
        else:
            raise ValueError("Must provide either: "
                             "1) enable_dynamic_registration=True (dynamic), or "
                             "2) client_id + client_secret (hybrid)")

        return self
