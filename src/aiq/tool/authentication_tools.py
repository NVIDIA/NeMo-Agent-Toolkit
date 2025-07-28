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

from pydantic import Field

from aiq.authentication.interfaces import AuthProviderBase
from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.authentication import AuthProviderBaseConfig
from aiq.data_models.authentication import AuthResult
from aiq.data_models.authentication import HTTPMethod
from aiq.data_models.authentication import HTTPResponse
from aiq.data_models.component_ref import AuthenticationRef
from aiq.data_models.function import FunctionBaseConfig
from aiq.tool.authenticated_request import make_authenticated_request

logger = logging.getLogger(__name__)


class AuthTool(FunctionBaseConfig, name="auth_tool"):
    """Authenticate to any registered API provider using OAuth2 authorization flow with browser consent handling."""
    auth_provider: AuthenticationRef = Field(description="Reference to the authentication provider "
                                             "to use for authentication.")


@register_function(config_type=AuthTool)
async def auth_tool(config: AuthTool, builder: Builder):
    """
    Uses HTTP Basic authentication to authenticate to any registered API provider.
    """
    basic_auth_client: AuthProviderBase = await builder.get_auth_provider(config.auth_provider)

    async def _arun(user_id: str) -> AuthResult:
        try:
            # Perform authentication (this will invoke the user authentication callback)
            auth_context: AuthResult = await basic_auth_client.authenticate(user_id=user_id)

            if not auth_context or not auth_context.credentials:
                raise RuntimeError(f"Failed to authenticate user: {user_id}: Invalid credentials")

            return auth_context

        except Exception as e:
            logger.exception("HTTP Basic authentication failed", exc_info=True)
            return f"HTTP Basic authentication for '{user_id}' failed: {str(e)}"

    yield FunctionInfo.from_fn(_arun, description="Perform authentication with a given user ID.")


class AuthenticatedRequestConfig(FunctionBaseConfig, name="authenticated_request_function"):
    """Make authenticated HTTP requests using an authentication provider."""
    auth_provider: AuthenticationRef = Field(description="Reference to the authentication provider to use for "
                                             "authentication before making an authenticated request.")
    url: str = Field(description="URL to make the request to")
    method: str | HTTPMethod = Field(default=HTTPMethod.GET, description="Default HTTP method")
    headers: str | dict | None = Field(default=None, description="Default headers")
    params: str | dict | None = Field(default=None, description="Default query parameters")
    body_data: str | dict | None = Field(default=None, description="Default request body")
    timeout: int = Field(default=30, description="Default timeout in seconds for HTTP requests")
    user_id: str | None = Field(default="default", description="Default user ID to use for authentication")


@register_function(config_type=AuthenticatedRequestConfig)
async def authenticated_request_function(config: AuthenticatedRequestConfig, builder: Builder):
    """
    Make authenticated HTTP requests to protected APIs.
    This tool automatically handles user authentication, applies proper authentication
    headers/parameters, manages token refresh, and returns structured responses.
    """
    auth_client: AuthProviderBaseConfig = await builder.get_auth_provider(config.auth_provider)

    async def _arun(user_id: str | None = "default") -> str:
        """
        Make an authenticated HTTP request to the configured endpoint.
        Args:
            user_id: Optional user identifier. Uses configured default if not provided.
        Returns:
            str: JSON string containing the API response data and metadata.
        """

        try:
            # Make the authenticated request using the utility function
            response: HTTPResponse = await make_authenticated_request(url=config.url,
                                                                      auth_client=auth_client,
                                                                      method=config.method,
                                                                      headers=config.headers,
                                                                      params=config.params,
                                                                      body_data=config.body_data,
                                                                      user_id=config.user_id,
                                                                      timeout=config.timeout)

            # Return structured response as JSON string
            if response.body is None:
                raise RuntimeError("No response body received from authenticated request")
            return response.body

        except Exception as e:
            error_msg = f"Authenticated request failed: {str(e)}"
            logger.exception("Authenticated request failed", exc_info=True)

            # Return error in consistent format
            error_response = {
                "status_code": 500,
                "body": {
                    "error": "Request failed",
                    "message": error_msg,
                    "url": config.url,
                    "method": config.method.value if isinstance(config.method, HTTPMethod) else config.method
                },
                "content_type": "application/json"
            }
            return str(error_response).replace("'", '"')

    yield FunctionInfo.from_fn(_arun,
                               description="Call protected APIs that require authentication. Use this tool to make "
                               "requests to secured endpoints - authentication and credential management "
                               "are handled automatically.")
