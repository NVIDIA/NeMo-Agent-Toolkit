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
import typing

import httpx

from aiq.authentication.exceptions.call_back_exceptions import AuthenticationError
from aiq.authentication.interfaces import OAuthClientBase
from aiq.builder.builder import Builder
from aiq.builder.context import AIQContext
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.authentication import AuthenticatedContext, AuthResult
from aiq.data_models.authentication import ConsentPromptMode
from aiq.data_models.authentication import CredentialLocation
from aiq.data_models.authentication import HeaderAuthScheme
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


async def _test_jira_api_call(oauth_client: OAuthClientBase) -> typing.Any | None:
    """
    Test OAuth authentication by making provider-specific API calls.
    """

    authentication_context: AuthenticatedContext | None = await oauth_client.construct_authentication_context(
        credential_location=CredentialLocation.HEADER, header_scheme=HeaderAuthScheme.BEARER)

    if not authentication_context:
        return "Failed to construct authentication headers"

    base_headers: httpx.Headers = httpx.Headers({"Accept": "application/json"})
    merged_headers: httpx.Headers = httpx.Headers({**(base_headers or {}), **(authentication_context.headers or {})})

    response: httpx.Response | None = None
    test_api_call_result: typing.Any | None = None

    # Test JIRA/Atlassian authentication with accessible resources
    response = await oauth_client.send_request(url="https://api.atlassian.com/oauth/token/accessible-resources",
                                               http_method="GET",
                                               headers=dict(merged_headers),
                                               query_params=None,
                                               body_data=None)

    data: typing.Any | None = None
    if (response and response.status_code == 200):
        data = response.json()

    cloud_id: str | None = None
    if (data):
        for site in data:
            cloud_id = site.get("id")

    if (cloud_id):
        # Test the authentication with provider-specific API calls
        response = await oauth_client.send_request(
            url=f"https://api.atlassian.com/ex/jira/{cloud_id}/rest/api/2/project",
            http_method="GET",
            headers=dict(merged_headers),
            query_params=None,
            body_data=None)

    if (response and response.status_code == 200):
        test_api_call_result = response.json()

    return test_api_call_result

class HTTPAuthTool(FunctionBaseConfig, name="http_auth_tool"):
    """Authenticate to any registered API provider using OAuth2 authorization flow with browser consent handling."""
    pass

@register_function(config_type=HTTPAuthTool)
async def http_auth_tool(config: HTTPAuthTool, builder: Builder):
    """
    Uses HTTP Basic authentication to authenticate to any registered API provider.
    """

    async def _arun(authentication_provider_name: str) -> str:
        try:
            # Get the http basic auth registered authentication client
            http_basic_auth_client: OAuthClientBase = await builder.get_authentication(authentication_provider_name)

            # Perform authentication (this will invoke the user authentication callback)
            auth_context: AuthResult = await http_basic_auth_client.authenticate(user_id="default_user")

            if not auth_context or not auth_context.credentials:
                return f"Failed to authenticate provider: {authentication_provider_name}: Invalid credentials"

            return (f"Your registered API Provider name: [{authentication_provider_name}] is now authenticated.\n"
                    f"Credentials: {auth_context.as_requests_kwargs()}.\n")

        except Exception as e:
            logger.exception("HTTP Basic authentication failed", exc_info=True)
            return f"HTTP Basic authentication to '{authentication_provider_name}' failed: {str(e)}"

    yield FunctionInfo.from_fn(
        _arun,
        description="Use HTTP basic authentication to authenticate with a given provider."
    )


class OAuth2BrowserAuthTool(FunctionBaseConfig, name="oauth2_browser_auth_tool"):
    """Authenticate to any registered API provider using OAuth2 authorization flow with browser consent handling."""
    pass


@register_function(config_type=OAuth2BrowserAuthTool)
async def oauth2_browser_auth_tool(config: OAuth2BrowserAuthTool, builder: Builder):
    """
    Authenticates to any registered API provider using OAuth 2.0 authentication code flow.

    Extracts the provider name from user prompts (e.g., "authenticate to my registered API provider: jira"),
    and uses that name to retrieve a registered authentication client. A user authentication callback is then invoked
    to initiate the OAuth 2.0 authentication flow, opening a browser to complete the consent prompt.

    All authentication credentials are stored and managed internally. The authentication client then verifies
    the authenticated connection as a TEST in this tool by making an HTTP request and handling the response.
    """

    async def _arun(authentication_provider_name: str) -> str:

        # Get the user interaction manager from context
        aiq_context = AIQContext.get()
        user_input_manager = aiq_context.user_interaction_manager

        try:
            # Get the oauth registered authentication client
            oauth_client: OAuthClientBase = await builder.get_authentication(authentication_provider_name)

            # Authenticate by calling the authenticate_oauth_client to with browser consent handling.
            authentication_error: AuthenticationError | None = await user_input_manager.authenticate_oauth_client(
                oauth_client, ConsentPromptMode.BROWSER)

            # If an authentication error occurs, the authentication flow has failed
            if authentication_error or not await oauth_client.validate_credentials():
                return (f"Failed to authenticate provider: {authentication_provider_name}: "
                        f"Error: {authentication_error.error_code if authentication_error else 'Invalid credentials'} ")

            # Make a test API call to the API provider.
            test_api_call_result: typing.Any | None = await _test_jira_api_call(oauth_client)

            return (f"Your registered API Provider name: [{authentication_provider_name}] is now authenticated.\n"
                    f"Test API Response to API Provider: {test_api_call_result}. \n")

        except Exception as e:
            logger.exception("OAuth authentication failed", exc_info=True)
            return f"OAuth authentication to '{authentication_provider_name}' failed: {str(e)}"

    yield FunctionInfo.from_fn(
        _arun,
        description=(
            "Authenticates to any registered API provider using OAuth 2.0 flow. "
            "When user mentions 'registered API provider: <name>', extract the provider name (e.g., 'jira') "
            "and pass it as authentication_provider_name parameter. Opens browser for OAuth consent and tests."))


class OAuth2FrontendAuthTool(FunctionBaseConfig, name="oauth2_frontend_auth_tool"):
    """Authenticate to any registered API provider using OAuth2 authorization flow with frontend consent handling."""
    pass


@register_function(config_type=OAuth2FrontendAuthTool)
async def oauth2_frontend_auth_tool(config: OAuth2FrontendAuthTool, builder: Builder):
    """
    Authenticates to any registered API provider using OAuth 2.0 authentication code flow.

    Extracts the provider name from user prompts (e.g., "authenticate to my registered API provider: jira"),
    and uses that name to retrieve a registered authentication client. A user authentication callback is then invoked
    to initiate the OAuth 2.0 authentication flow. A notification is displayed on the console notifying the user to
    complete the authentication flow handling the consent prompt on the frontend.

    All authentication credentials are stored and managed internally. The authentication client then verifies
    the authenticated connection as a TEST in this tool by making an HTTP request and handling the response.
    """

    async def _arun(authentication_provider_name: str) -> str:

        # Get the user interaction manager from context
        aiq_context = AIQContext.get()
        user_input_manager = aiq_context.user_interaction_manager

        try:
            # Get the oauth registered authentication client
            oauth_client: OAuthClientBase = await builder.get_authentication(authentication_provider_name)

            # Authenticate by calling the authenticate_oauth_client to with frontend consent handling.
            authentication_error: AuthenticationError | None = await user_input_manager.authenticate_oauth_client(
                oauth_client, ConsentPromptMode.FRONTEND)

            # If an authentication error occurs, the authentication flow has failed
            if authentication_error or not await oauth_client.validate_credentials():
                return (f"Failed to authenticate provider: {authentication_provider_name}: "
                        f"Error: {authentication_error.error_code if authentication_error else 'Invalid credentials'} ")

            # Make a test API call to the API provider.
            test_api_call_result: typing.Any | None = await _test_jira_api_call(oauth_client)

            return (f"Your registered API Provider name: [{authentication_provider_name}] is now authenticated.\n"
                    f"Test API Response to API Provider: {test_api_call_result}. \n")

        except Exception as e:
            logger.exception("OAuth authentication failed", exc_info=True)
            return f"OAuth authentication to '{authentication_provider_name}' failed: {str(e)}"

    yield FunctionInfo.from_fn(
        _arun,
        description=  # noqa: E251
        ("Authenticates to any registered API provider using OAuth 2.0 flow. "
         "When user mentions 'registered API provider: <name>', extract the provider name (e.g., 'jira') "
         "and pass it as authentication_provider_name parameter. A notification is displayed on the console notifying "
         "the user to complete the authentication flow handling the consent prompt on the frontend."))
