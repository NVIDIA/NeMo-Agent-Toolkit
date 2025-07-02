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
from aiq.data_models.authentication import ConsentPromptMode
from aiq.data_models.authentication import HeaderAuthScheme
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


async def _test_jira_api_call(oauth_client_manager: OAuthClientBase) -> typing.Any | None:
    """
    Test OAuth authentication by making provider-specific API calls.
    """
    authentication_header: httpx.Headers | None = await oauth_client_manager.construct_authentication_header(
        header_auth_scheme=HeaderAuthScheme.BEARER)

    if not authentication_header:
        return "Failed to construct authentication headers"

    base_headers: httpx.Headers = httpx.Headers({"Accept": "application/json"})
    merged_headers: httpx.Headers = httpx.Headers({**(base_headers or {}), **(authentication_header or {})})

    response: httpx.Response | None = None
    test_api_call_result: typing.Any | None = None

    # Test JIRA/Atlassian authentication with accessible resources
    response = await oauth_client_manager.send_request(url="https://api.atlassian.com/oauth/token/accessible-resources",
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
        response = await oauth_client_manager.send_request(
            url=f"https://api.atlassian.com/ex/jira/{cloud_id}/rest/api/2/project",
            http_method="GET",
            headers=dict(merged_headers),
            query_params=None,
            body_data=None)

    if (response and response.status_code == 200):
        test_api_call_result = response.json()

    return test_api_call_result


class OAuth2BrowserAuthTool(FunctionBaseConfig, name="oauth2_browser_auth_tool"):
    """OAuth 2.0 authentication to any registered API provider using authorization code flow with browser consent."""
    pass


@register_function(config_type=OAuth2BrowserAuthTool)
async def oauth2_browser_auth_tool(config: OAuth2BrowserAuthTool, builder: Builder):
    """
    Authenticates to any registered API provider using OAuth 2.0 authentication code flow.

    Extracts the provider name from user prompts like "authenticate to my registered API provider: jira"
    and uses that name to authenticate. Opens a browser for OAuth consent and tests the connection.
    """

    async def _arun(authentication_provider_name: str) -> str:

        # Get the user interaction manager from context
        aiq_context = AIQContext.get()
        user_input_manager = aiq_context.user_interaction_manager

        try:
            # Step 1: Get the OAuth registered authentication manager
            oauth_client_manager: OAuthClientBase = await builder.get_authentication(authentication_provider_name)

            # Step 2: Authenticate using browser consent flow
            authentication_error: AuthenticationError | None = await user_input_manager.authenticate_oauth_client(
                oauth_client_manager, ConsentPromptMode.BROWSER)

            if authentication_error:
                return (f"Failed to authenticate provider: {authentication_provider_name}: "
                        f"Error: {authentication_error.error_code} ")

            test_api_call_result: typing.Any | None = await _test_jira_api_call(oauth_client_manager)

            return (f"Your registered API Provider name: [{authentication_provider_name}] is now authenticated.\n"
                    f"Test API Response to API Provider: {test_api_call_result}. \n")

        except Exception as e:
            logger.exception("OAuth authentication failed", exc_info=True)
            return f"OAuth authentication to '{authentication_provider_name}' failed: {str(e)}"

    yield FunctionInfo.from_fn(
        _arun,
        description=(
            "Authenticates to any registered API provider using OAuth 2.0 authentication code flow. "
            "When user mentions 'registered API provider: <name>', extract the provider name (e.g., 'jira') "
            "and pass it as authentication_provider_name parameter. Opens browser for OAuth consent and tests."))
