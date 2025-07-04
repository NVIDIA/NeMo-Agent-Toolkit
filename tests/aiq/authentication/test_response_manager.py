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

import webbrowser
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx
import pytest

from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantFlowError
from aiq.authentication.oauth2.auth_code_grant_config import AuthCodeGrantConfig
from aiq.authentication.oauth2.auth_code_grant_manager import AuthCodeGrantClientManager
from aiq.authentication.response_manager import ResponseManager
from aiq.data_models.authentication import ConsentPromptMode


@pytest.fixture
def auth_code_grant_config():
    """Create a test AuthCodeGrantConfig instance."""
    return AuthCodeGrantConfig(client_server_url="https://test.com",
                               authorization_url="https://test.com/auth",
                               authorization_token_url="https://test.com/token",
                               consent_prompt_key="test_key_secure",
                               client_secret="test_secret_secure_16_chars_minimum",
                               client_id="test_client",
                               audience="test_audience",
                               scope=["test_scope"])


@pytest.fixture
def auth_code_grant_manager(auth_code_grant_config):
    """Create a test AuthCodeGrantClientManager instance."""
    return AuthCodeGrantClientManager(config=auth_code_grant_config, config_name="test_config")


@pytest.fixture
def response_manager(auth_code_grant_manager):
    """Create a test ResponseManager instance."""
    return ResponseManager(oauth_client_manager=auth_code_grant_manager)


async def test_auth_code_grant_consent_browser_redirect_error_302(auth_code_grant_manager: AuthCodeGrantClientManager):
    """Test handling of browser error in 302 consent browser."""

    location_header = "https://test.com/consent"

    # Set consent prompt mode to BROWSER
    auth_code_grant_manager.consent_prompt_mode = ConsentPromptMode.BROWSER

    # Create a response manager for this test
    response_manager = ResponseManager(oauth_client_manager=auth_code_grant_manager)

    # Raise AuthCodeGrantFlowError if browser error occurs.
    with patch('webbrowser.get', side_effect=webbrowser.Error("Browser error")):
        with pytest.raises(AuthCodeGrantFlowError):
            await response_manager.handle_consent_prompt_redirect(location_header)


async def test_auth_code_grant_redirect_302_without_location(response_manager: ResponseManager):
    """Test handling of 302 redirect response without Location header."""

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 302
    mock_response.headers = {}

    # Raises AuthCodeGrantFlowError if Location header is missing.
    with pytest.raises(AuthCodeGrantFlowError):
        await response_manager.process_http_response(mock_response)


async def test_auth_code_grant_handles_400_response(response_manager: ResponseManager):
    """Test handling of 400 error response."""
    error_codes = [400, 401, 403, 404, 405, 422, 429]

    mock_response = MagicMock(spec=httpx.Response)

    # Raises AuthCodeGrantFlowError if response status code is in the 400 range
    for error_code in error_codes:
        mock_response.status_code = error_code
        with pytest.raises(AuthCodeGrantFlowError):
            await response_manager.process_http_response(mock_response)


async def test_auth_code_grant_handles_unknown_response_codes(response_manager: ResponseManager):
    """Test handling of unknown error code in HTTPresponse."""
    error_codes = [500, 502, 503, 504, 900, 899]

    mock_response = MagicMock(spec=httpx.Response)

    # Raises AuthCodeGrantFlowError if response status code is unknown.
    for error_code in error_codes:
        mock_response.status_code = error_code
        with pytest.raises(AuthCodeGrantFlowError):
            await response_manager.process_http_response(mock_response)
