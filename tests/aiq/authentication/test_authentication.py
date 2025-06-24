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
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx
import pytest

from aiq.authentication.credentials_manager import _CredentialsManager
from aiq.authentication.exceptions import AuthCodeGrantError
from aiq.authentication.exceptions import BaseUrlValidationError
from aiq.authentication.exceptions import HTTPHeaderValidationError
from aiq.authentication.exceptions import HTTPMethodValidationError
from aiq.authentication.exceptions import QueryParameterValidationError
from aiq.authentication.oauth2.auth_code_grant_config import AuthCodeGrantConfig
from aiq.authentication.oauth2.auth_code_grant_manager import AuthCodeGrantManager
from aiq.authentication.request_manager import RequestManager
from aiq.authentication.response_manager import ResponseManager
from aiq.cli.cli_utils.config_override import load_and_override_config
from aiq.data_models.authentication import ConsentPromptMode
from aiq.data_models.authentication import ExecutionMode
from aiq.data_models.config import AIQConfig
from aiq.utils.data_models.schema_validator import validate_schema


async def test_credential_manager_singleton():
    """Test that the credential manager is a singleton class."""

    credentials1 = _CredentialsManager()
    credentials2 = _CredentialsManager()

    assert credentials1 is credentials2


async def test_credential_persistence():
    """Test that the credential manager can swap authorization configuration and persist credentials."""

    config_dict = load_and_override_config(Path("tests/aiq/authentication/config.yml"), overrides=())

    config = validate_schema(config_dict, AIQConfig)

    # Swap credentials and ensure they are not the same.
    assert _CredentialsManager()._get_authentication_config("jira") != config.authentication.get("jira")
    assert not config.authentication

    # Ensure credentials can only be swapped once.
    assert _CredentialsManager()._get_authentication_config("jira") != config.authentication.get("jira")

    # Ensure None is returned if the provider does not exist.
    test = _CredentialsManager()._get_authentication_config("invalid_provider")
    assert test is None


async def test_oauth_pydantic_model_state_field():
    """Test that the state field is not modifiable."""
    from pydantic import ValidationError

    config_dict = load_and_override_config(Path("tests/aiq/authentication/config.yml"), overrides=())
    auth_config = config_dict.get("authentication", {})
    jira_config = auth_config.get("jira", {})

    if not jira_config:
        pytest.skip("Jira configuration not found in test config")

    model = AuthCodeGrantConfig(**jira_config)

    # Throw error is state field is being modified.
    with pytest.raises(ValidationError):
        model.state = "mock_state"


@pytest.fixture
def request_manager():
    return RequestManager()


async def test_validate_base_url_valid(request_manager: RequestManager):
    """Test that the valid base URLs does NOT raise BaseUrlValidationError."""

    valid_urls = [
        "https://example.com/path",
        "http://example.com/path",
        "https://example.com:8080/path",
        "https://example.com/path?query=value"
    ]
    for url in valid_urls:
        try:
            request_manager.validate_base_url(url)
        except BaseUrlValidationError as e:
            pytest.fail(f"Valid URL '{url}' incorrectly raised BaseUrlValidationError: {e}")


async def test_validate_base_url_invalid(request_manager: RequestManager):
    """Test that the invalid base URLs raise BaseUrlValidationError."""

    invalid_urls = ["example.com/path", "ftp://example.com/path", "https:\\example.com"]

    for url in invalid_urls:
        with pytest.raises(BaseUrlValidationError):
            request_manager.validate_base_url(url)


async def test_validate_http_method_valid(request_manager: RequestManager):
    """Test that the valid HTTP methods does NOT raise ValueError."""

    valid_methods = ["GET", "POST", "PoSt", "PUT", "DELETE", "get", "post", "put", "delete"]

    for method in valid_methods:
        request_manager.validate_http_method(method)


async def test_validate_http_method_invalid(request_manager: RequestManager):
    """Test that the invalid HTTP methods raise ValueError."""

    invalid_methods = ["INVALID", "FOO", "BAR"]

    for method in invalid_methods:
        with pytest.raises(HTTPMethodValidationError):
            request_manager.validate_http_method(method)


async def test_validate_headers_valid(request_manager: RequestManager):
    """Test that the valid headers does NOT raise HeaderValidationError."""

    valid_headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer token",
        "X-Custom-Header": "value",
        "AuthorizationTest": "Basic token"
    }
    request_manager.validate_headers(valid_headers)


async def test_validate_headers_invalid(request_manager: RequestManager):
    """Test that the invalid HTTP headers raise HeaderValidationError."""

    invalid_headers = [
        {
            "Invalid@Header": "value"  # Invalid character in header name
        },
        {
            "Valid-Header": "value\nwith\nnewlines"  # Newlines in header value
        },
        {
            "": "value"  # Empty header name
        },
        {
            "Header With Spaces": "value"  # Space in header name
        },
        {
            "X-Valid-Header": ""  # Empty header value
        },
        {
            "X-Valid-Header": None  # None header value
        },
        {
            "X-Valid-Header": "value\x00with\x00nulls"  # Null characters in header value
        },
        {
            "X-Valid-Header": "value\r\nX-Injected: hacked"  # CRLF injection in header value
        },
        {
            "X-Valid-Header": "value\nX-Injected: hacked"  # LF injection in header value
        },
        {
            "X-Valid-Header\nX-Injected": "value"  # LF injection in header name
        },
        {
            "Content-Type: application/json": "value"  # Colon in header name
        },
        {
            None: "value"  # None as header name
        },
        {
            123: "value"  # Non-string header name
        },
        {
            "X-Valid-Header": "value\r\n"  # Trailing CRLF injection in header value
        },
        {
            "X-Valid-Header": "value\r"  # Lone CR injection
        },
        {
            "X-Valid-Header": "value\n"  # Lone LF injection
        },
        {
            "X-Valid-Header": "value\r\n"  # Trailing CRLF
        },
        {
            "X-Valid-Header": "value\nX-Another-Header: value"  # Header splitting via value
        },
    ]
    for headers in invalid_headers:
        with pytest.raises((HTTPHeaderValidationError, ValueError, TypeError)):
            request_manager.validate_headers(headers)


async def test_validate_query_parameters_valid(request_manager: RequestManager):
    """Test that the valid query parameters do NOT raise QueryParameterValidationError."""

    valid_query_params = {"key1": "value1", "key2": "value2", "special": "!@#$%^&*()"}

    request_manager.validate_query_parameters(valid_query_params)


async def test_validate_query_parameters_invalid(request_manager: RequestManager):
    """Test that the invalid query parameters raise QueryParameterValidationError."""

    invalid_query_params = [
        {
            "": "value"  # Empty key
        },
        {
            "key": ""  # Empty value
        },
        {
            " key": "value"  # Leading space in key
        },
        {
            "key ": "value"  # Trailing space in key
        },
        {
            " key ": "value"  # Leading and trailing spaces in key
        },
        {
            "key\nInjected": "value"  # Newline in key (potential header injection)
        },
        {
            "key": "value\nInjected"  # Newline in value (potential header injection)
        },
        {
            "key": "value\r\nAnother-Header: hacked"  # CRLF injection in value
        },
        {
            "key\r\nAnother-Header": "value"  # CRLF injection in key
        },
        {
            None: "value"  # None key
        },
        {
            "key": None  # None value
        },
        {
            123: "value"  # Non-string key
        }
    ]

    for query_params in invalid_query_params:
        with pytest.raises((QueryParameterValidationError, ValueError)):
            request_manager.validate_query_parameters(query_params)


@pytest.fixture
def mock_request_manager():
    return AsyncMock(spec=RequestManager)


@pytest.fixture
def mock_response_manager():
    return AsyncMock(spec=ResponseManager)


@pytest.fixture
def auth_code_grant_manager(mock_request_manager: AsyncMock, mock_response_manager: AsyncMock):
    return AuthCodeGrantManager(mock_request_manager, mock_response_manager, ExecutionMode.SERVER)


async def test_auth_code_grant_valid_credentials(auth_code_grant_manager: AuthCodeGrantManager):
    """Test Auth Code Grant flow valid credentials."""

    auth_code_grant_manager._config = AuthCodeGrantConfig(access_token="valid_token",
                                                          access_token_expires_in=datetime.now(timezone.utc) +
                                                          timedelta(hours=1),
                                                          client_server_url="https://test.com",
                                                          authorization_url="https://test.com/auth",
                                                          authorization_token_url="https://test.com/token",
                                                          consent_prompt_key="test_key",
                                                          client_secret="test_secret",
                                                          client_id="test_client",
                                                          audience="test_audience",
                                                          scope=["test_scope"])

    # Return True if access token is present and not expired.
    result = await auth_code_grant_manager.validate_authentication_credentials()
    assert result is True


async def test_auth_code_grant_credentials_expired(auth_code_grant_manager: AuthCodeGrantManager):
    """Test Auth Code Grant Flow valid credentials with expired access token."""

    auth_code_grant_manager._config = AuthCodeGrantConfig(access_token="expired_token",
                                                          access_token_expires_in=datetime.now(timezone.utc) -
                                                          timedelta(hours=1),
                                                          client_server_url="https://test.com",
                                                          authorization_url="https://test.com/auth",
                                                          authorization_token_url="https://test.com/token",
                                                          consent_prompt_key="test_key",
                                                          client_secret="test_secret",
                                                          client_id="test_client",
                                                          audience="test_audience",
                                                          scope=["test_scope"])

    # Return False if access token is expired.
    result = await auth_code_grant_manager.validate_authentication_credentials()
    assert result is False


async def test_auth_code_grant_no_access_token(auth_code_grant_manager: AuthCodeGrantManager):
    """Test Auth Code Grant Flow valid credentials without access token."""

    auth_code_grant_manager._config = AuthCodeGrantConfig(client_server_url="https://test.com",
                                                          authorization_url="https://test.com/auth",
                                                          authorization_token_url="https://test.com/token",
                                                          consent_prompt_key="test_key",
                                                          client_secret="test_secret",
                                                          client_id="test_client",
                                                          audience="test_audience",
                                                          scope=["test_scope"])

    # Return False if access token is missing.
    result = await auth_code_grant_manager.validate_authentication_credentials()
    assert result is False


async def test_get_access_token_with_refresh_token_success(auth_code_grant_manager: AuthCodeGrantManager,
                                                           mock_request_manager: AsyncMock):
    """Test successful refresh token flow."""

    # Assign mock request manager to the auth code grant manager
    auth_code_grant_manager._request_manager = mock_request_manager

    # Set expired access token to force refresh
    auth_code_grant_manager._config = AuthCodeGrantConfig(
        access_token="mock_access_token",
        access_token_expires_in=datetime.now(timezone.utc) - timedelta(hours=1),  # expired
        client_id="test_client",
        client_secret="test_secret",
        refresh_token="valid_refresh_token",
        authorization_token_url="https://example.com/token",  # use dummy safe URL
        client_server_url="https://example.com",
        authorization_url="https://example.com/auth",
        consent_prompt_key="test_key",
        audience="test_audience",
        scope=["test_scope"])

    # Setup mocked HTTP response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "access_token": "new_access_token", "expires_in": 3600, "refresh_token": "new_refresh_token"
    }
    mock_request_manager._send_request.return_value = mock_response

    # Attempt to get refresh token.
    await auth_code_grant_manager._get_access_token_with_refresh_token()

    # Mock refresh token flow
    assert auth_code_grant_manager._config.access_token == "new_access_token"
    assert auth_code_grant_manager._config.refresh_token == "new_refresh_token"
    assert await auth_code_grant_manager.validate_authentication_credentials() is True


@pytest.fixture
def response_manager():
    return ResponseManager()


async def test_auth_code_grant_redirect_302_without_location(response_manager: ResponseManager):
    """Test handling of 302 redirect response without Location header."""

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 302
    mock_response.headers = {}

    auth_provider = AuthCodeGrantConfig(client_server_url="https://test.com",
                                        authorization_url="https://test.com/auth",
                                        authorization_token_url="https://test.com/token",
                                        consent_prompt_key="test_key",
                                        client_secret="test_secret",
                                        client_id="test_client",
                                        audience="test_audience",
                                        scope=["test_scope"])

    # Raises OAuthCodeFlowError if Location header is missing.
    with pytest.raises(AuthCodeGrantError):
        await response_manager._handle_auth_code_grant_response_codes(mock_response, auth_provider)


async def test_auth_code_grant_handles_400_response(response_manager: ResponseManager):
    """Test handling of 400 error response."""
    error_codes = [400, 401, 403, 404, 405]

    mock_response = MagicMock(spec=httpx.Response)

    auth_provider = AuthCodeGrantConfig(client_server_url="https://test.com",
                                        authorization_url="https://test.com/auth",
                                        authorization_token_url="https://test.com/token",
                                        consent_prompt_key="test_key",
                                        client_secret="test_secret",
                                        client_id="test_client",
                                        audience="test_audience",
                                        scope=["test_scope"])

    # Raises OAuthCodeFlowError if response status code is in the 400 range
    for error_code in error_codes:
        mock_response.status_code = error_code
        with pytest.raises(AuthCodeGrantError):
            await response_manager._handle_auth_code_grant_response_codes(mock_response, auth_provider)


async def test_auth_code_grant_handles_unknown_response_codes(response_manager: ResponseManager):
    """Test handling of unknown error code in HTTPresponse."""
    error_codes = [500, 900, 899]

    mock_response = MagicMock(spec=httpx.Response)

    auth_provider = AuthCodeGrantConfig(client_server_url="https://test.com",
                                        authorization_url="https://test.com/auth",
                                        authorization_token_url="https://test.com/token",
                                        consent_prompt_key="test_key",
                                        client_secret="test_secret",
                                        client_id="test_client",
                                        audience="test_audience",
                                        scope=["test_scope"])

    # Raises OAuthCodeFlowError if response status code is unknown.
    for error_code in error_codes:
        mock_response.status_code = error_code
        with pytest.raises(AuthCodeGrantError):
            await response_manager._handle_auth_code_grant_response_codes(mock_response, auth_provider)


async def test_auth_code_grant_consent_browser_redirect_error_302(response_manager: ResponseManager):
    """Test handling of browser error in 302 consent browser."""

    location_header = "https://test.com/consent"
    auth_provider = AuthCodeGrantConfig(client_server_url="https://test.com",
                                        authorization_url="https://test.com/auth",
                                        authorization_token_url="https://test.com/token",
                                        consent_prompt_key="test_key",
                                        client_secret="test_secret",
                                        client_id="test_client",
                                        audience="test_audience",
                                        scope=["test_scope"],
                                        consent_prompt_mode=ConsentPromptMode.BROWSER)

    # Raise OAuthCodeFlowError if browser error occurs.
    with patch('webbrowser.get', side_effect=webbrowser.Error("Browser error")):

        with pytest.raises(AuthCodeGrantError):
            await response_manager._handle_auth_code_grant_302_consent_browser(location_header, auth_provider)
