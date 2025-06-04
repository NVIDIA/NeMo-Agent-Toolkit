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
from aiq.authentication.exceptions import BaseUrlValidationError
from aiq.authentication.exceptions import HTTPHeaderValidationError
from aiq.authentication.exceptions import HTTPMethodValidationError
from aiq.authentication.exceptions import OAuthCodeFlowError
from aiq.authentication.exceptions import QueryParameterValidationError
from aiq.authentication.oauth2_authenticator import OAuth2Authenticator
from aiq.authentication.request_manager import RequestManager
from aiq.authentication.response_manager import ResponseManager
from aiq.cli.cli_utils.config_override import load_and_override_config
from aiq.data_models.authentication import ConsentPromptMode
from aiq.data_models.authentication import OAuth2Config
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
    assert _CredentialsManager()._get_authentication_provider("jira") != config.authentication.get("jira")
    assert not config.authentication

    # Ensure credentials can only be swapped once.
    assert _CredentialsManager()._get_authentication_provider("jira") != config.authentication.get("jira")

    # Ensure None is returned if the provider does not exist.
    test = _CredentialsManager()._get_authentication_provider("invalid_provider")
    assert test is None


async def test_oauth_pydantic_model_state_field():
    """Test that the state field is not modifiable."""
    from pydantic import ValidationError

    config_dict = load_and_override_config(Path("tests/aiq/authentication/config.yml"), overrides=())
    auth_config = config_dict.get("authentication", {})
    jira_config = auth_config.get("jira", {})

    if not jira_config:
        pytest.skip("Jira configuration not found in test config")

    model = OAuth2Config(**jira_config)

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
            request_manager._validate_base_url(url)
        except BaseUrlValidationError as e:
            pytest.fail(f"Valid URL '{url}' incorrectly raised BaseUrlValidationError: {e}")


async def test_validate_base_url_invalid(request_manager: RequestManager):
    """Test that the invalid base URLs raise BaseUrlValidationError."""

    invalid_urls = ["example.com/path", "ftp://example.com/path", "https:\\example.com"]

    for url in invalid_urls:
        with pytest.raises(BaseUrlValidationError):
            request_manager._validate_base_url(url)


async def test_validate_http_method_valid(request_manager: RequestManager):
    """Test that the valid HTTP methods does NOT raise ValueError."""

    valid_methods = ["GET", "POST", "PoSt", "PUT", "DELETE", "get", "post", "put", "delete"]

    for method in valid_methods:
        request_manager._validate_http_method(method)


async def test_validate_http_method_invalid(request_manager: RequestManager):
    """Test that the invalid HTTP methods raise ValueError."""

    invalid_methods = ["INVALID", "FOO", "BAR"]

    for method in invalid_methods:
        with pytest.raises(HTTPMethodValidationError):
            request_manager._validate_http_method(method)


async def test_validate_headers_valid(request_manager: RequestManager):
    """Test that the valid headers does NOT raise HeaderValidationError."""

    valid_headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer token",
        "X-Custom-Header": "value",
        "AuthorizationTest": "Basic token"
    }
    request_manager._validate_headers(valid_headers)


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
            request_manager._validate_headers(headers)


async def test_validate_query_parameters_valid(request_manager: RequestManager):
    """Test that the valid query parameters do NOT raise QueryParameterValidationError."""

    valid_query_params = {"key1": "value1", "key2": "value2", "special": "!@#$%^&*()"}

    request_manager._validate_query_parameters(valid_query_params)


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
            request_manager._validate_query_parameters(query_params)


@pytest.fixture
def mock_request_manager():
    return AsyncMock(spec=RequestManager)


@pytest.fixture
def mock_response_manager():
    return AsyncMock(spec=ResponseManager)


@pytest.fixture
def oauth2_authenticator(mock_request_manager: AsyncMock, mock_response_manager: AsyncMock):
    return OAuth2Authenticator(mock_request_manager, mock_response_manager)


async def test_validate_oauth_credentials_valid(oauth2_authenticator: OAuth2Authenticator):
    """Test OAuth2.0 valid credentials."""

    oauth2_authenticator.authentication_provider = OAuth2Config(access_token="valid_token",
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
    result = await oauth2_authenticator._validate_credentials()
    assert result is True


async def test_validate_oauth_credentials_expired(oauth2_authenticator: OAuth2Authenticator):
    """Test OAuth2.0 valid credentials with expired access token."""

    oauth2_authenticator.authentication_provider = OAuth2Config(access_token="expired_token",
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
    result = await oauth2_authenticator._validate_credentials()
    assert result is False


async def test_validate_oauth_credentials_no_access_token(oauth2_authenticator: OAuth2Authenticator):
    """Test OAuth2.0 valid credentials without access token."""

    oauth2_authenticator.authentication_provider = OAuth2Config(client_server_url="https://test.com",
                                                                authorization_url="https://test.com/auth",
                                                                authorization_token_url="https://test.com/token",
                                                                consent_prompt_key="test_key",
                                                                client_secret="test_secret",
                                                                client_id="test_client",
                                                                audience="test_audience",
                                                                scope=["test_scope"])

    # Return False if access token is missing.
    result = await oauth2_authenticator._validate_credentials()
    assert result is False


async def test_get_access_token_with_refresh_token_success(oauth2_authenticator: OAuth2Authenticator,
                                                           mock_request_manager: AsyncMock):
    """Test successful refresh token flow."""
    oauth2_authenticator.authentication_provider = OAuth2Config(access_token="mock_access_token",
                                                                access_token_expires_in=datetime.now(timezone.utc) -
                                                                timedelta(hours=1),
                                                                client_id="test_client",
                                                                client_secret="test_secret",
                                                                refresh_token="valid_refresh_token",
                                                                authorization_token_url="https://test.com/token",
                                                                client_server_url="https://test.com",
                                                                authorization_url="https://test.com/auth",
                                                                consent_prompt_key="test_key",
                                                                audience="test_audience",
                                                                scope=["test_scope"])

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "access_token": "new_access_token", "expires_in": 3600, "refresh_token": "new_refresh_token"
    }
    mock_request_manager._send_request.return_value = mock_response

    # Execute refresh token flow.
    await oauth2_authenticator._get_access_token_with_refresh_token()

    # Assert that the access token and refresh token are updated.
    assert oauth2_authenticator.authentication_provider.access_token == "new_access_token"
    assert oauth2_authenticator.authentication_provider.refresh_token == "new_refresh_token"
    assert await oauth2_authenticator._validate_credentials() is True


@pytest.fixture
def response_manager():
    return ResponseManager()


async def test_handle_oauth_authorization_response_codes_302_no_location(response_manager: ResponseManager):
    """Test handling of 302 redirect response without Location header."""

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 302
    mock_response.headers = {}

    auth_provider = OAuth2Config(client_server_url="https://test.com",
                                 authorization_url="https://test.com/auth",
                                 authorization_token_url="https://test.com/token",
                                 consent_prompt_key="test_key",
                                 client_secret="test_secret",
                                 client_id="test_client",
                                 audience="test_audience",
                                 scope=["test_scope"])

    # Raises OAuthCodeFlowError if Location header is missing.
    with pytest.raises(OAuthCodeFlowError):
        await response_manager._handle_oauth_authorization_response_codes(mock_response, auth_provider)


async def test_handle_oauth_authorization_response_codes_400(response_manager: ResponseManager):
    """Test handling of 400 error response."""
    error_codes = [400, 401, 403, 404, 405]

    mock_response = MagicMock(spec=httpx.Response)

    auth_provider = OAuth2Config(client_server_url="https://test.com",
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
        with pytest.raises(OAuthCodeFlowError):
            await response_manager._handle_oauth_authorization_response_codes(mock_response, auth_provider)


async def test_handle_oauth_authorization_unknown_response_codes(response_manager: ResponseManager):
    """Test handling of unknown error code in HTTPresponse."""
    error_codes = [500, 900, 899]

    mock_response = MagicMock(spec=httpx.Response)

    auth_provider = OAuth2Config(client_server_url="https://test.com",
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
        with pytest.raises(OAuthCodeFlowError):
            await response_manager._handle_oauth_authorization_response_codes(mock_response, auth_provider)


async def test_handle_oauth_302_consent_browser_browser_error(response_manager: ResponseManager):
    """Test handling of browser error in 302 consent browser."""

    location_header = "https://test.com/consent"
    auth_provider = OAuth2Config(client_server_url="https://test.com",
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

        with pytest.raises(OAuthCodeFlowError):
            await response_manager._handle_oauth_302_consent_browser(location_header, auth_provider)
