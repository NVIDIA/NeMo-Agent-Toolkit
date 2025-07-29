# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx
import pytest

from aiq.authentication.api_key.api_key_auth_provider import APIKeyAuthProvider
from aiq.authentication.api_key.api_key_auth_provider_config import APIKeyAuthProviderConfig
from aiq.authentication.api_key.api_key_auth_provider_config import HeaderAuthScheme
from aiq.authentication.http_basic_auth.http_basic_auth_provider import HTTPBasicAuthProvider
from aiq.authentication.oauth2.oauth2_auth_code_flow_provider import OAuth2AuthCodeFlowProvider
from aiq.authentication.oauth2.oauth2_auth_code_flow_provider_config import OAuth2AuthCodeFlowProviderConfig
from aiq.data_models.authentication import AuthProviderBaseConfig
from aiq.data_models.authentication import HTTPResponse

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def api_key_config():
    """Create API Key configuration for testing."""
    return APIKeyAuthProviderConfig(raw_key="test-api-key",
                                    auth_scheme=HeaderAuthScheme.BEARER,
                                    custom_header_name="Authorization",
                                    custom_header_prefix="Bearer")


@pytest.fixture
def oauth2_config():
    """Create OAuth2 configuration for testing."""
    return OAuth2AuthCodeFlowProviderConfig(client_id="test-client-id",
                                            client_secret="test-client-secret",
                                            authorization_url="https://example.com/auth",
                                            token_url="https://example.com/token",
                                            redirect_uri="https://example.com/callback")


@pytest.fixture
def basic_auth_config():
    """Create HTTP Basic Auth configuration for testing."""
    return AuthProviderBaseConfig()


@pytest.fixture
def mock_http_response():
    """Create a mock HTTP response for testing."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.json.return_value = {"status": "success", "data": "test"}
    mock_response.text = '{"status": "success", "data": "test"}'
    mock_response.cookies = {}
    mock_response.url = "https://api.example.com/test"
    mock_response.elapsed = None
    return mock_response


# --------------------------------------------------------------------------- #
# AuthProviderMixin Interface Tests
# --------------------------------------------------------------------------- #


class TestAuthProviderMixinInterface:
    """Test that all AuthProviderMixin methods work correctly across all concrete providers."""

    @pytest.mark.parametrize("provider_class,config_fixture",
                             [
                                 (APIKeyAuthProvider, "api_key_config"),
                                 (OAuth2AuthCodeFlowProvider, "oauth2_config"),
                                 (HTTPBasicAuthProvider, "basic_auth_config"),
                             ])
    async def test_request_method(self, provider_class, config_fixture, request, mock_http_response):
        """Test that the request method can be called on all providers."""

        config = request.getfixturevalue(config_fixture)
        provider = provider_class(config)

        with patch.object(provider, 'authenticate', new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = None

            with patch.object(httpx.AsyncClient, 'request', new_callable=AsyncMock) as mock_request:
                mock_request.return_value = mock_http_response

                response = await provider.request("GET",
                                                  "https://api.example.com/test",
                                                  user_id="test-user",
                                                  apply_auth=True)

                assert isinstance(response, HTTPResponse)
                assert response.status_code == 200
                mock_auth.assert_called_once_with(user_id="test-user")

    @pytest.mark.parametrize("provider_class,config_fixture",
                             [
                                 (APIKeyAuthProvider, "api_key_config"),
                                 (OAuth2AuthCodeFlowProvider, "oauth2_config"),
                                 (HTTPBasicAuthProvider, "basic_auth_config"),
                             ])
    async def test_get_method(self, provider_class, config_fixture, request, mock_http_response):
        """Test that the get method can be called on all providers."""

        config = request.getfixturevalue(config_fixture)
        provider = provider_class(config)

        with patch.object(provider, 'authenticate', new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = None

            with patch.object(httpx.AsyncClient, 'request', new_callable=AsyncMock) as mock_request:
                mock_request.return_value = mock_http_response

                response = await provider.get("https://api.example.com/users", user_id="test-user")

                assert isinstance(response, HTTPResponse)
                assert response.status_code == 200
                mock_auth.assert_called_once_with(user_id="test-user")

    @pytest.mark.parametrize("provider_class,config_fixture",
                             [
                                 (APIKeyAuthProvider, "api_key_config"),
                                 (OAuth2AuthCodeFlowProvider, "oauth2_config"),
                                 (HTTPBasicAuthProvider, "basic_auth_config"),
                             ])
    async def test_post_method(self, provider_class, config_fixture, request, mock_http_response):
        """Test that the post method can be called on all providers."""

        config = request.getfixturevalue(config_fixture)
        provider = provider_class(config)

        with patch.object(provider, 'authenticate', new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = None

            with patch.object(httpx.AsyncClient, 'request', new_callable=AsyncMock) as mock_request:
                mock_request.return_value = mock_http_response

                response = await provider.post("https://api.example.com/users",
                                               user_id="test-user",
                                               json={
                                                   "name": "John Doe", "email": "john@example.com"
                                               })

                assert isinstance(response, HTTPResponse)
                assert response.status_code == 200
                mock_auth.assert_called_once_with(user_id="test-user")

    @pytest.mark.parametrize("provider_class,config_fixture",
                             [
                                 (APIKeyAuthProvider, "api_key_config"),
                                 (OAuth2AuthCodeFlowProvider, "oauth2_config"),
                                 (HTTPBasicAuthProvider, "basic_auth_config"),
                             ])
    async def test_put_method(self, provider_class, config_fixture, request, mock_http_response):
        """Test that the put method can be called on all providers."""

        config = request.getfixturevalue(config_fixture)
        provider = provider_class(config)

        with patch.object(provider, 'authenticate', new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = None

            with patch.object(httpx.AsyncClient, 'request', new_callable=AsyncMock) as mock_request:
                mock_request.return_value = mock_http_response

                response = await provider.put("https://api.example.com/users/123",
                                              user_id="test-user",
                                              json={
                                                  "name": "Jane Doe", "email": "jane@example.com"
                                              })

                assert isinstance(response, HTTPResponse)
                assert response.status_code == 200
                mock_auth.assert_called_once_with(user_id="test-user")

    @pytest.mark.parametrize("provider_class,config_fixture",
                             [
                                 (APIKeyAuthProvider, "api_key_config"),
                                 (OAuth2AuthCodeFlowProvider, "oauth2_config"),
                                 (HTTPBasicAuthProvider, "basic_auth_config"),
                             ])
    async def test_delete_method(self, provider_class, config_fixture, request, mock_http_response):
        """Test that the delete method can be called on all providers."""

        config = request.getfixturevalue(config_fixture)
        provider = provider_class(config)

        with patch.object(provider, 'authenticate', new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = None

            with patch.object(httpx.AsyncClient, 'request', new_callable=AsyncMock) as mock_request:
                mock_request.return_value = mock_http_response

                response = await provider.delete("https://api.example.com/users/123", user_id="test-user")

                assert isinstance(response, HTTPResponse)
                assert response.status_code == 200
                mock_auth.assert_called_once_with(user_id="test-user")

    @pytest.mark.parametrize("provider_class,config_fixture",
                             [
                                 (APIKeyAuthProvider, "api_key_config"),
                                 (OAuth2AuthCodeFlowProvider, "oauth2_config"),
                                 (HTTPBasicAuthProvider, "basic_auth_config"),
                             ])
    async def test_patch_method(self, provider_class, config_fixture, request, mock_http_response):
        """Test that the patch method can be called on all providers."""

        config = request.getfixturevalue(config_fixture)
        provider = provider_class(config)

        with patch.object(provider, 'authenticate', new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = None

            with patch.object(httpx.AsyncClient, 'request', new_callable=AsyncMock) as mock_request:
                mock_request.return_value = mock_http_response

                response = await provider.patch("https://api.example.com/users/123",
                                                user_id="test-user",
                                                json={"email": "newemail@example.com"})

                assert isinstance(response, HTTPResponse)
                assert response.status_code == 200
                mock_auth.assert_called_once_with(user_id="test-user")

    @pytest.mark.parametrize("provider_class,config_fixture",
                             [
                                 (APIKeyAuthProvider, "api_key_config"),
                                 (OAuth2AuthCodeFlowProvider, "oauth2_config"),
                                 (HTTPBasicAuthProvider, "basic_auth_config"),
                             ])
    async def test_head_method(self, provider_class, config_fixture, request, mock_http_response):
        """Test that the head method can be called on all providers."""

        config = request.getfixturevalue(config_fixture)
        provider = provider_class(config)

        with patch.object(provider, 'authenticate', new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = None

            with patch.object(httpx.AsyncClient, 'request', new_callable=AsyncMock) as mock_request:
                mock_request.return_value = mock_http_response

                response = await provider.head("https://api.example.com/users/123", user_id="test-user")

                assert isinstance(response, HTTPResponse)
                assert response.status_code == 200
                mock_auth.assert_called_once_with(user_id="test-user")

    @pytest.mark.parametrize("provider_class,config_fixture",
                             [
                                 (APIKeyAuthProvider, "api_key_config"),
                                 (OAuth2AuthCodeFlowProvider, "oauth2_config"),
                                 (HTTPBasicAuthProvider, "basic_auth_config"),
                             ])
    async def test_options_method(self, provider_class, config_fixture, request, mock_http_response):
        """Test that the options method can be called on all providers."""

        config = request.getfixturevalue(config_fixture)
        provider = provider_class(config)

        with patch.object(provider, 'authenticate', new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = None

            with patch.object(httpx.AsyncClient, 'request', new_callable=AsyncMock) as mock_request:
                mock_request.return_value = mock_http_response

                response = await provider.options("https://api.example.com/users", user_id="test-user")

                assert isinstance(response, HTTPResponse)
                assert response.status_code == 200
                mock_auth.assert_called_once_with(user_id="test-user")


# --------------------------------------------------------------------------- #
# Unauthenticated Request Tests
# --------------------------------------------------------------------------- #


class TestUnauthenticatedRequests:
    """Test that all providers can make unauthenticated requests."""

    @pytest.mark.parametrize("provider_class,config_fixture",
                             [
                                 (APIKeyAuthProvider, "api_key_config"),
                                 (OAuth2AuthCodeFlowProvider, "oauth2_config"),
                                 (HTTPBasicAuthProvider, "basic_auth_config"),
                             ])
    async def test_unauthenticated_request_methods(self, provider_class, config_fixture, request, mock_http_response):
        """Test that all HTTP methods work with apply_auth=False."""

        config = request.getfixturevalue(config_fixture)
        provider = provider_class(config)

        with patch.object(httpx.AsyncClient, 'request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_http_response

            # Test all HTTP methods without authentication
            test_cases = [
                ("request", {
                    "method": "GET", "url": "https://api.example.com/public"
                }),
                ("get", {
                    "url": "https://api.example.com/public"
                }),
                ("post", {
                    "url": "https://api.example.com/public", "json": {
                        "data": "test"
                    }
                }),
                ("put", {
                    "url": "https://api.example.com/public", "json": {
                        "data": "test"
                    }
                }),
                ("delete", {
                    "url": "https://api.example.com/public"
                }),
                ("patch", {
                    "url": "https://api.example.com/public", "json": {
                        "data": "test"
                    }
                }),
                ("head", {
                    "url": "https://api.example.com/public"
                }),
                ("options", {
                    "url": "https://api.example.com/public"
                }),
            ]

            for method_name, kwargs in test_cases:
                method = getattr(provider, method_name)
                kwargs["apply_auth"] = False

                response = await method(**kwargs)

                assert isinstance(response, HTTPResponse)
                assert response.status_code == 200


# --------------------------------------------------------------------------- #
# Mixed Authentication Scenarios
# --------------------------------------------------------------------------- #


class TestMixedAuthenticationScenarios:
    """Test mixed authenticated and unauthenticated requests on the same provider."""

    @pytest.mark.parametrize("provider_class,config_fixture",
                             [
                                 (APIKeyAuthProvider, "api_key_config"),
                                 (OAuth2AuthCodeFlowProvider, "oauth2_config"),
                                 (HTTPBasicAuthProvider, "basic_auth_config"),
                             ])
    async def test_mixed_auth_requests(self, provider_class, config_fixture, request, mock_http_response):
        """Test that providers can handle both authenticated and unauthenticated requests."""

        config = request.getfixturevalue(config_fixture)
        provider = provider_class(config)

        with patch.object(provider, 'authenticate', new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = None

            with patch.object(httpx.AsyncClient, 'request', new_callable=AsyncMock) as mock_request:
                mock_request.return_value = mock_http_response

                # Make an authenticated request
                auth_response = await provider.get("https://api.example.com/private",
                                                   user_id="test-user",
                                                   apply_auth=True)

                assert isinstance(auth_response, HTTPResponse)
                assert auth_response.status_code == 200
                mock_auth.assert_called_once_with(user_id="test-user")

                # Reset mock
                mock_auth.reset_mock()

                # Make an unauthenticated request
                public_response = await provider.get("https://api.example.com/public", apply_auth=False)

                assert isinstance(public_response, HTTPResponse)
                assert public_response.status_code == 200
                # Authenticate should not be called for unauthenticated requests
                mock_auth.assert_not_called()
