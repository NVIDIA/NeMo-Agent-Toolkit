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

from nat.authentication.api_key.api_key_auth_provider import APIKeyAuthProvider
from nat.authentication.api_key.api_key_auth_provider_config import APIKeyAuthProviderConfig
from nat.authentication.api_key.api_key_auth_provider_config import HeaderAuthScheme
from nat.authentication.http_basic_auth.http_basic_auth_provider import HTTPBasicAuthProvider
from nat.authentication.oauth2.oauth2_auth_code_flow_provider import OAuth2AuthCodeFlowProvider
from nat.authentication.oauth2.oauth2_auth_code_flow_provider_config import OAuth2AuthCodeFlowProviderConfig
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.authentication import HTTPResponse

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


# --------------------------------------------------------------------------- #
# Request Parameter Handling Tests
# --------------------------------------------------------------------------- #


class TestRequestParameterHandling:
    """Test custom headers, params, timeout, body data, and kwargs handling."""

    @pytest.fixture
    def provider(self, api_key_config):
        """Create a provider for parameter testing."""
        return APIKeyAuthProvider(api_key_config)

    async def test_custom_headers_and_params(self, provider, mock_http_response):
        """Test that custom headers and params are properly passed through."""

        with patch.object(provider, 'authenticate', new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = None

            with patch.object(httpx.AsyncClient, 'request', new_callable=AsyncMock) as mock_request:
                mock_request.return_value = mock_http_response

                custom_headers = {"X-Custom-Header": "test-value", "User-Agent": "TestAgent/1.0"}
                custom_params = {"api_version": "v2", "format": "json"}

                await provider.request("GET",
                                       "https://api.example.com/test",
                                       headers=custom_headers,
                                       params=custom_params,
                                       timeout=45,
                                       apply_auth=False)

                # Verify request was called with correct parameters
                call_args = mock_request.call_args
                assert call_args[0] == ("GET", "https://api.example.com/test")

                # Check that custom headers and params were included
                request_kwargs = call_args[1]
                assert "X-Custom-Header" in request_kwargs["headers"]
                assert request_kwargs["headers"]["X-Custom-Header"] == "test-value"
                assert request_kwargs["params"]["api_version"] == "v2"
                assert request_kwargs["timeout"] == 45

    async def test_json_body_data_handling(self, provider, mock_http_response):
        """Test that JSON body data is properly handled."""

        with patch.object(provider, 'authenticate', new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = None

            with patch.object(httpx.AsyncClient, 'request', new_callable=AsyncMock) as mock_request:
                mock_request.return_value = mock_http_response

                json_data = {"name": "test", "value": 123, "nested": {"key": "value"}}

                await provider.request("POST", "https://api.example.com/test", body_data=json_data, apply_auth=False)

                # Verify JSON data was set correctly
                call_args = mock_request.call_args
                request_kwargs = call_args[1]
                assert "json" in request_kwargs
                assert request_kwargs["json"] == json_data
                assert "data" not in request_kwargs

    async def test_form_data_handling(self, provider, mock_http_response):
        """Test that form/string data is properly handled."""

        with patch.object(provider, 'authenticate', new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = None

            with patch.object(httpx.AsyncClient, 'request', new_callable=AsyncMock) as mock_request:
                mock_request.return_value = mock_http_response

                form_data = "name=test&value=123"

                await provider.request("POST", "https://api.example.com/test", body_data=form_data, apply_auth=False)

                # Verify form data was set correctly
                call_args = mock_request.call_args
                request_kwargs = call_args[1]
                assert "data" in request_kwargs
                assert request_kwargs["data"] == form_data
                assert "json" not in request_kwargs

    async def test_kwargs_parameter_passing(self, provider, mock_http_response):
        """Test that **kwargs are properly passed through to httpx."""

        with patch.object(provider, 'authenticate', new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = None

            with patch.object(httpx.AsyncClient, 'request', new_callable=AsyncMock) as mock_request:
                mock_request.return_value = mock_http_response

                await provider.request("GET",
                                       "https://api.example.com/test",
                                       follow_redirects=True,
                                       verify=False,
                                       proxies={"http": "http://proxy.example.com:8080"},
                                       apply_auth=False)

                # Verify kwargs were passed through
                call_args = mock_request.call_args
                request_kwargs = call_args[1]
                assert request_kwargs["follow_redirects"] is True
                assert request_kwargs["verify"] is False
                assert "proxies" in request_kwargs

    async def test_post_put_patch_json_data_parameter(self, provider, mock_http_response):
        """Test that POST/PUT/PATCH methods handle json_data parameter correctly."""

        with patch.object(provider, 'authenticate', new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = None

            with patch.object(httpx.AsyncClient, 'request', new_callable=AsyncMock) as mock_request:
                mock_request.return_value = mock_http_response

                json_payload = {"field": "value"}

                # Test POST
                await provider.post("https://api.example.com/test", json_data=json_payload, apply_auth=False)
                call_args = mock_request.call_args
                assert call_args[1]["json"] == json_payload

                # Test PUT
                await provider.put("https://api.example.com/test", json_data=json_payload, apply_auth=False)
                call_args = mock_request.call_args
                assert call_args[1]["json"] == json_payload

                # Test PATCH
                await provider.patch("https://api.example.com/test", json_data=json_payload, apply_auth=False)
                call_args = mock_request.call_args
                assert call_args[1]["json"] == json_payload


# --------------------------------------------------------------------------- #
# Response Conversion Tests
# --------------------------------------------------------------------------- #


class TestResponseConversion:
    """Test the _convert_response method for different content types and metadata."""

    @pytest.fixture
    def provider(self, api_key_config):
        """Create a provider for response testing."""
        return APIKeyAuthProvider(api_key_config)

    def test_json_response_conversion(self, provider):
        """Test JSON response parsing."""
        from datetime import timedelta

        import httpx

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        # Use httpx.Headers for proper case-insensitive header handling
        mock_response.headers = httpx.Headers({"content-type": "application/json", "x-rate-limit": "100"})
        mock_response.json.return_value = {"status": "success", "data": {"id": 123}}
        mock_response.text = '{"status": "success", "data": {"id": 123}}'
        mock_response.cookies = httpx.Cookies({"session": "abc123", "csrf": "xyz789"})
        mock_response.url = "https://api.example.com/users/123"
        mock_response.elapsed = timedelta(seconds=1.5)

        http_response = provider._convert_response(mock_response)

        assert http_response.status_code == 200
        assert http_response.body == {"status": "success", "data": {"id": 123}}
        assert http_response.headers["content-type"] == "application/json"
        assert http_response.headers["x-rate-limit"] == "100"
        assert http_response.cookies == {"session": "abc123", "csrf": "xyz789"}
        assert http_response.content_type == "application/json"
        assert http_response.url == "https://api.example.com/users/123"
        assert http_response.elapsed == 1.5

    def test_text_response_conversion(self, provider):
        """Test plain text response parsing."""
        import httpx

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = httpx.Headers({"content-type": "text/plain"})
        mock_response.text = "Plain text response"
        mock_response.cookies = httpx.Cookies()
        mock_response.url = "https://api.example.com/status"
        mock_response.elapsed = None

        http_response = provider._convert_response(mock_response)

        assert http_response.status_code == 200
        assert http_response.body == "Plain text response"
        assert http_response.content_type == "text/plain"
        assert http_response.cookies is None
        assert http_response.elapsed is None

    def test_html_response_conversion(self, provider):
        """Test HTML response parsing."""
        import httpx

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = httpx.Headers({"content-type": "text/html"})
        mock_response.text = "<html><body>Hello World</body></html>"
        mock_response.cookies = httpx.Cookies()
        mock_response.url = "https://example.com/page"
        mock_response.elapsed = None

        http_response = provider._convert_response(mock_response)

        assert http_response.status_code == 200
        assert http_response.body == "<html><body>Hello World</body></html>"
        assert http_response.content_type == "text/html"

    def test_invalid_json_fallback(self, provider):
        """Test that invalid JSON falls back to text."""
        import httpx

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = httpx.Headers({"content-type": "application/json"})
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.text = "Invalid JSON content"
        mock_response.cookies = httpx.Cookies()
        mock_response.url = "https://api.example.com/error"
        mock_response.elapsed = None

        http_response = provider._convert_response(mock_response)

        assert http_response.status_code == 200
        assert http_response.body == "Invalid JSON content"
        assert http_response.content_type == "application/json"

    def test_response_with_auth_result(self, provider):
        """Test response conversion with auth result included."""
        import httpx
        from pydantic import SecretStr

        from nat.data_models.authentication import AuthResult
        from nat.data_models.authentication import BearerTokenCred

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = httpx.Headers({"content-type": "application/json"})
        mock_response.json.return_value = {"data": "test"}
        mock_response.text = '{"data": "test"}'
        mock_response.cookies = httpx.Cookies()
        mock_response.url = "https://api.example.com/test"
        mock_response.elapsed = None

        auth_result = AuthResult(credentials=[BearerTokenCred(token=SecretStr("test-token"))])

        http_response = provider._convert_response(mock_response, auth_result)

        assert http_response.auth_result == auth_result
        assert http_response.auth_result.credentials[0].token.get_secret_value() == "test-token"


# --------------------------------------------------------------------------- #
# Edge Cases Tests
# --------------------------------------------------------------------------- #


class TestEdgeCases:
    """Test edge cases like user_id validation, session fallback, and error scenarios."""

    @pytest.fixture
    def provider(self, api_key_config):
        """Create a provider for edge case testing."""
        return APIKeyAuthProvider(api_key_config)

    async def test_empty_user_id_validation(self, provider):
        """Test that empty/whitespace user_id raises ValueError when apply_auth=True."""

        with pytest.raises(ValueError, match="user_id cannot be empty or whitespace-only"):
            await provider.request("GET", "https://api.example.com/test", user_id="", apply_auth=True)

        with pytest.raises(ValueError, match="user_id cannot be empty or whitespace-only"):
            await provider.request("GET", "https://api.example.com/test", user_id="   ", apply_auth=True)

        with pytest.raises(ValueError, match="user_id cannot be empty or whitespace-only"):
            await provider.request("GET", "https://api.example.com/test", user_id="\t\n", apply_auth=True)

    async def test_network_error_handling(self, provider):
        """Test handling of network errors."""

        with patch.object(provider, 'authenticate', new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = None

            with patch.object(httpx.AsyncClient, 'request', new_callable=AsyncMock) as mock_request:
                mock_request.side_effect = Exception("Network timeout")

                response = await provider.request("GET", "https://api.example.com/test", apply_auth=False)

                # Should return error response instead of throwing
                assert response.status_code == 500
                assert "Network timeout" in response.body["message"]
                assert response.body["error"] == "Request failed"

    async def test_large_response_handling(self, provider):
        """Test handling of responses with large content."""
        import httpx

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = httpx.Headers({"content-type": "application/json"})
        large_data = {"data": "x" * 10000, "items": [{"id": i} for i in range(1000)]}
        mock_response.json.return_value = large_data
        mock_response.text = "Large response content"
        mock_response.cookies = httpx.Cookies()
        mock_response.url = "https://api.example.com/large"
        mock_response.elapsed = None

        http_response = provider._convert_response(mock_response)

        assert http_response.status_code == 200
        assert len(http_response.body["data"]) == 10000
        assert len(http_response.body["items"]) == 1000

    async def test_timeout_parameter_handling(self, provider, mock_http_response):
        """Test that timeout parameter is properly set."""

        with patch.object(provider, 'authenticate', new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = None

            with patch.object(httpx.AsyncClient, 'request', new_callable=AsyncMock) as mock_request:
                mock_request.return_value = mock_http_response

                # Test custom timeout
                await provider.request("GET", "https://api.example.com/test", timeout=60, apply_auth=False)
                call_args = mock_request.call_args
                assert call_args[1]["timeout"] == 60

                # Test default timeout
                await provider.request("GET", "https://api.example.com/test", apply_auth=False)
                call_args = mock_request.call_args
                assert call_args[1]["timeout"] == 30

    async def test_authentication_failure_raises_exception(self, provider):
        """Test that authentication failure properly raises an exception."""

        with patch.object(provider, 'authenticate', new_callable=AsyncMock) as mock_auth:
            # Mock authentication to throw an exception
            mock_auth.side_effect = Exception("Authentication failed - invalid credentials")

            # Verify that the exception is raised and not swallowed
            with pytest.raises(Exception, match="Authentication failed - invalid credentials"):
                await provider.request("GET", "https://api.example.com/test", user_id="test-user", apply_auth=True)

            # Verify authentication was called
            mock_auth.assert_called_once_with(user_id="test-user")
