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

import secrets
import string
from urllib.parse import parse_qs
from urllib.parse import urlparse

import httpx
import pytest
from authlib.integrations.httpx_client import AsyncOAuth2Client
from httpx import ASGITransport
from mock_oauth2_server import MockOAuth2Server

from nat.authentication.oauth2.oauth2_auth_code_flow_provider_config import OAuth2AuthCodeFlowProviderConfig


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture
def oauth_config() -> OAuth2AuthCodeFlowProviderConfig:
    """OAuth2 configuration for testing."""
    return OAuth2AuthCodeFlowProviderConfig(client_id="test_client",
                                            client_secret="test_secret",
                                            authorization_url="http://testserver/oauth/authorize",
                                            token_url="http://testserver/oauth/token",
                                            redirect_uri="https://app.example.com/auth/redirect",
                                            scopes=["read", "write"],
                                            use_pkce=True)


@pytest.fixture
def mock_server(oauth_config) -> MockOAuth2Server:
    """Mock OAuth2 server with registered client."""
    srv = MockOAuth2Server(host="testserver", port=0)
    # Register client using config values
    srv.register_client(client_id=oauth_config.client_id,
                        client_secret=oauth_config.client_secret,
                        redirect_base="https://app.example.com")
    return srv


# --------------------------------------------------------------------------- #
# Redirect URI validation
# --------------------------------------------------------------------------- #
class TestOAuth2RedirectURIValidation:
    """Test OAuth2 redirect URI validation using actual authorization endpoint."""

    async def test_valid_redirect_uri(self, mock_server, oauth_config):
        """Positive test: Valid redirect URI should return 302 redirect to exact URI."""
        # Create client with transport
        transport = ASGITransport(app=mock_server._app)
        oauth_client = AsyncOAuth2Client(
            client_id=oauth_config.client_id,
            client_secret=oauth_config.client_secret,
            redirect_uri=oauth_config.redirect_uri,
            scope=" ".join(oauth_config.scopes) if oauth_config.scopes else None,
            token_endpoint=oauth_config.token_url,
            base_url="http://testserver",
            transport=transport,
        )

        # Generate authorization URL
        authorization_url, _ = oauth_client.create_authorization_url(
            "http://testserver/oauth/authorize",
            state="test_state"
        )

        # Make request to authorization URL
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            follow_redirects=False,
        ) as client:
            response = await client.get(authorization_url)

            # Positive assertion: 302 redirect to exact URI
            assert response.status_code == 302
            assert response.headers["location"].startswith(oauth_config.redirect_uri)

    @pytest.mark.parametrize("malicious_redirect_uri",
                             [
                                 "https://evil.example.com/auth/redirect",
                                 "http://app.example.com/auth/redirect",
                                 "https://app.example.com/auth/redirect/extra",
                             ])
    async def test_invalid_redirect_uri(self, mock_server, oauth_config, malicious_redirect_uri):
        """Negative tests: Invalid redirect URI variations should not be redirected to."""

        # Create client with malicious redirect URI
        transport = ASGITransport(app=mock_server._app)
        malicious_client = AsyncOAuth2Client(
            client_id=oauth_config.client_id,
            client_secret=oauth_config.client_secret,
            redirect_uri=malicious_redirect_uri,  # Use malicious URI
            scope=" ".join(oauth_config.scopes) if oauth_config.scopes else None,
            token_endpoint=oauth_config.token_url,
            base_url="http://testserver",
            transport=transport,
        )

        # Generate authorization URL with malicious redirect URI
        authorization_url, _ = malicious_client.create_authorization_url(
            "http://testserver/oauth/authorize",
            state="test_state"
        )

        # Make request to authorization URL
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            follow_redirects=False,
        ) as client:
            response = await client.get(authorization_url)

            # Negative assertion: not redirected to the mismatched URI
            assert response.status_code >= 400
            # Ensure no redirect to malicious URI
            if "location" in response.headers:
                assert not response.headers["location"].startswith(malicious_redirect_uri)

    @pytest.mark.parametrize(
        "attack_uri,attack_description",
        [
            # URL encoding attacks
            ("https://app.example.com/auth/redirect%2F..%2F..%2Fevil", "path traversal via URL encoding"),
            ("https://app.example.com/auth/redirect%2F%2E%2E%2F%2E%2E%2Fevil", "double-dot path traversal"),
            ("https://app.example.com/auth/redirect%252F..%252F..%252Fevil", "double URL encoding"),
            ("https://app.example.com/auth/redirect%3F/evil", "encoded query separator"),
            ("https://app.example.com/auth/redirect%23evil", "encoded fragment"),
            ("https://app.example.com/auth/redirect%2Fevil%2Fpath", "encoded path separators"),

            # Advanced subdomain attacks
            ("https://app.example.com.evil.com/auth/redirect", "subdomain attack - legitimate.evil.com"),
            ("https://nat.nvidia.com.evil.com/auth/redirect", "nvidia subdomain attack"),
            ("https://app-example-com.evil.com/auth/redirect", "dash-separated domain attack"),
            ("https://appexample.com/auth/redirect", "typosquatting domain"),
            ("https://app.exampl3.com/auth/redirect", "character substitution attack"),

            # Scheme manipulation
            ("http://app.example.com/auth/redirect", "scheme downgrade attack"),
            ("javascript://app.example.com/auth/redirect", "javascript scheme injection"),
            ("data://app.example.com/auth/redirect", "data scheme injection"),
            ("file://app.example.com/auth/redirect", "file scheme injection"),

            # Unicode and IDN attacks
            ("https://app.еxample.com/auth/redirect", "cyrillic character substitution"),
            ("https://app.example.com/auth/rеdirect", "cyrillic path attack"),
            ("https://xn--app-example-com.evil.com/auth/redirect", "punycode domain attack"),

            # Port manipulation attacks
            ("https://app.example.com:80/auth/redirect", "wrong port for https"),
            ("https://app.example.com:443:8080/auth/redirect", "port confusion attack"),
            ("https://app.example.com.:8080/auth/redirect", "trailing dot port attack"),

            # Host header confusion
            ("https://evil.com@app.example.com/auth/redirect", "user info attack"),
            ("https://app.example.com\\@evil.com/auth/redirect", "backslash confusion"),
            ("https://app.example.com%40evil.com/auth/redirect", "encoded @ attack"),

            # IPv6/IPv4 confusion
            ("https://[::1]:443/auth/redirect", "IPv6 localhost"),
            ("https://127.0.0.1/auth/redirect", "IPv4 localhost attack"),
            ("https://0x7f000001/auth/redirect", "hex IP attack"),
            ("https://2130706433/auth/redirect", "decimal IP attack"),

            # Case normalization attacks
            ("HTTPS://APP.EXAMPLE.COM/AUTH/REDIRECT", "uppercase scheme and domain"),
            ("https://APP.EXAMPLE.COM/auth/redirect", "uppercase domain only"),
            ("https://app.EXAMPLE.com/AUTH/redirect", "mixed case attack"),

            # Additional edge cases
            ("https://App.Example.com/auth/redirect", "case change in domain"),
            ("https://app.example.com/Auth/Redirect", "case change in path"),
            ("https://app.example.com:8443/auth/redirect", "added port number"),
            ("https://app.example.com/auth/redirect/", "added trailing slash"),
            ("https://app.example.com/auth/redirect#evil", "added fragment"),
            ("https://app.example.com/../evil.com/redirect", "path traversal attack"),
            ("https://phishing.app.example.com/auth/redirect", "subdomain attack"),
            ("https://app.example.com/auth/redirect?evil=true", "added query parameter"),
        ])
    async def test_uri_validation_logic(self, mock_server, oauth_config, attack_uri, attack_description):
        """Test comprehensive URI validation against various attack vectors."""
        # Create client with attack URI
        transport = ASGITransport(app=mock_server._app)
        attack_client = AsyncOAuth2Client(
            client_id=oauth_config.client_id,
            client_secret=oauth_config.client_secret,
            redirect_uri=attack_uri,
            scope=" ".join(oauth_config.scopes) if oauth_config.scopes else None,
            token_endpoint=oauth_config.token_url,
            base_url="http://testserver",
            transport=transport,
        )

        # Generate authorization URL with attack URI
        authorization_url, _ = attack_client.create_authorization_url(
            "http://testserver/oauth/authorize",
            state="test_state"
        )

        # Make request to authorization URL
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            follow_redirects=False,
        ) as client:
            response = await client.get(authorization_url)

            # Should reject all attack vectors
            assert response.status_code >= 400
            # Ensure no redirect to attack URI
            if "location" in response.headers:
                location = response.headers["location"]
                assert not location.startswith(attack_uri)
                # Also check for decoded versions
                from urllib.parse import unquote
                decoded_attack = unquote(attack_uri)
                assert not location.startswith(decoded_attack)


# --------------------------------------------------------------------------- #
# Authorization Request                                                       #
# --------------------------------------------------------------------------- #
class TestOAuth2AuthorizationRequest:
    """Test OAuth2 authorization request parameter validation."""

    async def test_valid_response_type_code(self, mock_server, oauth_config):
        """Valid test: response_type=code should be accepted."""
        # Create client with transport
        transport = ASGITransport(app=mock_server._app)
        oauth_client = AsyncOAuth2Client(
            client_id=oauth_config.client_id,
            client_secret=oauth_config.client_secret,
            redirect_uri=oauth_config.redirect_uri,
            scope=" ".join(oauth_config.scopes) if oauth_config.scopes else None,
            token_endpoint=oauth_config.token_url,
            base_url="http://testserver",
            transport=transport,
        )

        # Generate authorization URL
        authorization_url, _ = oauth_client.create_authorization_url(
            "http://testserver/oauth/authorize",
            state="test_state"
        )

        # Parse the authorization URL to verify response_type=code parameter

        parsed_url = urlparse(authorization_url)
        params = parse_qs(parsed_url.query)

        # Verify response_type parameter exists and equals "code"
        assert "response_type" in params
        assert params["response_type"][0] == "code"

    async def test_scope_parameter_formatting(self, mock_server, oauth_config):
        """Test that scope parameters are properly formatted in authorization requests."""
        # Create client with transport
        transport = ASGITransport(app=mock_server._app)
        oauth_client = AsyncOAuth2Client(
            client_id=oauth_config.client_id,
            client_secret=oauth_config.client_secret,
            redirect_uri=oauth_config.redirect_uri,
            scope=" ".join(oauth_config.scopes) if oauth_config.scopes else None,
            token_endpoint=oauth_config.token_url,
            base_url="http://testserver",
            transport=transport,
        )

        # Generate authorization URL using configured scopes
        authorization_url, _ = oauth_client.create_authorization_url(
            "http://testserver/oauth/authorize",
            state="test_state"
        )

        # Verify scope parameter is properly formatted in the authorization URL.

        parsed_url = urlparse(authorization_url)
        params = parse_qs(parsed_url.query)

        # Validate scope parameter exists and is properly formatted
        assert 'scope' in params
        scope_param = params['scope'][0] if 'scope' in params else ''
        actual_scopes = scope_param.split(' ') if scope_param else []

        # Verify scopes match config
        assert actual_scopes == oauth_config.scopes

    async def test_state_parameter_compliance(self, mock_server, oauth_config):
        """Test that state parameter meets OAuth2 RFC 6749 compliance requirements."""
        # Create client with transport
        transport = ASGITransport(app=mock_server._app)
        oauth_client = AsyncOAuth2Client(
            client_id=oauth_config.client_id,
            client_secret=oauth_config.client_secret,
            redirect_uri=oauth_config.redirect_uri,
            scope=" ".join(oauth_config.scopes) if oauth_config.scopes else None,
            token_endpoint=oauth_config.token_url,
            base_url="http://testserver",
            transport=transport,
        )

        # Generate state parameter same way as production code (secrets.token_urlsafe(16))
        state = secrets.token_urlsafe(16)

        # Generate authorization URL with cryptographically secure state parameter
        authorization_url, returned_state = oauth_client.create_authorization_url(
            "http://testserver/oauth/authorize",
            state=state
        )

        # Parse URL to extract state parameter
        parsed_url = urlparse(authorization_url)
        params = parse_qs(parsed_url.query)

        # Validate state parameter exists
        assert 'state' in params
        state_param = params['state'][0]

        # State parameter should be unguessable value with sufficient entropy
        assert len(state_param) >= 20

        # Verify URL-safe characters (secrets.token_urlsafe uses URL-safe base64: A-Z, a-z, 0-9, -, _)
        url_safe_base64_chars = string.ascii_letters + string.digits + '-_'
        assert all(c in url_safe_base64_chars for c in state_param)

        # Verify state parameter has sufficient entropy (secrets.token_urlsafe provides crypto randomness)
        # 16 bytes = 128 bits of entropy, well above minimum recommendations (>= 128 bits)
        assert len(state_param) == 22

        # Verify state parameter appears random (should have good character distribution)
        unique_chars = len(set(state_param))
        assert unique_chars >= 10

        # Verify the state parameter matches what we generated
        assert state_param == state

        # Verify state parameter is properly encoded in URL
        assert returned_state == state

    async def test_validate_state_parameter_generation_and_usage(self, mock_server, oauth_config):
        """Test state parameter generation and usage exactly as implemented in source code."""
        state = secrets.token_urlsafe(16)

        # Create client and transport exactly as source does
        transport = ASGITransport(app=mock_server._app)
        oauth_client = AsyncOAuth2Client(
            client_id=oauth_config.client_id,
            client_secret=oauth_config.client_secret,
            redirect_uri=oauth_config.redirect_uri,
            scope=" ".join(oauth_config.scopes) if oauth_config.scopes else None,
            token_endpoint=oauth_config.token_url,
            base_url="http://testserver",
            transport=transport,
        )

        authorization_url, returned_state = oauth_client.create_authorization_url(
            "http://testserver/oauth/authorize",
            state=state
        )

        # Simulate the authorization flow by making the authorization request
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            follow_redirects=False,
        ) as client:

            # Step 1: Authorization request (like user clicking the auth URL)
            auth_response = await client.get(authorization_url)
            assert auth_response.status_code == 302

            # Step 2: Extract redirect location (simulates auth server redirect)
            redirect_location = auth_response.headers["location"]
            assert redirect_location.startswith(oauth_config.redirect_uri)

            # Step 3: Parse redirect to validate state parameter usage
            redirect_parsed = urlparse(redirect_location)
            redirect_params = parse_qs(redirect_parsed.query)

            # Verify state parameter is preserved in the redirect (security requirement)
            assert 'state' in redirect_params
            redirected_state = redirect_params['state'][0]
            assert redirected_state == state


# ========================================
# Error Handling and Recovery Tests
# ========================================


class TestOAuth2ErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms for OAuth2 flows."""

    @pytest.mark.parametrize("test_case_name,fake_client_id,fake_scope,fake_response_type",
                             [
                                 ("invalid_request", "", "", ""),
                                 ("unauthorized_client", "unauthorized_client", "read", "code"),
                                 ("invalid_scope", "test_client", "invalid_scope", "code"),
                                 ("invalid_grant", "test_client", "read", "invalid_grant"),
                                 ("invalid_client", "nonexistent_client", "read", "code"),
                                 ("unsupported_grant_type", "test_client", "read", "unsupported_type"),
                             ])
    async def test_invalid_request_handling(self,
                                            mock_server,
                                            oauth_config,
                                            test_case_name,
                                            fake_client_id,
                                            fake_scope,
                                            fake_response_type):
        """Test that OAuth2 errors are properly caught and handled."""
        transport = ASGITransport(app=mock_server._app)

        # Test that invalid requests raise an exception with response status
        with pytest.raises(Exception) as exc_info:
            oauth_client = AsyncOAuth2Client(
                client_id=fake_client_id or oauth_config.client_id,
                client_secret=oauth_config.client_secret,
                redirect_uri=oauth_config.redirect_uri,
                scope=fake_scope,
                base_url="http://testserver",
                transport=transport,
            )

            # Use parameters that will trigger specific error codes
            authorization_url, _ = oauth_client.create_authorization_url(
                "http://testserver/oauth/authorize",
                response_type=fake_response_type or "code",
                state="test_state",
                client_id=fake_client_id if fake_client_id else None,
                scope=fake_scope if fake_scope else None
            )

            async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                    follow_redirects=False,
            ) as client:
                response = await client.get(authorization_url)

                # Check if response is an error and raise exception
                if response.status_code in [400, 401, 403]:
                    raise Exception(f"OAuth error: {response.status_code}")

        # Assert that an exception was raised
        assert exc_info.value is not None


# --------------------------------------------------------------------------- #
# Authorization Code Security Handling                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "test_case, setup_behavior, expected_error_detail, description",
    [
        ("invalid_code",
         "use_invalid_code",
         "invalid_grant",
         "Invalid authorization code should return invalid_grant error"),
        ("code_reuse", "reuse_authorization_code", "invalid_grant", "Code reuse should return invalid_grant error"),
        ("invalid_client",
         "use_invalid_client",
         "invalid_grant",
         "Invalid client credentials should return invalid_grant error"),
        ("expired_code", "use_expired_code", "invalid_grant", "Expired code should return invalid_grant error"),
    ],
)
@pytest.mark.asyncio
async def test_authorization_code_security_handling(mock_server,
                                                    oauth_config,
                                                    test_case,
                                                    setup_behavior,
                                                    expected_error_detail,
                                                    description):
    """
    Comprehensive parameterized test for authorization code security handling.
    """

    transport = ASGITransport(app=mock_server._app)

    if setup_behavior == "use_invalid_code":
        # Test with invalid authorization code
        oauth_client = AsyncOAuth2Client(
            client_id=oauth_config.client_id,
            client_secret=oauth_config.client_secret,
            redirect_uri=oauth_config.redirect_uri,
            scope=" ".join(oauth_config.scopes),
            token_endpoint=oauth_config.token_url,
            base_url="http://testserver",
            transport=transport,
        )

        # This should return an error response, not raise an exception
        result = await oauth_client.fetch_token(
            url=oauth_config.token_url,
            code="invalid_code_12345",
            redirect_uri=oauth_config.redirect_uri,
        )

        # Check for error in the response
        assert "detail" in result, f"Expected error detail in response for {test_case}"
        assert expected_error_detail in result["detail"], (
            f"Expected {expected_error_detail} in error detail for {test_case}")

    elif setup_behavior == "reuse_authorization_code":
        # Test code reuse - use a code twice
        oauth_client = AsyncOAuth2Client(
            client_id=oauth_config.client_id,
            client_secret=oauth_config.client_secret,
            redirect_uri=oauth_config.redirect_uri,
            scope=" ".join(oauth_config.scopes),
            token_endpoint=oauth_config.token_url,
            base_url="http://testserver",
            transport=transport,
        )

        # Get valid authorization code
        authorization_url, state = oauth_client.create_authorization_url(
            oauth_config.authorization_url,
            state=secrets.token_urlsafe(16)
        )

        async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
                follow_redirects=False,
        ) as client:
            auth_response = await client.get(authorization_url)
            assert auth_response.status_code == 302

            redirect_url = auth_response.headers["location"]
            parsed = urlparse(redirect_url)
            params = parse_qs(parsed.query)
            code_to_reuse = params["code"][0]

        # Use the code once successfully
        first_token = await oauth_client.fetch_token(
            url=oauth_config.token_url,
            code=code_to_reuse,
            redirect_uri=oauth_config.redirect_uri,
        )
        assert "access_token" in first_token

        # Try to reuse the same code - should fail
        result = await oauth_client.fetch_token(
            url=oauth_config.token_url,
            code=code_to_reuse,  # Same code - should be marked as used
            redirect_uri=oauth_config.redirect_uri,
        )

        # Check for error in the response
        assert "detail" in result, f"Expected error detail in response for {test_case}"
        assert expected_error_detail in result["detail"], (
            f"Expected {expected_error_detail} in error detail for {test_case}")

    elif setup_behavior == "use_invalid_client":
        # Test with invalid client credentials
        bad_client = AsyncOAuth2Client(
            client_id="invalid_client_id",
            client_secret="invalid_client_secret",
            redirect_uri=oauth_config.redirect_uri,
            scope=" ".join(oauth_config.scopes),
            token_endpoint=oauth_config.token_url,
            base_url="http://testserver",
            transport=transport,
        )

        result = await bad_client.fetch_token(
            url=oauth_config.token_url,
            code="any_code",  # Code doesn't matter - client is invalid
            redirect_uri=oauth_config.redirect_uri,
        )

        # Check for error in the response
        assert "detail" in result, f"Expected error detail in response for {test_case}"
        assert expected_error_detail in result["detail"], (
            f"Expected {expected_error_detail} in error detail for {test_case}")

    elif setup_behavior == "use_expired_code":
        # Test with expired code
        oauth_client = AsyncOAuth2Client(
            client_id=oauth_config.client_id,
            client_secret=oauth_config.client_secret,
            redirect_uri=oauth_config.redirect_uri,
            scope=" ".join(oauth_config.scopes),
            token_endpoint=oauth_config.token_url,
            base_url="http://testserver",
            transport=transport,
        )

        # Get valid authorization code
        authorization_url, state = oauth_client.create_authorization_url(
            oauth_config.authorization_url,
            state=secrets.token_urlsafe(16)
        )

        async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
                follow_redirects=False,
        ) as client:
            auth_response = await client.get(authorization_url)
            assert auth_response.status_code == 302

            redirect_url = auth_response.headers["location"]
            parsed = urlparse(redirect_url)
            params = parse_qs(parsed.query)
            code_to_expire = params["code"][0]

        # Manually expire the code
        import time
        if code_to_expire in mock_server._codes:
            mock_server._codes[code_to_expire].expires_at = time.time() - 1

        # Try to use expired code
        result = await oauth_client.fetch_token(
            url=oauth_config.token_url,
            code=code_to_expire,
            redirect_uri=oauth_config.redirect_uri,
        )

        # Check for error in the response
        assert "detail" in result, f"Expected error detail in response for {test_case}"
        assert expected_error_detail in result["detail"], (
            f"Expected {expected_error_detail} in error detail for {test_case}")


# --------------------------------------------------------------------------- #
# Security Best Practices                                                    #
# --------------------------------------------------------------------------- #


class TestSecurityBestPractices:
    """Test security best practices implementation."""

    @pytest.mark.asyncio
    async def test_validate_authentication_log_suppression(self, caplog):
        """Test that authentication-related logs are properly suppressed to prevent leaking sensitive data."""
        import logging

        from nat.utils.log_utils import LogFilter

        # Test the LogFilter functionality directly
        filter_obj = LogFilter(["/auth/redirect"])

        # Create test log records
        sensitive_record = logging.LogRecord(name="uvicorn.access",
                                             level=logging.INFO,
                                             pathname="",
                                             lineno=0,
                                             msg="GET /auth/redirect?code=abc123&state=xyz789",
                                             args=(),
                                             exc_info=None)

        normal_record = logging.LogRecord(name="uvicorn.access",
                                          level=logging.INFO,
                                          pathname="",
                                          lineno=0,
                                          msg="GET /api/workflow",
                                          args=(),
                                          exc_info=None)

        # Test filter functionality
        assert filter_obj.filter(sensitive_record) is False, "OAuth callback logs should be filtered out"
        assert filter_obj.filter(normal_record) is True, "Normal API logs should pass through"

        # Test the log suppression mechanism by directly testing the LogFilter behavior
        uvicorn_logger = logging.getLogger("uvicorn.access")
        original_filters = list(uvicorn_logger.filters)  # Save original filters

        try:
            # Add the LogFilter to suppress auth callback logs
            uvicorn_logger.addFilter(filter_obj)

            with caplog.at_level(logging.INFO, logger="uvicorn.access"):
                # Clear any existing captured logs
                caplog.clear()

                # Try to log sensitive auth callback information
                uvicorn_logger.info("GET /auth/redirect?code=sensitive_auth_code&state=sensitive_state")

                # Try to log normal API request
                uvicorn_logger.info("GET /api/workflow")

                # Verify that auth callback logs are suppressed
                oauth_logs = [record for record in caplog.records if "/auth/redirect" in record.message]

                normal_logs = [record for record in caplog.records if "/api/workflow" in record.message]

                assert len(oauth_logs) == 0, f"Auth callback logs should be suppressed, but found: {oauth_logs}"
                assert len(normal_logs) == 1, f"Normal API logs should pass through, but found: {normal_logs}"

                # Test multiple sensitive patterns
                caplog.clear()
                uvicorn_logger.info("POST /auth/redirect with authorization code")
                uvicorn_logger.info("GET /auth/redirect?error=access_denied")

                oauth_logs = [record for record in caplog.records if "/auth/redirect" in record.message]

                assert len(oauth_logs) == 0, "All auth callback logs should be suppressed"

        finally:
            # Restore original filters
            uvicorn_logger.filters.clear()
            for f in original_filters:
                uvicorn_logger.addFilter(f)
