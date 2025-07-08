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
import threading
import time
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from enum import Enum
from urllib.parse import parse_qs
from urllib.parse import urlencode
from urllib.parse import urlparse

import httpx
from flask import Flask
from flask import jsonify
from flask import redirect
from flask import request

from aiq.authentication.interfaces import OAuthClientBase
from aiq.authentication.oauth2.oauth_user_consent_base_config import OAuthUserConsentConfigBase
from aiq.data_models.authentication import OAuth2AuthorizationQueryParams


class OAuth2Flow(Enum):
    """Enum representing different OAuth2.0 flow types."""
    AUTHORIZATION_CODE = "authorization_code"
    AUTHORIZATION_CODE_PKCE = "authorization_code_pkce"
    CLIENT_CREDENTIALS = "client_credentials"
    DEVICE_AUTHORIZATION = "device_code"


class MockOAuth2Client:
    """Mock OAuth2 client for testing purposes."""

    auth_route = "/auth"
    redirect_route = "/redirect"

    def __init__(self, client_id: str, client_secret: str, base_url: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = f"{base_url}{self.auth_route}{self.redirect_route}"
        self.response_types = ["code"]

    @staticmethod
    async def mock_redirect_uri(response) -> str:
        """
        Mock redirect URI for testing purposes.
        Extracts authorization code from OAuth2 server redirect response.

        Args:
            response: The HTTP response from the authorization endpoint

        Returns:
            str: The authorization code extracted from the redirect

        Raises:
            Exception: If the response is not a valid redirect or no auth code is found
        """
        if response.status_code != 302:
            raise Exception(f"Expected redirect response (302), got {response.status_code}")

        location = response.headers.get('Location')
        if not location:
            raise Exception("No Location header in redirect response")

        parsed_url = urlparse(location)
        auth_code = parse_qs(parsed_url.query).get('code', [None])[0]

        if not auth_code:
            raise Exception("Mock server did not return authorization code")

        return auth_code

    def generate_authorization_code(self, scope: str, user_id: str, redirect_uri: str) -> dict:
        """
        Generate a mock authorization code with metadata.

        Args:
            scope: The requested scope
            user_id: The user ID
            redirect_uri: The specific redirect URI used in the authorization request

        Returns:
            dict: Authorization code data including code, expiry, and metadata
        """
        auth_code = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

        return {
            'code': auth_code,
            'client_id': self.client_id,
            'redirect_uri': redirect_uri,
            'scope': scope,
            'user_id': user_id,
            'expires_at': expires_at.timestamp(),
            'used': False
        }


class MockOAuth2Token:
    """Mock OAuth2 token for testing purposes."""

    def __init__(self, access_token: str, client_id: str, scope: str, user_id: str | None = None):
        self.access_token = access_token
        self.client_id = client_id
        self.scope = scope
        self.user_id = user_id
        self.expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        self.refresh_token = secrets.token_urlsafe(32)
        self.token_type = "Bearer"


class MockOAuth2Server:
    """
    Standards-compliant OAuth 2.0 mock server using Authlib.

    Implements multiple OAuth flows for comprehensive testing of OAuth client implementations.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = secrets.token_hex(16)
        self.server_thread = None
        self.is_running = False

        self.clients: dict[str, MockOAuth2Client] = {}
        self.authorization_codes: dict[str, dict] = {}
        self.tokens: dict[str, MockOAuth2Token] = {}

        self._setup_routes()

    def register_client(self, client_id: str, client_secret: str, base_url: str):
        """Register a new OAuth2 client."""
        client = MockOAuth2Client(client_id, client_secret, base_url)
        self.clients[client_id] = client
        return client

    def _setup_routes(self):
        """Setup OAuth2 endpoint routes."""

        @self.app.route('/oauth/authorize')
        def authorize():
            """Authorization endpoint - handles authorization requests."""
            client_id = request.args.get('client_id')
            redirect_uri = request.args.get('redirect_uri')
            response_type = request.args.get('response_type')
            scope = request.args.get('scope', 'read')
            state = request.args.get('state')

            # Validate client
            if client_id not in self.clients:
                return jsonify({'error': 'invalid_client'}), 400

            client = self.clients[client_id]

            # Validate redirect URI
            if redirect_uri and (redirect_uri != client.redirect_uri):
                return jsonify({'error': 'invalid_redirect_uri'}), 400

            # Validate response type
            if response_type not in client.response_types:
                return jsonify({'error': 'unsupported_response_type'}), 400

            if response_type == 'code':
                if not redirect_uri:
                    return jsonify({'error': 'invalid_request'}), 400

                auth_code = secrets.token_urlsafe(32)
                self.authorization_codes[auth_code] = client.generate_authorization_code(scope=scope,
                                                                                         user_id="user123",
                                                                                         redirect_uri=redirect_uri)

                redirect_params = {'code': auth_code}
                if state:
                    redirect_params['state'] = state

                redirect_url = "{0}?{1}".format(redirect_uri, urlencode(redirect_params))
                return redirect(redirect_url)

            return jsonify({'error': 'unsupported_response_type'}), 400

        @self.app.route('/oauth/token', methods=['POST'])
        def token():
            """Token endpoint - handles token requests."""
            if request.content_type == 'application/json':
                data = request.get_json() or {}
                grant_type = data.get('grant_type')
            else:
                data = request.form
                grant_type = request.form.get('grant_type')

            if grant_type == 'authorization_code':
                return self._handle_authorization_code_grant(data)
            elif grant_type == 'refresh_token':
                return self._handle_refresh_token_grant(data)
            elif grant_type == 'client_credentials':
                return self._handle_client_credentials_grant(data)
            else:
                return jsonify({'error': 'unsupported_grant_type'}), 400

        @self.app.route('/oauth/introspect', methods=['POST'])
        def introspect():
            """Token introspection endpoint."""
            token = request.form.get('token')

            if token in self.tokens:
                token_obj = self.tokens[token]
                if token_obj.expires_at.timestamp() > time.time():
                    return jsonify({
                        'active': True,
                        'client_id': token_obj.client_id,
                        'scope': token_obj.scope,
                        'token_type': token_obj.token_type,
                        'exp': token_obj.expires_at.timestamp()
                    })

            return jsonify({'active': False})

        @self.app.route('/oauth/revoke', methods=['POST'])
        def revoke():
            """Token revocation endpoint."""
            token = request.form.get('token')

            if token in self.tokens:
                del self.tokens[token]
                return '', 200

            return '', 200

    def _handle_authorization_code_grant(self, data):
        """Handle authorization code grant flow."""
        code = data.get('code')
        client_id = data.get('client_id')
        client_secret = data.get('client_secret')
        redirect_uri = data.get('redirect_uri')

        # Validate authorization code
        if code not in self.authorization_codes:
            return jsonify({'error': 'invalid_grant'}), 400

        auth_code_data = self.authorization_codes[code]

        # Validate code hasn't expired
        if auth_code_data['expires_at'] < time.time():
            del self.authorization_codes[code]
            return jsonify({'error': 'invalid_grant'}), 400

        # Validate client
        if client_id != auth_code_data['client_id']:
            return jsonify({'error': 'invalid_client'}), 400

        if client_id not in self.clients:
            return jsonify({'error': 'invalid_client'}), 400

        client = self.clients[client_id]
        if client.client_secret != client_secret:
            return jsonify({'error': 'invalid_client'}), 400

        # Validate redirect URI
        if redirect_uri != auth_code_data['redirect_uri']:

            return jsonify({'error': 'invalid_grant'}), 400

        # Generate access token
        access_token = secrets.token_urlsafe(32)
        token_obj = MockOAuth2Token(access_token=access_token,
                                    client_id=client_id,
                                    scope=auth_code_data['scope'],
                                    user_id=auth_code_data['user_id'])

        self.tokens[access_token] = token_obj

        # Clean up authorization code (one-time use)
        del self.authorization_codes[code]

        return jsonify({
            'access_token': access_token,
            'token_type': 'Bearer',
            'expires_in': 3600,
            'refresh_token': token_obj.refresh_token,
            'scope': auth_code_data['scope']
        })

    def _handle_refresh_token_grant(self, data):
        """Handle refresh token grant flow."""
        refresh_token = data.get('refresh_token')
        client_id = data.get('client_id')
        client_secret = data.get('client_secret')

        # Find token by refresh token
        token_obj = None
        for token in self.tokens.values():
            if token.refresh_token == refresh_token and token.client_id == client_id:
                token_obj = token
                break

        if not token_obj:
            return jsonify({'error': 'invalid_grant'}), 400

        # Validate client
        if client_id not in self.clients:
            return jsonify({'error': 'invalid_client'}), 400

        client = self.clients[client_id]
        if client.client_secret != client_secret:
            return jsonify({'error': 'invalid_client'}), 400

        # Generate new access token
        new_access_token = secrets.token_urlsafe(32)
        new_token_obj = MockOAuth2Token(access_token=new_access_token,
                                        client_id=client_id,
                                        scope=token_obj.scope,
                                        user_id=token_obj.user_id)

        # Remove old token and store new one
        old_access_token = token_obj.access_token
        if old_access_token in self.tokens:
            del self.tokens[old_access_token]

        self.tokens[new_access_token] = new_token_obj

        return jsonify({
            'access_token': new_access_token,
            'token_type': 'Bearer',
            'expires_in': 3600,
            'refresh_token': new_token_obj.refresh_token,
            'scope': token_obj.scope
        })

    def _handle_client_credentials_grant(self, data):
        """Handle client credentials grant flow."""
        client_id = data.get('client_id')
        client_secret = data.get('client_secret')
        scope = data.get('scope', 'read')

        # Validate client
        if client_id not in self.clients:
            return jsonify({'error': 'invalid_client'}), 400

        client = self.clients[client_id]
        if client.client_secret != client_secret:
            return jsonify({'error': 'invalid_client'}), 400

        # Generate access token
        access_token = secrets.token_urlsafe(32)
        token_obj = MockOAuth2Token(access_token=access_token, client_id=client_id, scope=scope)

        self.tokens[access_token] = token_obj

        return jsonify({'access_token': access_token, 'token_type': 'Bearer', 'expires_in': 3600, 'scope': scope})

    def start_server(self, threaded: bool = False):
        """Start the mock OAuth2 server."""
        if threaded:
            self.server_thread = threading.Thread(target=self._run_server)
            self.server_thread.daemon = True
            self.server_thread.start()
            time.sleep(0.5)
            self.is_running = True
        else:
            self._run_server()

    def _run_server(self):
        """Run the Flask server."""
        self.app.run(host=self.host, port=self.port, debug=False)

    def stop_server(self):
        """Stop the mock OAuth2 server."""
        self.is_running = False

    def get_base_url(self):
        """Get the base URL of the mock server."""
        return f"http://{self.host}:{self.port}"

    def get_authorization_url(self):
        """Get the authorization endpoint URL."""
        return f"{self.get_base_url()}/oauth/authorize"

    def get_token_url(self):
        """Get the token endpoint URL."""
        return f"{self.get_base_url()}/oauth/token"


class OAuth2FlowTester:
    """
    Comprehensive OAuth2 flow testing class.

    Tests OAuth client implementations against the mock server.
    Handles all setup, execution, and validation of OAuth2 flows.
    """

    def __init__(self, oauth_client: OAuthClientBase, flow: OAuth2Flow):

        assert isinstance(oauth_client, OAuthClientBase), \
            "OAuth client must be an instance of OAuthClientBase"

        assert isinstance(oauth_client.config, OAuthUserConsentConfigBase), \
            ("Config must be of type OAuthUserConsentConfigBase, "
                f"got {type(oauth_client.config).__name__}")

        self._oauth_client = oauth_client
        self._flow = flow
        self._mock_server = MockOAuth2Server()

    async def run(self):
        """
        Main entry point for running the complete OAuth2 flow test.

        Handles all setup, execution, and validation:
        - Creates and configures the appropriate OAuth client
        - Creates and configures mock OAuth2 server
        - Registers test client
        - Runs OAuth2 flow tests
        - Validates results
        - Cleans up resources

        Returns:
            bool: True if all tests passed, False otherwise
        """
        try:
            if self._flow != OAuth2Flow.AUTHORIZATION_CODE:
                raise NotImplementedError(f"Flow {self._flow.value} not yet implemented")

            # Setup client with properly configured config
            await self._setup_oauth_client()

            # Create mock server
            await self._create_mock_server()

            # Run OAuth2 flow tests - will raise exception if any assertion fails
            await self._run_oauth2_tests()

            return True

        except Exception as e:
            import logging
            logging.error("OAuth2 Flow Test Suite failed: %s", str(e))
            return False

        finally:
            await self._cleanup()

    async def _setup_oauth_client(self):
        """Setup and configure the OAuth client based on the flow type."""

        # Check if the config has the required attributes for OAuthUserConsentConfigBase
        required_attrs = ['client_id', 'client_secret', 'audience', 'scope']
        missing_attrs = [attr for attr in required_attrs if not hasattr(self._oauth_client.config, attr)]

        if missing_attrs:
            raise AssertionError(f"Invalid config: Missing required attributes {missing_attrs}. "
                                 f"The config must have all required attributes for OAuth2 testing. "
                                 f"Config type: {type(self._oauth_client.config).__name__}")

        # Update oauth client config parameters to match mock server parameters
        if hasattr(self._oauth_client.config, 'authorization_url'):
            setattr(self._oauth_client.config, 'authorization_url', self._mock_server.get_authorization_url())
        if hasattr(self._oauth_client.config, 'authorization_token_url'):
            setattr(self._oauth_client.config, 'authorization_token_url', self._mock_server.get_token_url())
        if hasattr(self._oauth_client.config, 'client_server_url'):
            setattr(self._oauth_client.config, 'client_server_url', self._mock_server.get_base_url())
        if hasattr(self._oauth_client.config, 'client_secret'):
            setattr(self._oauth_client.config, 'client_secret', 'test_client_secret')

    async def _create_mock_server(self):
        """Create and configure the mock OAuth2 server."""

        mock_base_url = self._mock_server.get_base_url()

        if not self._oauth_client.config:
            raise Exception("OAuth client config is None")

        self._mock_server.register_client(client_id=self._oauth_client.config.client_id,
                                          client_secret=self._oauth_client.config.client_secret,
                                          base_url=mock_base_url)

        self._mock_server.start_server(threaded=True)
        time.sleep(1)

    async def _run_oauth2_tests(self):
        """Run the OAuth2 flow tests."""
        if self._flow == OAuth2Flow.AUTHORIZATION_CODE:
            await self.test_authorization_code_grant_flow()
        else:
            raise NotImplementedError(f"Flow {self._flow.value} not implemented yet")

    async def _cleanup(self):
        """Cleanup resources."""

        if self._mock_server and self._mock_server.is_running:
            self._mock_server.stop_server()

    async def test_authorization_code_grant_flow(self):
        """Test Authorization Code Grant flow interface compliance."""
        # Test authorization request query parameters
        query_params = await self.test_authorization_request_query_params("code", "consent")

        # Test sending authorization request to receive an authorization code from the oauth2 server
        response = await self.test_send_authorization_request(self._mock_server.get_authorization_url(), query_params)

        # Extract authorization code from redirect response using mock_redirect_uri
        auth_code = await MockOAuth2Client.mock_redirect_uri(response)

        # Test sending token request to receive an access token from the oauth2 server
        token_response = await self.test_send_token_request(
            client_authorization_path=MockOAuth2Client.auth_route,
            client_authorization_endpoint=MockOAuth2Client.redirect_route,
            authorization_code=auth_code)

        # Process token response and validate the response data was properly persisted in the config
        await self.test_process_token_response(token_response)

    async def test_process_token_response(self, response: httpx.Response):
        """Test OAuth2 token response processing."""

        # Process the token response and ensure no errors are raised
        try:
            await self._oauth_client.process_token_response(response)
        except Exception as e:
            # If any error occurs during token processing, fail the test with detailed error message
            raise AssertionError(f"Token response processing failed: {str(e)}") from e

        # Test that the response data was properly persisted in the config
        if response.status_code != 200:
            raise AssertionError(f"Token response failed with status code {response.status_code}: {response.text}")

        token_data = response.json()

        # Assert that the access token matches the one the mock server returned
        if self._oauth_client.config and hasattr(self._oauth_client.config, 'access_token'):
            expected_token = token_data.get('access_token')
            assert self._oauth_client.config.access_token == expected_token, \
                f"Access token mismatch: expected {expected_token}, got {self._oauth_client.config.access_token}"

        # Assert that the access token expiry matches the one the mock server returned
        if self._oauth_client.config and hasattr(self._oauth_client.config, 'access_token_expires_in'):
            assert self._oauth_client.config.access_token_expires_in is not None, \
                "Access token expiry should be set based on mock server response"

        # Assert that the refresh token matches the one the mock server returned
        if self._flow in [OAuth2Flow.AUTHORIZATION_CODE, OAuth2Flow.AUTHORIZATION_CODE_PKCE]:
            expected_refresh_token = token_data.get('refresh_token')
            if (expected_refresh_token and self._oauth_client.config
                    and hasattr(self._oauth_client.config, 'refresh_token')):
                assert self._oauth_client.config.refresh_token == expected_refresh_token, \
                    (f"Refresh token mismatch: expected {expected_refresh_token}, "
                     f"got {self._oauth_client.config.refresh_token}")

        # Assert that the credentials are valid
        assert await self._oauth_client.validate_credentials(), "Credentials should be valid"

    async def test_authorization_request_query_params(self, response_type: str,
                                                      prompt: str) -> OAuth2AuthorizationQueryParams:
        """Test OAuth2 authorization query parameters construction."""

        query_params = self._oauth_client.construct_authorization_query_params(response_type=response_type,
                                                                               prompt=prompt)

        # Validate query parameters
        if self._oauth_client.config:
            assert query_params.client_id == self._oauth_client.config.client_id, \
                f"Expected client_id {self._oauth_client.config.client_id}, got {query_params.client_id}"

            if hasattr(self._oauth_client.config, 'audience'):
                assert query_params.audience == self._oauth_client.config.audience, \
                    f"Expected audience {self._oauth_client.config.audience}, got {query_params.audience}"

            if hasattr(self._oauth_client.config, 'state'):
                assert query_params.state == self._oauth_client.config.state, \
                    f"Expected state {self._oauth_client.config.state}, got {query_params.state}"

            if hasattr(self._oauth_client.config, 'scope'):
                expected_scope = " ".join(self._oauth_client.config.scope)
                assert query_params.scope == expected_scope, \
                    f"Expected scope {expected_scope}, got {query_params.scope}"

        assert query_params.redirect_uri is not None, "Redirect URI should not be None"
        assert query_params.response_type == response_type, \
            f"Expected response_type {response_type}, got {query_params.response_type}"
        assert query_params.prompt == prompt, \
            f"Expected prompt {prompt}, got {query_params.prompt}"

        return query_params

    async def test_send_authorization_request(self,
                                              authorization_url: str,
                                              authorization_query_params: OAuth2AuthorizationQueryParams):
        """Test OAuth2 authorization request sending."""

        response = await self._oauth_client.send_authorization_request(authorization_url, authorization_query_params)
        assert response is not None, "Authorization request should return a response"
        # For OAuth flows, 302 redirect is expected for authorization endpoint
        assert response.status_code in [200, 302], \
            f"Authorization request should return 200 or 302, got {response.status_code}"
        return response

    async def test_send_token_request(self,
                                      client_authorization_path: str,
                                      client_authorization_endpoint: str,
                                      authorization_code: str):
        """Test OAuth2 token request sending."""

        response = await self._oauth_client.send_token_request(client_authorization_path,
                                                               client_authorization_endpoint,
                                                               authorization_code)
        assert response is not None, "Token request should return a response"
        assert response.status_code == 200, \
            f"Token request should return 200, got {response.status_code}"
        return response
