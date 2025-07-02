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

import logging
import typing
from datetime import datetime
from datetime import timedelta
from datetime import timezone

import httpx

from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantFlowError
from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantFlowRefreshTokenError
from aiq.authentication.interfaces import OAuthClientBase
from aiq.authentication.request_manager import RequestManager
from aiq.authentication.response_manager import ResponseManager
from aiq.data_models.authentication import ConsentPromptMode
from aiq.data_models.authentication import HeaderAuthScheme
from aiq.data_models.authentication import HTTPMethod
from aiq.data_models.authentication import OAuth2AuthorizationQueryParams
from aiq.data_models.authentication import OAuth2TokenRequest
from aiq.data_models.authentication import RefreshTokenRequest
from aiq.front_ends.fastapi.fastapi_front_end_controller import _FastApiFrontEndController

logger = logging.getLogger(__name__)

if (typing.TYPE_CHECKING):
    from aiq.authentication.oauth2.auth_code_grant_config import AuthCodeGrantConfig


class AuthCodeGrantClientManager(OAuthClientBase):

    def __init__(self, config: "AuthCodeGrantConfig", config_name: str | None = None) -> None:

        self._config_name: str | None = config_name
        self._config: "AuthCodeGrantConfig" = config
        self._request_manager: RequestManager = RequestManager()
        self._response_manager: ResponseManager = ResponseManager(self)
        self._consent_prompt_mode: ConsentPromptMode | None = None
        self._oauth2_client_server: _FastApiFrontEndController | None = None
        super().__init__()

    @property
    def config_name(self) -> str | None:
        """
        Get the name of the authentication configuration.

        Returns:
            str | None: The name of the authentication configuration, or None if not set.
        """
        return self._config_name

    @config_name.setter
    def config_name(self, config_name: str | None) -> None:
        """
        Set the name of the authentication configuration.

        Args:
            config_name (str | None): The name of the authentication configuration.
        """
        self._config_name = config_name

    @property
    def config(self) -> "AuthCodeGrantConfig | None":
        """
        Get the authentication configuration.

        Returns:
            AuthCodeGrantConfig | None: The authentication configuration, or None if not set.
        """
        return self._config

    @property
    def consent_prompt_mode(self) -> ConsentPromptMode | None:
        """
        Get the consent prompt mode for the OAuth client.

        Returns:
            ConsentPromptMode: The consent prompt mode (BROWSER or FRONTEND).
        """
        return self._consent_prompt_mode

    @consent_prompt_mode.setter
    def consent_prompt_mode(self, consent_prompt_mode: ConsentPromptMode) -> None:
        """
        Set the consent prompt mode for the OAuth client.

        Args:
            consent_prompt_mode (ConsentPromptMode): The consent prompt mode to set.
        """
        self._consent_prompt_mode = consent_prompt_mode

    @property
    def response_manager(self) -> "ResponseManager | None":
        """
        Get the response manager for the authentication manager.

        Returns:
            ResponseManager | None: The response manager or None if not set.
        """
        return self._response_manager

    @response_manager.setter
    def response_manager(self, response_manager: "ResponseManager") -> None:
        """
        Set the response manager for the authentication manager.
        """
        self._response_manager = response_manager

    async def validate_credentials(self) -> bool:
        """
        Validates the Auth Code Grant grant flow authentication credentials and returns True if the credentials are
        valid and False if they are not. To reliably validate Auth Code Grant flow credentials, a request should be sent
        either to the authorization server's introspection endpoint or to a protected API endpoint, monitoring for a 200
        response. Since introspection endpoints are not standardized the most consistent approach is to check whether
        the access is valid token has not expired.

        Returns:
            bool: True if the credentials are valid and False if they are not.
        """

        if (self._config.access_token and (self._config.access_token_expires_in is not None)
                and (datetime.now(timezone.utc) <= self._config.access_token_expires_in)):
            return True
        else:
            return False

    async def initiate_authorization_flow_console(self) -> None:
        """
        Acquires an access token if the token is absent, expired, or revoked,
        by the options listed below.

        1. Initiate the authorization flow to obtain a new access token and optional request token.
        2. Use a refresh token to get another access token and refresh token pair if refresh token is available.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager
        from aiq.authentication.exceptions.call_back_exceptions import OAuthClientConsoleError

        try:
            # Initiate code flow is there is no access token.
            if (self._config.access_token is None):

                # Spawn an authentication client server to handle oauth code flow.
                await self._spawn_oauth_client_server()

                # Initiate oauth code flow by sending authorization request.
                await self._send_authorization_request()

                await _CredentialsManager().wait_for_oauth_credentials()

                if self._oauth2_client_server is not None:
                    await self._oauth2_client_server.stop_server()

            # Initiate refresh token request if the access token is expired or revoked.
            if (self._config.refresh_token and (self._config.access_token_expires_in is not None)
                    and (datetime.now(timezone.utc) >= self._config.access_token_expires_in)):
                await self._get_access_token_with_refresh_token()

            return

        except AuthCodeGrantFlowError as e:
            error_message = (
                f"Failed to complete Authorization Code Grant Flow for: {self._config_name} Error: {str(e)}")
            logger.error(error_message, exc_info=True)
            raise OAuthClientConsoleError(error_code='auth_code_grant_flow_error', message=error_message) from e

        except AuthCodeGrantFlowRefreshTokenError as e:
            logger.error("Failed to get Auth Code Grant flow credentials using refresh token: %s",
                         str(e),
                         exc_info=True)

        except Exception as e:
            logger.error("Failed to get Auth Code Grant flow credentials due to unexpected error: %s",
                         str(e),
                         exc_info=True)

    async def initiate_authorization_flow_server(self) -> None:
        """
        Initiate Auth Code Grant flow to receive access token, and optional refresh token.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager
        from aiq.authentication.exceptions.call_back_exceptions import OAuthClientServerError

        try:
            # Initiate code flow is there is no access token.
            if (self._config.access_token is None):

                # Initiate oauth code flow by sending authorization request.
                await self._send_authorization_request()

                await _CredentialsManager().wait_for_oauth_credentials()

            # Initiate refresh token request if the access token is expired or revoked.
            if (self._config.refresh_token and (self._config.access_token_expires_in is not None)
                    and (datetime.now(timezone.utc) >= self._config.access_token_expires_in)):
                await self._get_access_token_with_refresh_token()

            return

        except AuthCodeGrantFlowError as e:
            error_message = (
                f"Failed to complete Authorization Code Grant Flow for: {self._config_name} Error: {str(e)}")
            logger.error(error_message, exc_info=True)
            raise OAuthClientServerError(error_code='auth_code_grant_flow_error', message=error_message) from e

        except AuthCodeGrantFlowRefreshTokenError as e:
            logger.error("Failed to get Auth Code Grant flow credentials using refresh token: %s",
                         str(e),
                         exc_info=True)

        except Exception as e:
            logger.error("Failed to get Auth Code Grant flow credentials due to unexpected error: %s",
                         str(e),
                         exc_info=True)

    async def _send_authorization_request(self) -> None:
        """
        Constructs Auth Code Grant flow authoriation URL and sends request to authentication server.
        """
        try:
            authorization_url: httpx.URL = httpx.URL(self._config.authorization_url).copy_merge_params(
                self._construct_authorization_query_params().model_dump(exclude_none=True))

            response: httpx.Response | None = await self._request_manager.send_request(url=str(authorization_url),
                                                                                       http_method="GET")
            if response is None:
                error_message = "Unexpected error occurred while sending authorization request - no response received"
                raise AuthCodeGrantFlowError('auth_response_null', error_message)

            if not response.status_code == 200:
                await self._response_manager.handle_auth_code_grant_response_codes(response)

        except Exception as e:
            error_message = f"Unexpected error occurred during authorization request process: {str(e)}"
            logger.error(error_message, exc_info=True)
            raise AuthCodeGrantFlowError('auth_request_failed', error_message) from e

    async def _get_access_token_with_refresh_token(self) -> None:
        """
        Performs the Auth Code Grant token refresh flow by sending a POST request
        to the token endpoint with the required client credentials and refresh token.
        """

        try:
            if not self._config.refresh_token:
                error_message = "Refresh token is missing during the refresh token request"
                raise AuthCodeGrantFlowRefreshTokenError('refresh_token_missing', error_message)

            if not self._config.client_id:
                error_message = "Client ID is missing during the refresh token request"
                raise AuthCodeGrantFlowRefreshTokenError('client_id_missing', error_message)

            if not self._config.client_secret:
                error_message = "Client secret is missing during the refresh token request"
                raise AuthCodeGrantFlowRefreshTokenError('client_secret_missing', error_message)

            body_data: RefreshTokenRequest = RefreshTokenRequest(client_id=self._config.client_id,
                                                                 client_secret=self._config.client_secret,
                                                                 refresh_token=self._config.refresh_token)

            # Send Refresh Token Request
            response: httpx.Response | None = await self._request_manager.send_request(
                url=self._config.authorization_token_url,
                http_method="POST",
                headers={"Content-Type": "application/json"},
                body_data=body_data.model_dump())

            if response is None:
                error_message = "Invalid response received while making refresh token request"
                raise AuthCodeGrantFlowRefreshTokenError('refresh_response_null', error_message)

            if not response.status_code == 200:
                await self._response_manager.handle_auth_code_grant_response_codes(response)

            if response.json().get("access_token") is None:
                error_message = "Access token not in successful token request response payload"
                raise AuthCodeGrantFlowRefreshTokenError('access_token_missing', error_message)

            if response.json().get("expires_in") is None:
                error_message = "Access token expiration time not in successful token request response payload"
                raise AuthCodeGrantFlowRefreshTokenError('token_expiry_missing', error_message)

            if response.json().get("refresh_token") is None:
                error_message = "Refresh token not in successful token request response payload"
                raise AuthCodeGrantFlowRefreshTokenError('refresh_token_response_missing', error_message)

            self._config.access_token = response.json().get("access_token")

            self._config.access_token_expires_in = (datetime.now(timezone.utc) +
                                                    timedelta(seconds=response.json().get("expires_in")))

            self._config.refresh_token = response.json().get("refresh_token")

        except Exception as e:
            error_message = (f"Failed to get Auth Code Grant flow credentials using refresh token "
                             f"for config {self._config_name}: {str(e)}")
            logger.error(error_message, exc_info=True)
            raise AuthCodeGrantFlowRefreshTokenError('refresh_token_flow_failed', error_message) from e

    async def _spawn_oauth_client_server(self) -> None:
        """
        Instantiate _FastApiFrontEndController instance to spin up OAuth2.0 client server
        """
        from aiq.authentication.credentials_manager import _CredentialsManager
        from aiq.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker
        from aiq.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorkerBase

        # Instantiate OAuth2.0 server
        full_config = _CredentialsManager().full_config
        if full_config is None:
            raise RuntimeError("Full configuration is not available for OAuth2 client setup")
        oauth2_client: FastApiFrontEndPluginWorkerBase = FastApiFrontEndPluginWorker(config=full_config)

        # Delegate setup and tear down of server to the controller.
        self._oauth2_client_server = _FastApiFrontEndController(oauth2_client.build_app())

        await self._oauth2_client_server.start_server()

    async def shut_down_code_flow_console(self) -> None:
        """
        Shuts down the Auth Code Grant flow in CONSOLE execution mode if a any unrecoverable errors occur during the
        authentication process.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager

        await _CredentialsManager().set_consent_prompt_url()
        await _CredentialsManager().set_oauth_credentials()
        if self._oauth2_client_server is not None:
            await self._oauth2_client_server.stop_server()

    async def shut_down_code_flow_server(self) -> None:
        """
        Shuts down the Auth Code Grant flow in SERVER execution mode if a any unrecoverable errors occur during the
        authentication process.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager

        await _CredentialsManager().set_consent_prompt_url()
        await _CredentialsManager().set_oauth_credentials()

    async def construct_authentication_header(self,
                                              header_auth_scheme: HeaderAuthScheme = HeaderAuthScheme.BEARER
                                              ) -> httpx.Headers | None:
        """
        Constructs the authenticated HTTP header based on the authentication scheme.

        Args:
            header_auth_scheme (HeaderAuthScheme): The authentication scheme to use. Only BEARER scheme is supported.

        Returns:
            httpx.Headers | None: The constructed HTTP header if successful, otherwise returns None.
        """
        from aiq.authentication.interfaces import AUTHORIZATION_HEADER

        if not await self.validate_credentials():
            logger.error('Credentials invalid. Please authenticate the provider: %s', self._config_name)
            return None

        if header_auth_scheme == HeaderAuthScheme.BEARER:
            return httpx.Headers(
                {f"{AUTHORIZATION_HEADER}": f"{HeaderAuthScheme.BEARER.value} {self._config.access_token}"})

        logger.error('Header authentication scheme not supported for provider: %s', self._config_name)
        return None

    async def send_token_request(self,
                                 client_authorization_path: str,
                                 client_authorization_endpoint: str,
                                 authorization_code: str) -> httpx.Response | None:
        """
        Sends a token request to the authentication server to exchange authorization code for access token.

        Args:
            client_authorization_path (str): The client authorization path for constructing redirect URI
            client_authorization_endpoint (str): The client authorization endpoint for constructing redirect URI
            authorization_code (str): The authorization code received from the OAuth provider

        Returns:
            httpx.Response | None: The HTTP response from the token request, or None if the request fails
        """

        # Build Token request body.
        headers: httpx.Headers = httpx.Headers({"Content-Type": "application/json"})

        redirect_uri: str = self._construct_redirect_uri(client_authorization_path=client_authorization_path,
                                                         client_authorization_endpoint=client_authorization_endpoint)

        token_request_body: OAuth2TokenRequest = self._construct_token_request_body(
            redirect_uri=redirect_uri, authorization_code=authorization_code)

        # Send Token HTTP Request
        return await self.send_request(url=self.config.authorization_token_url,
                                       http_method=HTTPMethod.POST.value,
                                       headers=dict(headers),
                                       body_data=token_request_body.model_dump(exclude_none=True))

    def _construct_authorization_query_params(self,
                                              response_type: str = "code",
                                              prompt: str = "consent") -> OAuth2AuthorizationQueryParams:
        """
        Constructs the OAuth2 authorization query parameters for the authorization URL.

        Args:
            response_type (str): The OAuth2 response type
            prompt (str): The consent prompt behavior

        Returns:
            OAuth2AuthorizationQueryParams: The constructed query parameters for OAuth2 authorization
        """
        from aiq.data_models.authentication import AuthenticationEndpoint
        from aiq.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig

        return OAuth2AuthorizationQueryParams(client_id=self._config.client_id,
                                              audience=self._config.audience,
                                              state=self._config.state,
                                              scope=(" ".join(self._config.scope)),
                                              redirect_uri=(f"{self._config.client_server_url}"
                                                            f"{FastApiFrontEndConfig().authorization.path}"
                                                            f"{AuthenticationEndpoint.REDIRECT_URI.value}"),
                                              response_type=response_type,
                                              prompt=prompt)

    def _construct_token_request_body(self,
                                      redirect_uri: str,
                                      authorization_code: str,
                                      grant_type: str = "authorization_code") -> OAuth2TokenRequest:
        """
        Constructs the OAuth2 token request body for exchanging authorization code for access token.

        Args:
            redirect_uri (str): The redirect URI used in the authorization request
            authorization_code (str): The authorization code received from the OAuth provider
            grant_type (str): The OAuth2 grant type (default: "authorization_code")

        Returns:
            OAuth2TokenRequest: The constructed token request body for OAuth2 token exchange
        """
        return OAuth2TokenRequest(client_id=self._config.client_id,
                                  client_secret=self._config.client_secret,
                                  redirect_uri=redirect_uri,
                                  code=authorization_code,
                                  grant_type=grant_type)

    def _construct_redirect_uri(self, client_authorization_path: str, client_authorization_endpoint: str) -> str:
        """
        Constructs the redirect URI for the OAuth client by combining server URL with authorization paths.

        Args:
            client_authorization_path (str): The base authorization path
            client_authorization_endpoint (str): The specific authorization endpoint

        Returns:
            str: The complete redirect URI for OAuth authorization flow
        """
        return f"{self._config.client_server_url}{client_authorization_path}{client_authorization_endpoint}"

    async def send_request(self,
                           url: str,
                           http_method: str | HTTPMethod,
                           headers: dict | None = None,
                           query_params: dict | None = None,
                           body_data: dict | None = None) -> httpx.Response | None:
        """
        Sends an HTTP request to the API using the configured request manager.

        Args:
            url (str): The URL to send the request to
            http_method (str | HTTPMethod): The HTTP method to use (GET, POST, etc.)
            headers (dict | None): Optional dictionary of HTTP headers
            query_params (dict | None): Optional dictionary of query parameters
            body_data (dict | None): Optional dictionary representing the request body

        Returns:
            httpx.Response | None: The HTTP response from the request, or None if the request fails
        """
        return await self._request_manager.send_request(url,
                                                        http_method,
                                                        headers=headers,
                                                        query_params=query_params,
                                                        body_data=body_data)
