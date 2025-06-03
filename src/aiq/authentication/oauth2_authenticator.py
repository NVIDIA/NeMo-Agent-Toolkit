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
from collections.abc import Awaitable
from collections.abc import Callable
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import TYPE_CHECKING

import httpx

from aiq.authentication.exceptions import OAuthCodeFlowError
from aiq.authentication.exceptions import OAuthRefreshTokenError
from aiq.authentication.interfaces import AuthenticationBase
from aiq.data_models.authentication import ExecutionMode
from aiq.data_models.authentication import OAuth2Config
from aiq.data_models.authentication import RefreshTokenRequest
from aiq.front_ends.fastapi.fastapi_front_end_controller import _FastApiFrontEndController

if TYPE_CHECKING:
    from aiq.authentication.request_manager import RequestManager
    from aiq.authentication.response_manager import ResponseManager

logger = logging.getLogger(__name__)


class OAuth2Authenticator(AuthenticationBase):

    def __init__(self, request_manager: "RequestManager", response_manager: "ResponseManager") -> None:

        self._request_manager: "RequestManager" = request_manager
        self._response_manager: "ResponseManager" = response_manager
        self._execution_mode: ExecutionMode = None
        self._initiate_code_flow_dispatch: dict[ExecutionMode, Callable[..., Awaitable[None]]] = {
            ExecutionMode.CONSOLE: self._initiate_authorization_code_flow_console,
            ExecutionMode.SERVER: self._initiate_authorization_code_flow_server
        }
        self._shutdown_code_flow_dispatch: dict[ExecutionMode, Callable[..., Awaitable[None]]] = {
            ExecutionMode.CONSOLE: self._shut_down_code_flow_console,
            ExecutionMode.SERVER: self._shut_down_code_flow_server
        }
        self._authentication_provider: OAuth2Config = None
        self._oauth2_client_server: _FastApiFrontEndController = None
        super().__init__()

    @property
    def authentication_provider(self) -> OAuth2Config:
        return self._authentication_provider

    @authentication_provider.setter
    def authentication_provider(self, authentication_provider: OAuth2Config) -> None:
        self._authentication_provider = authentication_provider

    @property
    def execution_mode(self) -> ExecutionMode:
        return self._execution_mode

    @execution_mode.setter
    def execution_mode(self, execution_mode: ExecutionMode) -> None:
        self._execution_mode = execution_mode

    async def _validate_credentials(self) -> bool:
        """
        Validates the OAuth2.0 authentication credentials and returns True if the credentials are valid and False if
        they are not. To reliably validate OAuth2.0 credentials, a request should be sent either to the authorization
        server's introspection endpoint or to a protected API endpoint, monitoring for a 200 response. Since
        introspection endpoints are not standardized the most consistent approach is to check whether the access is
        valid token has not expired.

        Returns:
            bool: True if the credentials are valid and False if they are not.
        """
        if (self._authentication_provider.access_token
                and (self._authentication_provider.access_token_expires_in is not None)
                and (datetime.now(timezone.utc) <= self._authentication_provider.access_token_expires_in)):
            return True
        else:
            return False

    async def _get_credentials(self) -> bool:
        """
        Acquires an access token if the token is absent, expired, or revoked,
        by the options listed below.

        1. Initiate the authorization flow to obtain a new access token and optional request token.
        2. Use a refresh token to get another access token and refresh token pair if refresh token is available.

        Returns:
            bool: True if the credentials are valid and false if they are not after acquiring credentials.
        """
        try:
            # Initiate code flow is there is no access token.
            if (self._authentication_provider.access_token is None):
                await self._initiate_code_flow_dispatch[self.execution_mode]()

            # Initiate refresh token request if the access token is expired or revoked.
            if (self._authentication_provider.refresh_token
                    and (self._authentication_provider.access_token_expires_in is not None)
                    and (datetime.now(timezone.utc) >= self._authentication_provider.access_token_expires_in)):
                await self._get_access_token_with_refresh_token()

            return await self._validate_credentials()

        except OAuthCodeFlowError as e:
            logger.error("Failed to get OAuth2.0 credentials using code flow: %s", str(e), exc_info=True)
            return False

        except OAuthRefreshTokenError as e:
            logger.error("Failed to get OAuth2.0 credentials using refresh token: %s", str(e), exc_info=True)
            return False

        except Exception as e:
            logger.error("Failed to get OAuth2.0 credentials due to unexpected error: %s", str(e), exc_info=True)
            return False

    async def _initiate_authorization_code_flow_console(self) -> None:
        """
        Initiate OAuth2.0 authorization code flow to receive access token, and optional refresh token.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager

        try:
            # Spawn an authentication client server to handle oauth code flow.
            await self._spawn_oauth_client_server()

            # Initiate oauth code flow by sending authorization request.
            await self._send_oauth_authorization_request()

            await _CredentialsManager()._wait_for_oauth_credentials()

            await self._oauth2_client_server.stop_server()

        except OAuthCodeFlowError as e:
            logger.error("Failed to complete OAuth2.0 authentication for provider Error: %s", str(e), exc_info=True)
            await self._shutdown_code_flow_dispatch[self.execution_mode]()
            raise e

    async def _initiate_authorization_code_flow_server(self) -> None:
        """
        Initiate OAuth2.0 authorization code flow to receive access token, and optional refresh token.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager

        try:
            # Initiate oauth code flow by sending authorization request.
            await self._send_oauth_authorization_request()

            await _CredentialsManager()._wait_for_oauth_credentials()

        except OAuthCodeFlowError as e:
            logger.error("Failed to complete OAuth2.0 authentication for provider Error: %s", str(e), exc_info=True)
            await self._shutdown_code_flow_dispatch[self.execution_mode]()
            raise e

    async def _send_oauth_authorization_request(self) -> None:
        """
        Constructs OAuth2.0 Code Flow Authoriation URL and sends request to authentication server.
        """
        try:
            authorization_url: httpx.URL = await self._request_manager._build_oauth_authorization_url(
                self._authentication_provider)

            response: httpx.Response | None = await self._request_manager._send_request(url=str(authorization_url),
                                                                                        http_method="GET")
            if response is None:
                raise OAuthCodeFlowError("Unexpected error occured while sending authorization request.")

            if not response.status_code == 200:
                await self._response_manager._handle_oauth_authorization_response_codes(
                    response, self._authentication_provider)

        except Exception as e:
            logger.error("Unexpected error occured during authorization request process: %s", str(e), exc_info=True)
            raise OAuthCodeFlowError("Unexpected error occured during authorization request process:") from e

    async def _get_access_token_with_refresh_token(self) -> None:
        """
        Performs the OAuth2.0 token refresh flow by sending a POST request
        to the token endpoint with the required client credentials and refresh token.
        """
        try:
            if not self.authentication_provider.refresh_token:
                raise OAuthRefreshTokenError("Refresh token is missing during the refresh token request.")

            if not self.authentication_provider.client_id:
                raise OAuthRefreshTokenError("Client ID is missing during the refresh token request.")

            if not self.authentication_provider.client_secret:
                raise OAuthRefreshTokenError("Client secret is missing during the refresh token request.")

            body_data: RefreshTokenRequest = RefreshTokenRequest(
                client_id=self._authentication_provider.client_id,
                client_secret=self._authentication_provider.client_secret,
                refresh_token=self._authentication_provider.refresh_token)

            # Send Refresh Token Request
            response: httpx.Response | None = await self._request_manager._send_request(
                url=self._authentication_provider.authorization_token_url,
                http_method="POST",
                authentication_provider=None,
                headers={"Content-Type": "application/json"},
                body_data=body_data.model_dump())

            if response is None:
                raise OAuthRefreshTokenError("Invalid response received while making refresh token request.")

            if not response.status_code == 200:
                await self._response_manager._handle_oauth_authorization_response_codes(
                    response, self._authentication_provider)

            if response.json().get("access_token") is None:
                raise OAuthRefreshTokenError("Access token not in successful token request response payload.")

            if response.json().get("expires_in") is None:
                raise OAuthRefreshTokenError(
                    "Access token expiration time not in successful token request response payload.")

            if response.json().get("refresh_token") is None:
                raise OAuthRefreshTokenError("Refresh token not in successful token request response payload.")

            self.authentication_provider.access_token = response.json().get("access_token")

            self.authentication_provider.access_token_expires_in = (
                datetime.now(timezone.utc) + timedelta(seconds=response.json().get("expires_in")))

            self.authentication_provider.refresh_token = response.json().get("refresh_token")

        except Exception as e:
            logger.error("Failed to complete OAuth2.0 authentication for provider Error: %s", str(e), exc_info=True)
            raise OAuthRefreshTokenError("Failed to get OAuth2.0 credentials using refresh token.") from e

    async def _spawn_oauth_client_server(self) -> None:
        """
        Instantiate _FastApiFrontEndController instance to spin up OAuth2.0 client server
        """
        from aiq.authentication.credentials_manager import _CredentialsManager
        from aiq.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker
        from aiq.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorkerBase

        # Instantiate OAuth2.0 server
        oauth2_client: FastApiFrontEndPluginWorkerBase = FastApiFrontEndPluginWorker(
            config=_CredentialsManager().full_config)

        # Delegate setup and tear down of server to the controller.
        self._oauth2_client_server = _FastApiFrontEndController(oauth2_client.build_app())

        await self._oauth2_client_server.start_server()

    async def _shut_down_code_flow_console(self) -> None:
        """
        Shuts down the Oauth2.0 code flow in CONSOLE execution mode if a any unrecoverable errors occur during the
        authentication process.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager

        await _CredentialsManager()._set_consent_prompt_url()
        await _CredentialsManager()._set_oauth_credentials()
        await self._oauth2_client_server.stop_server()

    async def _shut_down_code_flow_server(self) -> None:
        """
        Shuts down the Oauth2.0 code flow in SERVER execution mode if a any unrecoverable errors occur during the
        authentication process.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager

        await _CredentialsManager()._set_consent_prompt_url()
        await _CredentialsManager()._set_oauth_credentials()
