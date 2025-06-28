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

import httpx

from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantFlowError
from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantFlowRefreshTokenError
from aiq.authentication.interfaces import AuthenticationManagerBase
from aiq.authentication.oauth2.auth_code_grant_config import AuthCodeGrantConfig
from aiq.authentication.request_manager import RequestManager
from aiq.authentication.response_manager import ResponseManager
from aiq.data_models.authentication import ExecutionMode
from aiq.data_models.authentication import RefreshTokenRequest
from aiq.front_ends.fastapi.fastapi_front_end_controller import _FastApiFrontEndController

logger = logging.getLogger(__name__)


class AuthCodeGrantManager(AuthenticationManagerBase):

    def __init__(self, config_name: str, encrypted_config: AuthCodeGrantConfig, execution_mode: ExecutionMode) -> None:

        self._config_name: str = config_name
        self._encrypted_config: AuthCodeGrantConfig = encrypted_config
        self._request_manager: RequestManager = RequestManager()
        self._response_manager: ResponseManager = ResponseManager()
        self._execution_mode: ExecutionMode = execution_mode

        self._initiate_code_flow_dispatch: dict[ExecutionMode, Callable[..., Awaitable[None]]] = {
            ExecutionMode.CONSOLE: self._initiate_authorization_code_flow_console,
            ExecutionMode.SERVER: self._initiate_authorization_code_flow_server
        }
        self._shutdown_code_flow_dispatch: dict[ExecutionMode, Callable[..., Awaitable[None]]] = {
            ExecutionMode.CONSOLE: self._shut_down_code_flow_console,
            ExecutionMode.SERVER: self._shut_down_code_flow_server
        }
        self._oauth2_client_server: _FastApiFrontEndController = None
        super().__init__()

    async def validate_authentication_credentials(self) -> bool:
        """
        Validates the Auth Code Grant grant flow authentication credentials and returns True if the credentials are
        valid and False if they are not. To reliably validate Auth Code Grant flow credentials, a request should be sent
        either to the authorization server's introspection endpoint or to a protected API endpoint, monitoring for a 200
        response. Since introspection endpoints are not standardized the most consistent approach is to check whether
        the access is valid token has not expired.

        Returns:
            bool: True if the credentials are valid and False if they are not.
        """

        if (self._encrypted_config.access_token and (self._encrypted_config.access_token_expires_in is not None)
                and (datetime.now(timezone.utc) <= self._encrypted_config.access_token_expires_in)):
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
            if (self._encrypted_config.access_token is None):
                await self._initiate_code_flow_dispatch[self._execution_mode]()

            # Initiate refresh token request if the access token is expired or revoked.
            if (self._encrypted_config.refresh_token and (self._encrypted_config.access_token_expires_in is not None)
                    and (datetime.now(timezone.utc) >= self._encrypted_config.access_token_expires_in)):
                await self._get_access_token_with_refresh_token()

            return await self.validate_authentication_credentials()

        except AuthCodeGrantFlowError as e:
            logger.error("Failed to get Auth Code Grant flow credentials: %s", str(e), exc_info=True)
            return False

        except AuthCodeGrantFlowRefreshTokenError as e:
            logger.error("Failed to get Auth Code Grant flow credentials using refresh token: %s",
                         str(e),
                         exc_info=True)
            return False

        except Exception as e:
            logger.error("Failed to get Auth Code Grant flow credentials due to unexpected error: %s",
                         str(e),
                         exc_info=True)
            return False

    async def construct_authentication_header(self) -> httpx.Headers | None:
        """
        Constructs the authenticated API key HTTP header.

        Returns:
            httpx.Headers | None: Returns the constructed HTTP header if the API key is valid, otherwise returns None.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager

        if self._encrypted_config.access_token is None:
            logger.error("Access token is not set for config: %s authentication header can not be retreived.",
                         self._config_name)
            return None

        return httpx.Headers(
            {"Authorization": f"Bearer {_CredentialsManager().decrypt_value(self._encrypted_config.access_token)}"})

    async def get_authentication_header(self) -> httpx.Headers | None:
        """
        Gets the authenticated header for the registered authentication config.

        Returns:
            httpx.Headers | None: Returns the authentication header if the config is valid and credentials are
            functional, otherwise returns None.
        """
        construct_authenticated_header: bool = False

        # Ensure authentication credentials are valid and functional.
        credentials_validated: bool = await self.validate_authentication_credentials()

        if credentials_validated:
            construct_authenticated_header = True
        else:
            # If the authentication credentials are invalid, attempt to retrieve and set the new credentials.
            construct_authenticated_header = await self._get_credentials()

        if construct_authenticated_header:
            return await self.construct_authentication_header()
        else:
            logger.error(
                "Auth Code Grant credentials are not valid for config: %s authentication header can not be retreived.",
                self._config_name)
            return None

    async def _initiate_authorization_code_flow_console(self) -> None:
        """
        Initiate Auth Code Grant flow to receive access token, and optional refresh token.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager

        try:
            # Spawn an authentication client server to handle oauth code flow.
            await self._spawn_oauth_client_server()

            # Initiate oauth code flow by sending authorization request.
            await self._send_oauth_authorization_request()

            await _CredentialsManager().wait_for_oauth_credentials()

            await self._oauth2_client_server.stop_server()

        except AuthCodeGrantFlowError as e:
            logger.error("Failed to complete Authorization Code Grant Flow for: %s Error: %s",
                         self._config_name,
                         str(e),
                         exc_info=True)
            await self._shutdown_code_flow_dispatch[self._execution_mode]()
            raise e

    async def _initiate_authorization_code_flow_server(self) -> None:
        """
        Initiate Auth Code Grant flow to receive access token, and optional refresh token.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager

        try:
            # Initiate oauth code flow by sending authorization request.
            await self._send_oauth_authorization_request()

            await _CredentialsManager().wait_for_oauth_credentials()

        except AuthCodeGrantFlowError as e:
            logger.error("Failed to complete Authorization Code Grant Flow for: %s Error: %s",
                         self._config_name,
                         str(e),
                         exc_info=True)
            await self._shutdown_code_flow_dispatch[self._execution_mode]()
            raise e

    async def _send_oauth_authorization_request(self) -> None:
        """
        Constructs Auth Code Grant flow authoriation URL and sends request to authentication server.
        """
        try:
            authorization_url: httpx.URL = await self._request_manager.build_auth_code_grant_url(self._encrypted_config)

            response: httpx.Response | None = await self._request_manager.send_request(url=str(authorization_url),
                                                                                       http_method="GET")
            if response is None:
                error_message = "Unexpected error occurred while sending authorization request - no response received"
                raise AuthCodeGrantFlowError('auth_response_null', error_message)

            if not response.status_code == 200:
                await self._response_manager.handle_auth_code_grant_response_codes(response, self._encrypted_config)

        except Exception as e:
            error_message = f"Unexpected error occurred during authorization request process: {str(e)}"
            logger.error(error_message, exc_info=True)
            raise AuthCodeGrantFlowError('auth_request_failed', error_message) from e

    async def _get_access_token_with_refresh_token(self) -> None:
        """
        Performs the Auth Code Grant token refresh flow by sending a POST request
        to the token endpoint with the required client credentials and refresh token.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager
        try:
            if not self._encrypted_config.refresh_token:
                error_message = "Refresh token is missing during the refresh token request"
                raise AuthCodeGrantFlowRefreshTokenError('refresh_token_missing', error_message)

            if not self._encrypted_config.client_id:
                error_message = "Client ID is missing during the refresh token request"
                raise AuthCodeGrantFlowRefreshTokenError('client_id_missing', error_message)

            if not self._encrypted_config.client_secret:
                error_message = "Client secret is missing during the refresh token request"
                raise AuthCodeGrantFlowRefreshTokenError('client_secret_missing', error_message)

            body_data: RefreshTokenRequest = RefreshTokenRequest(
                client_id=_CredentialsManager().decrypt_value(self._encrypted_config.client_id),
                client_secret=_CredentialsManager().decrypt_value(self._encrypted_config.client_secret),
                refresh_token=_CredentialsManager().decrypt_value(self._encrypted_config.refresh_token))

            # Send Refresh Token Request
            response: httpx.Response | None = await self._request_manager.send_request(
                url=_CredentialsManager().decrypt_value(self._encrypted_config.authorization_token_url),
                http_method="POST",
                authentication_header=None,
                headers={"Content-Type": "application/json"},
                body_data=body_data.model_dump())

            if response is None:
                error_message = "Invalid response received while making refresh token request"
                raise AuthCodeGrantFlowRefreshTokenError('refresh_response_null', error_message)

            if not response.status_code == 200:
                await self._response_manager.handle_auth_code_grant_response_codes(response, self._encrypted_config)

            if response.json().get("access_token") is None:
                error_message = "Access token not in successful token request response payload"
                raise AuthCodeGrantFlowRefreshTokenError('access_token_missing', error_message)

            if response.json().get("expires_in") is None:
                error_message = "Access token expiration time not in successful token request response payload"
                raise AuthCodeGrantFlowRefreshTokenError('token_expiry_missing', error_message)

            if response.json().get("refresh_token") is None:
                error_message = "Refresh token not in successful token request response payload"
                raise AuthCodeGrantFlowRefreshTokenError('refresh_token_response_missing', error_message)

            self._encrypted_config.access_token = _CredentialsManager().encrypt_value(
                response.json().get("access_token"))

            self._encrypted_config.access_token_expires_in = (datetime.now(timezone.utc) +
                                                              timedelta(seconds=response.json().get("expires_in")))

            self._encrypted_config.refresh_token = _CredentialsManager().encrypt_value(
                response.json().get("refresh_token"))

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
        oauth2_client: FastApiFrontEndPluginWorkerBase = FastApiFrontEndPluginWorker(
            config=_CredentialsManager().full_config)

        # Delegate setup and tear down of server to the controller.
        self._oauth2_client_server = _FastApiFrontEndController(oauth2_client.build_app())

        await self._oauth2_client_server.start_server()

    async def _shut_down_code_flow_console(self) -> None:
        """
        Shuts down the Auth Code Grant flow in CONSOLE execution mode if a any unrecoverable errors occur during the
        authentication process.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager

        await _CredentialsManager().set_consent_prompt_url()
        await _CredentialsManager().set_oauth_credentials()
        await self._oauth2_client_server.stop_server()

    async def _shut_down_code_flow_server(self) -> None:
        """
        Shuts down the Auth Code Grant flow in SERVER execution mode if a any unrecoverable errors occur during the
        authentication process.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager

        await _CredentialsManager().set_consent_prompt_url()
        await _CredentialsManager().set_oauth_credentials()
