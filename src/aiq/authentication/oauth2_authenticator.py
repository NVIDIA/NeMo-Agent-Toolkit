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
from typing import TYPE_CHECKING

from aiq.front_ends.fastapi.fastapi_front_end_controller import _FastApiFrontEndController

if TYPE_CHECKING:
    from aiq.authentication.request_manager import RequestManager

logger = logging.getLogger(__name__)


class OAuthError(Exception):
    """Raised when OAuth flow fails unexpectedly."""
    pass


class OAuth2Authenticator:

    def __init__(self, request_manager: "RequestManager") -> None:
        self._request_manager: "RequestManager" = request_manager
        self._authentication_provider: str = None
        self._oauth2_client_server: _FastApiFrontEndController = None

    @property
    def request_manager(self) -> "RequestManager":
        return self._request_manager

    @property
    def authentication_provider(self) -> str:
        return self._authentication_provider

    @authentication_provider.setter
    def authentication_provider(self, authentication_provider: str) -> None:
        self._authentication_provider = authentication_provider

    async def validate_credentials(self,
                                   authentication_provider: str) -> None:  # TODO EE: Update case for valid access token
        """
        Validate the credentials for OAuth2 authentication.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager

        self._authentication_provider = authentication_provider

        if _CredentialsManager()._get_authentication_provider(self.authentication_provider).access_token is None:
            await self._get_access_token()

    async def _get_access_token(self) -> None:
        """
        Get the access token from the credentials.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager
        try:

            if _CredentialsManager().command_name == "console":  # TODO EE: get list of valid commands.
                # Spawn a authentication client server to handle oauth code flow.
                await self._spawn_oauth_client_server()

                # Initiate oauth code flow.
                await self._send_authorization_request()

                await _CredentialsManager()._wait_for_oauth_credentials(
                )  # TODO EE: Check for errors and unblock / shutdown

                await self._oauth2_client_server.stop_server()

            if _CredentialsManager().command_name == "fastapi":

                # Initiate oauth code flow.
                await self._send_authorization_request()

                await _CredentialsManager()._wait_for_oauth_credentials()

        except OAuthError as e:
            logger.error("Failed to complete OAuth2.0 authentication for provider:  %s,  Error: %s",
                         self.authentication_provider,
                         str(e),
                         exc_info=True)
            await self._shut_down_code_flow()

    async def _spawn_oauth_client_server(self) -> None:
        from aiq.authentication.credentials_manager import _CredentialsManager
        from aiq.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
        from aiq.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker
        from aiq.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorkerBase

        # TODO EE: Add comments
        _CredentialsManager().full_config.general.front_end = FastApiFrontEndConfig()
        oauth2_client: FastApiFrontEndPluginWorkerBase = FastApiFrontEndPluginWorker(
            config=_CredentialsManager().full_config)

        oauth2_client.request_manager = self._request_manager

        self._oauth2_client_server = _FastApiFrontEndController(oauth2_client.build_app())
        await self._oauth2_client_server.start_server()

    async def _send_authorization_request(self):
        # TODO EE: Update logic to pass authentication provider to functions instead of string.
        await self.request_manager.send_authorization_request(self.authentication_provider)

    async def initiate_code_flow(self):
        pass

    async def _shut_down_code_flow(self):
        from aiq.authentication.credentials_manager import _CredentialsManager
        await _CredentialsManager()._set_consent_prompt()
        await _CredentialsManager()._set_oauth_credentials()
        await self._oauth2_client_server.stop_server()
