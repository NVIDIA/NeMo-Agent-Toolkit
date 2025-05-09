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

from aiq.authentication.exceptions import OAuthError
from aiq.data_models.authentication import OAuth2Config
from aiq.data_models.authentication import RunMode
from aiq.front_ends.fastapi.fastapi_front_end_controller import _FastApiFrontEndController

if TYPE_CHECKING:
    from aiq.authentication.request_manager import RequestManager

logger = logging.getLogger(__name__)


class OAuth2Authenticator:

    def __init__(self, request_manager: "RequestManager") -> None:
        self._request_manager: "RequestManager" = request_manager
        self._authentication_provider: OAuth2Config = None
        self._oauth2_client_server: _FastApiFrontEndController = None

    @property
    def request_manager(self) -> "RequestManager":
        return self._request_manager

    @property
    def authentication_provider(self) -> OAuth2Config:
        return self._authentication_provider

    @authentication_provider.setter
    def authentication_provider(self, authentication_provider: OAuth2Config) -> None:
        self._authentication_provider = authentication_provider

    async def _validate_credentials(self) -> bool:
        """
        Validate the credentials for OAuth2 authentication.
        """
        # TODO EE: Need to handle refresh token etc....
        if self._authentication_provider.access_token is None:
            await self._initiate_code_flow()

        if self._authentication_provider.access_token:
            return True
        else:
            return False

    async def _initiate_code_flow(self) -> None:
        """
        Initiate OAuth2.0 code flow to get access token.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager

        try:
            if _CredentialsManager().command_name == RunMode.CONSOLE.value:
                # Spawn a authentication client server to handle oauth code flow.
                await self._spawn_oauth_client_server()

                # Initiate oauth code flow by sending authorization request.
                await self.request_manager.send_oauth_authorization_request(self.authentication_provider)

                await _CredentialsManager()._wait_for_oauth_credentials()

                await self._oauth2_client_server.stop_server()

            if _CredentialsManager().command_name == RunMode.SERVER.value:

                # Initiate oauth code flow.
                await self.request_manager.send_oauth_authorization_request(self.authentication_provider)

                await _CredentialsManager()._wait_for_oauth_credentials()

        except OAuthError as e:
            logger.error("Failed to complete OAuth2.0 authentication for provider Error: %s", str(e), exc_info=True)
            await self._shut_down_code_flow()

    async def _spawn_oauth_client_server(self) -> None:
        """
        Instantiate _FastApiFrontEndController instance to spin up OAuth2.0 client server
        """
        from aiq.authentication.credentials_manager import _CredentialsManager
        from aiq.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
        from aiq.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker
        from aiq.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorkerBase

        # Overwrite the front end config to default to spawn fastapi server.
        _CredentialsManager().full_config.general.front_end = FastApiFrontEndConfig()

        # Instantiate OAuth2.0 server
        oauth2_client: FastApiFrontEndPluginWorkerBase = FastApiFrontEndPluginWorker(
            config=_CredentialsManager().full_config)

        # Pass request manager to OAuth2.0 server to manage request and responses.
        oauth2_client.request_manager = self._request_manager

        # Delegate setup and tear down of server to the controller.
        self._oauth2_client_server = _FastApiFrontEndController(oauth2_client.build_app())

        await self._oauth2_client_server.start_server()

    async def _shut_down_code_flow(self) -> None:
        """
        Shuts down the oauth server and cancels the Oauth2.0 if a any unrecoverable erros occur during authentication
        """
        from aiq.authentication.credentials_manager import _CredentialsManager

        await _CredentialsManager()._set_consent_prompt()
        await _CredentialsManager()._set_oauth_credentials()
        await self._oauth2_client_server.stop_server()
