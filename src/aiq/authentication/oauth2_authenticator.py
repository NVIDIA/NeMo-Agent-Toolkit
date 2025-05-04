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

from aiq.data_models.authentication import OAuth2Config
from aiq.front_ends.fastapi.fastapi_front_end_controller import _FastApiFrontEndController


class OAuth2Authenticator:

    def __init__(self) -> None:
        self._oauth2_client_server: _FastApiFrontEndController = None

    async def validate_credentials(self, credentials: OAuth2Config) -> None:
        """
        Validate the credentials for OAuth2 authentication.

        Args:
            credentials (OAuth2Config): The OAuth2 credentials to validate.
        """
        if credentials.access_token is None:
            await self._get_access_token()

    async def _get_access_token(self) -> None:
        """
        Get the access token from the credentials.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager

        if _CredentialsManager().command_name == "console":  # TODO EE: get list of valid commands.

            await self._spawn_oauth_client_server()

            # TODO EE: Init the Oauth2.0 Code Grant Flow
            print("\nSENDING REQUEST TO AUTHENTICATION PROVIDER\n")

            await _CredentialsManager()._get_credentials()

            await self._oauth2_client_server.stop_server()

        if _CredentialsManager().command_name == "fastapi":
            # TODO EE: Init the Oauth2.0 Code Grant Flow
            print("\nSENDING REQUEST TO AUTHENTICATION PROVIDER SERVER\n")

            await _CredentialsManager()._get_credentials()

    async def _spawn_oauth_client_server(self) -> None:
        from aiq.authentication.credentials_manager import _CredentialsManager
        from aiq.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
        from aiq.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker
        from aiq.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorkerBase

        _CredentialsManager().full_config.general.front_end = FastApiFrontEndConfig()
        oauth2_client: FastApiFrontEndPluginWorkerBase = FastApiFrontEndPluginWorker(
            config=_CredentialsManager().full_config)

        self._oauth2_client_server = _FastApiFrontEndController(oauth2_client.build_app())
        await self._oauth2_client_server.start_server()
