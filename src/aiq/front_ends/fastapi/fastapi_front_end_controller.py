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

import asyncio
import logging

from fastapi import FastAPI
from uvicorn import Config
from uvicorn import Server

logger = logging.getLogger(__name__)


class _FastApiFrontEndController:
    """
    _FastApiFrontEndController class controls the spawing and tear down of the API server in environments where
    the server is needed and not already running.
    """

    def __init__(self, app: FastAPI):
        from aiq.authentication.credentials_manager import _CredentialsManager
        self._app: FastAPI = app
        self._config: Config = Config(app=self._app,
                                      host=_CredentialsManager().full_config.general.front_end.host,
                                      port=_CredentialsManager().full_config.general.front_end.port,
                                      workers=_CredentialsManager().full_config.general.front_end.workers,
                                      log_level="warning")
        self._server: Server = Server(config=self._config)
        self._server_background_task: asyncio.Task = None

    async def start_server(self) -> None:
        "Starts the API server."
        from aiq.authentication.oauth2_authenticator import OAuthCodeFlowError
        try:
            self._server_background_task = asyncio.create_task(self._server.serve())

        except asyncio.CancelledError as e:
            logger.error("Task error occured while starting OAuth2.0 server: %s", str(e), exc_info=True)
            raise OAuthCodeFlowError("Task error occured while starting OAuth2.0 server:") from e

        except Exception as e:
            logger.error("Unexpected error occured while starting OAuth2.0 server: %s", str(e), exc_info=True)
            raise OAuthCodeFlowError("Unexpected error occured while starting OAuth2.0 server:") from e

    async def stop_server(self) -> None:
        "Stops the API server."
        try:
            # Shut the server instance down.
            self._server.should_exit = True

            # Wait for the background task to clean up.
            await self._server_background_task

        except asyncio.CancelledError as e:
            logger.error("Server shutdown failed: %s", str(e), exc_info=True)
        except Exception as e:
            logger.error("Unexpected error occured: %s", str(e), exc_info=True)
