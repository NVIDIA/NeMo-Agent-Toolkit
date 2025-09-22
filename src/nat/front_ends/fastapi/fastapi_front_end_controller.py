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
        self._app: FastAPI = app
        self._server: Server | None = None
        self._server_background_task: asyncio.Task | None = None

    async def start_server(self, host: str, port: int) -> None:
        """Starts the API server."""

        server_host = host
        server_port = port

        config = Config(app=self._app, host=server_host, port=server_port, log_level="warning")
        self._server = Server(config=config)

        try:
            self._server_background_task = asyncio.create_task(self._server.serve())
        except asyncio.CancelledError as e:
            error_message = f"Task error occurred while starting API server: {str(e)}"
            logger.error(error_message)
            raise RuntimeError(error_message) from e
        except Exception as e:
            error_message = f"Unexpected error occurred while starting API server: {str(e)}"
            logger.exception(error_message)
            raise RuntimeError(error_message) from e

    async def stop_server(self) -> None:
        """Stops the API server."""
        if not self._server or not self._server_background_task:
            return

        try:
            self._server.should_exit = True
            await self._server_background_task
        except asyncio.CancelledError as e:
            logger.exception("Server shutdown failed: %s", str(e))
        except Exception as e:
            logger.exception("Unexpected error occurred: %s", str(e))
