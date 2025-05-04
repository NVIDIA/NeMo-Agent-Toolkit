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

from fastapi import FastAPI
from uvicorn import Config
from uvicorn import Server


class _FastApiFrontEndController:

    def __init__(self, app: FastAPI):
        from aiq.authentication.credentials_manager import _CredentialsManager
        self._app: FastAPI = app
        self._config: Config = Config(app=self._app,
                                      host=_CredentialsManager().full_config.general.front_end.host,
                                      port=_CredentialsManager().full_config.general.front_end.port,
                                      workers=_CredentialsManager().full_config.general.front_end.workers)
        self._server: Server = Server(config=self._config)
        self._server_background_task: asyncio.Task = None

    async def start_server(self) -> None:

        from aiq.authentication.credentials_manager import _CredentialsManager
        from aiq.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig

        # Overwrite the front end config to default to spawn fastapi server.
        _CredentialsManager().full_config.general.front_end = FastApiFrontEndConfig()

        print("\nSPAWNING SERVER\n")

        self._server_background_task = asyncio.create_task(self._server.serve())

    async def stop_server(self) -> None:
        print("\nSHUTTING SERVER DOWN\n")

        self._server.should_exit = True

        await self._server_background_task
