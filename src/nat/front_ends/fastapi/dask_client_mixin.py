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

import typing
from abc import ABC
from collections.abc import AsyncGenerator
from collections.abc import Generator
from contextlib import asynccontextmanager
from contextlib import contextmanager

if typing.TYPE_CHECKING:
    from dask.distributed import Client


class DaskClientMixin(ABC):

    @asynccontextmanager
    async def client(self, address: str) -> AsyncGenerator["Client"]:
        """
        Async context manager for obtaining a Dask client.

        Yields
        ------
        Client
            An async Dask client connected to the scheduler. The client is automatically closed when exiting the
            context manager.
        """
        from dask.distributed import Client
        client = await Client(address=address, asynchronous=True)

        try:
            yield client
        finally:
            await client.close()

    @contextmanager
    def blocking_client(self, address: str) -> Generator["Client"]:
        """
        context manager for obtaining a blocking Dask client.

        Yields
        ------
        Client
            A blocking Dask client connected to the scheduler. The client is automatically closed when exiting the
            context manager.
        """
        from dask.distributed import Client
        client = Client(address=address)

        try:
            yield client
        finally:
            client.close()
