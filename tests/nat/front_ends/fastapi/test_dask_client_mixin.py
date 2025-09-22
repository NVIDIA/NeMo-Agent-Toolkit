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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import pytest_asyncio

from nat.front_ends.fastapi.dask_client_mixin import DaskClientMixin


class ConcreteDaskClientMixin(DaskClientMixin):
    """Concrete implementation for testing the abstract DaskClientMixin."""
    pass


@pytest.fixture(name="dask_client_mixin")
def dask_client_mixin_fixture():
    """Fixture providing a concrete instance of DaskClientMixin."""
    return ConcreteDaskClientMixin()


@pytest_asyncio.fixture(name="mock_async_client")
async def mock_async_client_fixture():
    """Fixture providing a mocked async Dask client."""
    mock_client = AsyncMock()
    mock_client.return_value = mock_client
    mock_client.close = AsyncMock()
    return mock_client


@pytest.fixture(name="mock_blocking_client")
def mock_blocking_client_fixture():
    """Fixture providing a mocked blocking Dask client."""
    mock_client = MagicMock()
    mock_client.return_value = mock_client
    mock_client.close = MagicMock()
    return mock_client


@pytest.fixture(name="test_address")
def test_address_fixture():
    """Fixture providing a test Dask client address."""
    return "tcp://127.0.0.1:8786"


async def test_async_client_context_manager(dask_client_mixin: DaskClientMixin,
                                            mock_async_client: AsyncMock,
                                            test_address: str):
    """Test async client context manager creates and closes client properly."""

    with patch('dask.distributed.Client', new=mock_async_client) as mock_client_class:
        async with dask_client_mixin.client(test_address) as client:
            # Verify the client was constructed with correct parameters
            mock_client_class.assert_called_once_with(address=test_address, asynchronous=True)

            # Verify the yielded client is the mocked client
            assert client is mock_async_client

            # At this point, close should not have been called yet
            mock_async_client.close.assert_not_called()

        # After exiting context manager, close should have been called
        mock_async_client.close.assert_called_once()


async def test_async_client_exception_handling(dask_client_mixin: DaskClientMixin,
                                               mock_async_client: AsyncMock,
                                               test_address: str):
    """Test that the async client is closed even when an exception occurs."""

    with patch('dask.distributed.Client', new=mock_async_client):
        with pytest.raises(ValueError, match="Test exception"):
            async with dask_client_mixin.client(test_address):
                raise ValueError("Test exception")

        # Verify close was still called despite the exception
        mock_async_client.close.assert_called_once()


def test_blocking_client_context_manager(dask_client_mixin: DaskClientMixin,
                                         mock_blocking_client: MagicMock,
                                         test_address: str):
    """Test blocking client context manager creates and closes properly."""

    with patch('dask.distributed.Client', new=mock_blocking_client) as mock_client_class:
        with dask_client_mixin.blocking_client(test_address) as client:
            # Verify the client was constructed with correct parameters
            mock_client_class.assert_called_once_with(address=test_address)

            # Verify the yielded client is the mocked client
            assert client is mock_blocking_client

            # At this point, close should not have been called yet
            mock_blocking_client.close.assert_not_called()

        # After exiting context manager, close should have been called
        mock_blocking_client.close.assert_called_once()


def test_blocking_client_exception_handling(dask_client_mixin: DaskClientMixin,
                                            mock_blocking_client: MagicMock,
                                            test_address: str):
    """Test blocking client is closed even when an exception occurs."""

    with patch('dask.distributed.Client', new=mock_blocking_client):
        with pytest.raises(RuntimeError, match="Test exception"):
            with dask_client_mixin.blocking_client(test_address):
                raise RuntimeError("Test exception")

        # Verify close was still called despite the exception
        mock_blocking_client.close.assert_called_once()
