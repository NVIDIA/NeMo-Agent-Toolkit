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

import socket
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from aiq_alert_triage_agent.network_connectivity_check_tool import _check_service_banner


@pytest.fixture
def mock_sock():
    """A reusable mock socket whose recv and settimeout we can configure."""
    sock = MagicMock()
    return sock


@patch('socket.create_connection')
def test_successful_banner_read(mock_create_conn, mock_sock):
    # Simulate a two‐chunk banner (one before the pattern, the pattern itself) then EOF
    mock_sock.recv.side_effect = [
        b"Welcome to test server\n",
        b"Escape character is '^]'.\n",
        b""  # EOF
    ]
    mock_create_conn.return_value.__enter__.return_value = mock_sock

    result = _check_service_banner("my.host", port=8080)
    assert "Welcome to test server" in result
    assert "Escape character is '^]'." in result

    mock_create_conn.assert_called_once_with(("my.host", 8080), timeout=10)
    mock_sock.settimeout.assert_called_once_with(10)


@pytest.mark.parametrize(
    "side_effect, port, conn_to, read_to",
    [
        (socket.timeout(), 80, 10, 10),
        (ConnectionRefusedError(), 80, 10, 10),
        (OSError(), 1234, 5, 2),
    ],
)
@patch('socket.create_connection')
def test_error_conditions(mock_create_conn, side_effect, port, conn_to, read_to):
    """
    If create_connection raises timeout/conn refused/OS error,
    _check_service_banner should return empty string and
    propagate the connection parameters correctly.
    """
    mock_create_conn.side_effect = side_effect

    result = _check_service_banner("any.host", port=port, connect_timeout=conn_to, read_timeout=read_to)
    assert result == ""
    mock_create_conn.assert_called_once_with(("any.host", port), timeout=conn_to)


@patch('socket.create_connection')
def test_reading_until_eof_without_banner(mock_create_conn, mock_sock):
    """
    If the server never emits the banner and closes the connection,
    we should still return whatever was read before EOF (even empty).
    """
    # Single empty chunk simulates immediate EOF
    mock_sock.recv.side_effect = [b""]
    mock_create_conn.return_value.__enter__.return_value = mock_sock

    result = _check_service_banner("no.banner.host")
    assert result == ""  # nothing was ever received

    mock_create_conn.assert_called_once_with(("no.banner.host", 80), timeout=10)
    mock_sock.settimeout.assert_called_once_with(10)
