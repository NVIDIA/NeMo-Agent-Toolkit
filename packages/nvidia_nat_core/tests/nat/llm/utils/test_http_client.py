# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for the HTTP client."""

import sys
import typing
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.llm.utils.http_client import _create_http_client
from nat.llm.utils.http_client import _handle_litellm_verify_ssl
from nat.llm.utils.http_client import http_clients

from ._llm_configs import LLMConfig
from ._llm_configs import LLMConfigWithTimeout
from ._llm_configs import LLMConfigWithTimeoutAndSSL

if typing.TYPE_CHECKING:
    from nat.data_models.llm import LLMBaseConfig


@pytest.mark.parametrize("use_async", [True, False], ids=["async", "sync"])
@pytest.mark.parametrize(
    "llm_config,expected_timeout",
    [
        (LLMConfig(), None),
        (LLMConfigWithTimeout(), None),
        (LLMConfigWithTimeout(request_timeout=45.0), 45.0),
    ],
    ids=["no_request_timeout_attr", "request_timeout_none", "request_timeout_float"],
)
def test_create_http_client_timeout(
    llm_config: "LLMBaseConfig",
    expected_timeout: float | None,
    use_async: bool,
    mock_httpx_async_client,
    mock_httpx_sync_client,
):
    """Client receives timeout from config when request_timeout is set."""
    if use_async:
        mock_client = mock_httpx_async_client
    else:
        mock_client = mock_httpx_sync_client
    _create_http_client(llm_config=llm_config, use_async=use_async)
    mock_client.assert_called_once()
    call_kwargs = mock_client.call_args.kwargs
    if expected_timeout is None:
        assert "timeout" not in call_kwargs
    else:
        assert call_kwargs["timeout"] == expected_timeout


@pytest.mark.parametrize("use_async", [True, False], ids=["async", "sync"])
@pytest.mark.parametrize(
    "llm_config,expected_verify",
    [
        (LLMConfig(), None),
        (LLMConfigWithTimeoutAndSSL(verify_ssl=True), True),
        (LLMConfigWithTimeoutAndSSL(verify_ssl=False), False),
    ],
    ids=["no_verify_ssl_attr", "verify_ssl_true", "verify_ssl_false"],
)
def test_create_http_client_verify_ssl(
    llm_config: "LLMBaseConfig",
    expected_verify: bool | None,
    use_async: bool,
    mock_httpx_async_client,
    mock_httpx_sync_client,
):
    """Client receives verify from config when verify_ssl is set."""
    if use_async:
        mock_client = mock_httpx_async_client
    else:
        mock_client = mock_httpx_sync_client
    _create_http_client(llm_config=llm_config, use_async=use_async)
    mock_client.assert_called_once()
    call_kwargs = mock_client.call_args.kwargs
    if expected_verify is None:
        assert "verify" not in call_kwargs
    else:
        assert call_kwargs["verify"] is expected_verify


@pytest.mark.parametrize(
    "llm_config,expected_verify",
    [
        (LLMConfig(), None),
        (LLMConfigWithTimeoutAndSSL(verify_ssl=True), True),
        (LLMConfigWithTimeoutAndSSL(verify_ssl=False), False),
    ],
    ids=["no_verify_ssl_attr", "verify_ssl_true", "verify_ssl_false"],
)
async def test_http_clients(
    mock_httpx_async_client,
    mock_httpx_sync_client,
    llm_config: "LLMBaseConfig",
    expected_verify: bool | None,
):
    """http_clients yields both sync and async clients and passes verify_ssl when set."""
    async with http_clients(llm_config) as result:
        assert set(result.keys()) == {"http_client", "async_http_client"}
        mock_httpx_sync_client.assert_called_once()
        mock_httpx_async_client.assert_called_once()
        assert result["http_client"] is mock_httpx_sync_client.return_value
        assert result["async_http_client"] is mock_httpx_async_client.return_value
        if expected_verify is None:
            assert "verify" not in mock_httpx_sync_client.call_args.kwargs
            assert "verify" not in mock_httpx_async_client.call_args.kwargs
        else:
            assert mock_httpx_sync_client.call_args.kwargs["verify"] is expected_verify
            assert mock_httpx_async_client.call_args.kwargs["verify"] is expected_verify


@pytest.mark.parametrize(
    "llm_config,expected_value",
    [
        (LLMConfig(), True),
        (LLMConfigWithTimeoutAndSSL(verify_ssl=True), True),
        (LLMConfigWithTimeoutAndSSL(verify_ssl=False), False),
    ],
    ids=["no_verify_ssl_attr", "verify_ssl_true", "verify_ssl_false"],
)
def test_handle_litellm_verify_ssl(llm_config: "LLMBaseConfig", expected_value: bool):
    """litellm.ssl_verify is set from config verify_ssl."""
    mock_litellm = MagicMock()
    with patch.dict(sys.modules, {"litellm": mock_litellm}):
        _handle_litellm_verify_ssl(llm_config)
    assert mock_litellm.ssl_verify == expected_value
