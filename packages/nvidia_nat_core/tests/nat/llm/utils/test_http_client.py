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
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic import Field

from nat.data_models.llm import LLMBaseConfig
from nat.data_models.llm import SSLVerificationMixin
from nat.llm.utils.http_client import _create_http_client
from nat.llm.utils.http_client import _handle_litellm_verify_ssl


class LLMConfig(LLMBaseConfig):
    pass


class LLMConfigWithTimeout(LLMBaseConfig):
    request_timeout: float | None = Field(default=None, gt=0.0, description="HTTP request timeout in seconds.")


class LLMConfigWithSSL(LLMConfigWithTimeout, SSLVerificationMixin):
    pass


@pytest.fixture(name="mock_httpx_async_client")
def fixture_mock_async_httpx_client():
    import httpx

    with patch.object(httpx, "AsyncClient") as mock_httpx:
        mock_httpx.return_value = mock_httpx
        yield mock_httpx


@pytest.fixture(name="mock_httpx_sync_client")
def fixture_mock_sync_httpx_client():
    import httpx

    with patch.object(httpx, "Client") as mock_httpx:
        mock_httpx.return_value = mock_httpx
        yield mock_httpx


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
    llm_config: LLMBaseConfig,
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
        (LLMConfigWithSSL(verify_ssl=True), True),
        (LLMConfigWithSSL(verify_ssl=False), False),
    ],
    ids=["no_verify_ssl_attr", "verify_ssl_true", "verify_ssl_false"],
)
def test_create_http_client_verify_ssl(
    llm_config: LLMBaseConfig,
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
    "llm_config,expected_value",
    [
        (LLMConfig(), True),
        (LLMConfigWithSSL(verify_ssl=True), True),
        (LLMConfigWithSSL(verify_ssl=False), False),
    ],
    ids=["no_verify_ssl_attr", "verify_ssl_true", "verify_ssl_false"],
)
def test_handle_litellm_verify_ssl(llm_config: LLMBaseConfig, expected_value: bool):
    """litellm.ssl_verify is set from config verify_ssl."""
    mock_litellm = MagicMock()
    with patch.dict(sys.modules, {"litellm": mock_litellm}):
        _handle_litellm_verify_ssl(llm_config)
    assert mock_litellm.ssl_verify == expected_value
