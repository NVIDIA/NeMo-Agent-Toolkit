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

import contextlib
import typing

if typing.TYPE_CHECKING:
    import httpx

    from nat.data_models.llm import LLMBaseConfig


def _create_http_client(llm_config: "LLMBaseConfig",
                        use_async: bool = True,
                        **kwargs) -> "httpx.AsyncClient | httpx.Client":
    """
    Create an httpx client with timeout and verify setting based on LLM configuration parameters.

    Args:
        llm_config: LLM configuration object
        use_async: Whether to create an AsyncClient (True) or a regular Client (False). Defaults to True.

    Returns:
        An httpx.AsyncClient or httpx.Client
    """

    import httpx

    def _set_kwarg(kwarg_name: str, config_attr: str):
        if kwarg_name not in kwargs and getattr(llm_config, config_attr, None) is not None:
            kwargs[kwarg_name] = getattr(llm_config, config_attr)

    _set_kwarg("verify", "verify_ssl")
    _set_kwarg("timeout", "request_timeout")

    if use_async:
        client_class = httpx.AsyncClient
    else:
        client_class = httpx.Client

    return client_class(**kwargs)


@contextlib.contextmanager
def http_client(llm_config: "LLMBaseConfig", **kwargs) -> "httpx.Client":
    """
    Context manager for a synchronous httpx client, to ensure that the client is properly closed after use.
    """
    client = _create_http_client(llm_config=llm_config, use_async=False, **kwargs)
    try:
        yield client
    finally:
        client.close()


@contextlib.asynccontextmanager
async def async_http_client(llm_config: "LLMBaseConfig", **kwargs) -> "httpx.AsyncClient":
    """
    Async context manager for an asynchronous httpx client, to ensure that the client is properly closed after use.
    """
    client = _create_http_client(llm_config=llm_config, use_async=True, **kwargs)
    try:
        yield client
    finally:
        await client.aclose()


class HttpClients(typing.TypedDict):
    """
    Return type for the `http_clients` context manager.
    """
    http_client: "httpx.Client"
    async_http_client: "httpx.AsyncClient"


@contextlib.asynccontextmanager
async def http_clients(llm_config: "LLMBaseConfig", **kwargs) -> HttpClients:
    """
    Get a dictionary of HTTP clients, one sync one async.

    This is a wrapper around `async_http_client` and `http_client`, useful for LLMs that support both sync and async
    clients.
    """
    async with async_http_client(llm_config, **kwargs) as async_client:
        with http_client(llm_config, **kwargs) as sync_client:
            yield HttpClients(http_client=sync_client, async_http_client=async_client)


def _handle_litellm_verify_ssl(llm_config: "LLMBaseConfig") -> None:
    """
    Disable SSL verification for litellm if verify_ssl is set to False in the LLM configuration.

    Currently litellm does not support disabling this on a per-LLM basis for any backend other than Bedrock and AIM
    Guardrail, calling this function will set the global litellm.ssl_verify and impact all subsequent litellm calls.
    """

    import litellm
    litellm.ssl_verify = getattr(llm_config, "verify_ssl", True)
