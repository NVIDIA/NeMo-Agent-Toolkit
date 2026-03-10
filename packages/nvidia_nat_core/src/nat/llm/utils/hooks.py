# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
HTTP event hooks for LLM clients.

This module provides httpx event hooks that inject custom metadata from
input payloads as HTTP headers to LLM requests, enabling end-to-end
traceability in LLM server logs.
"""

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

    from nat.data_models.llm import LLMBaseConfig

from nat.llm.utils.constants import LLMHeaderPrefix
from nat.llm.utils.http_client import async_http_client

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _create_metadata_injection_client(llm_config: "LLMBaseConfig") -> "httpx.AsyncClient":
    """
    Httpx event hook that injects custom metadata as HTTP headers.

    This client injects custom payload fields as X-Payload-* HTTP headers,
    enabling end-to-end traceability in LLM server logs.

    Args:
        llm_config: LLM configuration object

    Returns:
        An httpx.AsyncClient configured with metadata header injection
    """
    import httpx

    from nat.builder.context import ContextState

    async def on_request(request: httpx.Request) -> None:
        """Inject custom metadata headers from input payload before each LLM request."""
        try:
            context_state: ContextState = ContextState.get()
            input_message = context_state.input_message.get()

            if input_message and hasattr(input_message, 'model_extra') and input_message.model_extra:
                for key, value in input_message.model_extra.items():
                    if value is not None:
                        header_name: str = f"{LLMHeaderPrefix.PAYLOAD}-{key.replace('_', '-')}"
                        request.headers[header_name] = str(value)
                        logger.debug("Injected custom metadata header: %s=%s", header_name, value)
        except Exception as e:
            logger.debug("Could not inject custom metadata headers, request will proceed without them: %s", e)

    async with async_http_client(llm_config=llm_config, event_hooks={"request": [on_request]}) as client:
        yield client
