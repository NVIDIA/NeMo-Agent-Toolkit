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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

from nat.llm.utils.constants import LLMHeaderPrefix

logger = logging.getLogger(__name__)


def create_metadata_injection_client(timeout: float = 600.0) -> "httpx.AsyncClient":
    """
    Httpx event hook that injects custom metadata as HTTP headers.

    This client injects custom payload fields as X-Payload-* HTTP headers,
    enabling end-to-end traceability in LLM server logs.

    Args:
        timeout: HTTP request timeout in seconds

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
                        header_name: str = f"{LLMHeaderPrefix.PAYLOAD.value}-{key.replace('_', '-')}"
                        request.headers[header_name] = str(value)
                        logger.debug("Injected custom metadata header: %s=%s", header_name, value)
        except Exception as e:
            logger.debug("Could not inject custom metadata headers, request will proceed without them: %s", e)

    return httpx.AsyncClient(
        event_hooks={"request": [on_request]},
        timeout=httpx.Timeout(timeout),
    )
