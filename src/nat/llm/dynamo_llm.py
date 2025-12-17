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
"""
Dynamo LLM provider with automatic prefix header injection for KV cache optimization.

This module provides a specialized OpenAI-compatible LLM that sends Dynamo prefix headers
for optimal KV cache management and request routing. The prefix parameters are optimizable
via the NAT optimizer.

The implementation uses httpx event hooks to inject headers at the HTTP transport level,
making it framework-agnostic (works with LangChain, LlamaIndex, etc.).

Dynamo Prefix Parameters
-------------------------

prefix_osl (Output Sequence Length)
    Hint for expected response length:

    - LOW: decode_cost=1.0, short responses
    - MEDIUM: decode_cost=2.0, typical responses
    - HIGH: decode_cost=3.0, long responses

prefix_iat (Inter-Arrival Time)
    Hint for request pacing:

    - LOW: iat_factor=1.5, rapid bursts -> high worker stickiness
    - MEDIUM: iat_factor=1.0, normal pacing
    - HIGH: iat_factor=0.6, slow requests -> more exploration

prefix_total_requests
    Expected requests per conversation:

    - Higher values increase KV cache affinity and worker stickiness
    - Lower values allow more load balancing
"""

import logging
import uuid
from collections.abc import Callable
from collections.abc import Coroutine
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

if TYPE_CHECKING:
    import httpx

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.llm import LLMProviderInfo
from nat.cli.register_workflow import register_llm_provider
from nat.data_models.optimizable import OptimizableField
from nat.data_models.optimizable import SearchSpace
from nat.llm.openai_llm import OpenAIModelConfig

logger = logging.getLogger(__name__)

# Define valid prefix hint values
PrefixLevel = Literal["LOW", "MEDIUM", "HIGH"]

# =============================================================================
# CONTEXT MANAGEMENT FOR DYNAMO PREFIX ID
# =============================================================================


class DynamoPrefixContext:
    """
    Singleton class for managing Dynamo prefix IDs across LLM calls.

    This allows evaluation code to set a prefix ID that persists across all LLM
    calls for a single evaluation question (multi-turn conversation).

    Usage::

        from nat.llm.dynamo_llm import DynamoPrefixContext

        # Set prefix ID at the start of each evaluation question
        DynamoPrefixContext.set("eval-q001-abc123")

        # ... perform LLM calls ...

        # Clear when done
        DynamoPrefixContext.clear()

        # Or use as a context manager
        with DynamoPrefixContext.scope("eval-q001-abc123"):
            # ... perform LLM calls ...
    """

    _current_prefix_id: ContextVar[str | None] = ContextVar('dynamo_prefix_id', default=None)

    @classmethod
    def set(cls, prefix_id: str) -> None:
        """
        Set the Dynamo prefix ID for the current context.

        Call this at the start of each evaluation question to ensure all LLM calls
        for that question share the same prefix ID (enabling KV cache reuse).

        Args:
            prefix_id: The unique prefix ID (e.g., "eval-q001-abc123")
        """
        cls._current_prefix_id.set(prefix_id)
        logger.debug("Set Dynamo prefix ID: %s", prefix_id)

    @classmethod
    def clear(cls) -> None:
        """Clear the current Dynamo prefix ID context."""
        cls._current_prefix_id.set(None)
        logger.debug("Cleared Dynamo prefix ID")

    @classmethod
    def get(cls) -> str | None:
        """Get the current Dynamo prefix ID from context, if any."""
        return cls._current_prefix_id.get()

    @classmethod
    @contextmanager
    def scope(cls, prefix_id: str) -> Iterator[None]:
        """
        Context manager for scoped prefix ID usage.

        Automatically sets the prefix ID on entry and clears it on exit,
        ensuring proper cleanup even if exceptions occur.

        Args:
            prefix_id: The unique prefix ID for this scope

        Yields:
            None

        Usage:
            with DynamoPrefixContext.scope("eval-q001"):
                # All LLM calls here will use "eval-q001" prefix
                await llm.ainvoke(...)
        """
        cls.set(prefix_id)
        try:
            yield
        finally:
            cls.clear()


# =============================================================================
# DYNAMO MODEL CONFIGURATION
# =============================================================================


class DynamoModelConfig(OpenAIModelConfig, name="dynamo"):
    """
    A Dynamo LLM provider with automatic prefix header injection for KV cache optimization.

    This is a specialized OpenAI-compatible LLM that sends Dynamo prefix headers
    for optimal KV cache management and request routing. Prefix headers are enabled
    by default using the template "nat-dynamo-{uuid}". The prefix routing parameters
    (prefix_total_requests, prefix_osl, prefix_iat) are optimizable via the NAT optimizer.

    To disable prefix headers, set prefix_template to null/None in your config.
    """

    # =========================================================================
    # DYNAMO PREFIX PARAMETERS
    # =========================================================================

    prefix_template: str | None = Field(
        default="nat-dynamo-{uuid}",
        description="Template for prefix ID. The {uuid} placeholder will be replaced with a unique ID. "
        "Prefix headers are sent by default for KV cache optimization. "
        "Set to null/None to disable prefix header injection.",
    )

    prefix_total_requests: int = OptimizableField(
        default=10,
        ge=1,
        le=50,
        description=("Expected number of requests for this conversation/prefix. "
                     "Higher values increase worker stickiness and KV cache locality. "
                     "Lower values allow more load balancing across workers."),
        space=SearchSpace(low=1, high=20, step=5))

    prefix_osl: PrefixLevel = OptimizableField(default="MEDIUM",
                                               description=("Output Sequence Length hint for the Dynamo router. "
                                                            "LOW=short responses (decode_cost=1.0), "
                                                            "MEDIUM=typical (decode_cost=2.0), "
                                                            "HIGH=long responses (decode_cost=3.0)."),
                                               space=SearchSpace(values=["LOW", "MEDIUM", "HIGH"]))

    prefix_iat: PrefixLevel = OptimizableField(default="MEDIUM",
                                               description=("Inter-Arrival Time hint for the Dynamo router. "
                                                            "LOW=rapid bursts (iat_factor=1.5, high stickiness), "
                                                            "MEDIUM=normal (iat_factor=1.0), "
                                                            "HIGH=slow requests (iat_factor=0.6, more exploration)."),
                                               space=SearchSpace(values=["LOW", "MEDIUM", "HIGH"]))

    request_timeout: float = Field(
        default=600.0,
        gt=0.0,
        description="HTTP request timeout in seconds for LLM requests.",
    )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @staticmethod
    def get_dynamo_field_names() -> frozenset[str]:
        """
        Get the set of Dynamo-specific field names for model_dump exclusion.

        Use this when building config dicts for framework clients to exclude
        Dynamo-specific parameters that should not be passed to the underlying client.

        Returns:
            A frozenset of Dynamo-specific field names.

        Example::

            config_dict = config.model_dump(
                exclude={"type", "thinking", *DynamoModelConfig.get_dynamo_field_names()},
                ...
            )
        """
        return frozenset({
            "prefix_template",
            "prefix_total_requests",
            "prefix_osl",
            "prefix_iat",
            "request_timeout",
        })


# =============================================================================
# HTTPX EVENT HOOK FOR HEADER INJECTION
# =============================================================================


def _create_dynamo_request_hook(
    prefix_template: str | None,
    total_requests: int,
    osl: str,
    iat: str,
) -> Callable[["httpx.Request"], Coroutine[Any, Any, None]]:
    """
    Create an httpx event hook that injects Dynamo prefix headers into requests.

    This hook is called before each HTTP request is sent, allowing us to inject
    headers dynamically. The prefix ID is generated ONCE when the hook is created,
    ensuring all requests from the same client share the same prefix ID. This enables
    Dynamo's KV cache optimization across multi-turn conversations.

    The context variable can override this for scenarios where you need different
    prefix IDs (e.g., per-question in batch evaluation).

    Args:
        prefix_template: Template string with {uuid} placeholder
        total_requests: Expected number of requests for this prefix
        osl: Output sequence length hint (LOW/MEDIUM/HIGH)
        iat: Inter-arrival time hint (LOW/MEDIUM/HIGH)

    Returns:
        An async function suitable for use as an httpx event hook.
    """
    # Generate the default prefix ID ONCE when the hook is created
    # This ensures all requests from this client share the same prefix ID
    unique_id = uuid.uuid4().hex[:16]
    if prefix_template:
        default_prefix_id = prefix_template.format(uuid=unique_id)
    else:
        default_prefix_id = f"nat-dynamo-{unique_id}"

    logger.debug("Created Dynamo request hook with default prefix ID: %s", default_prefix_id)

    async def on_request(request):
        """Inject Dynamo prefix headers before each request."""
        # Check context variable first (allows per-question override in batch evaluation)
        context_prefix_id = DynamoPrefixContext.get()

        if context_prefix_id:
            prefix_id = context_prefix_id
            logger.debug("Using context prefix ID: %s", prefix_id)
        else:
            # Use the pre-generated prefix ID (same for all requests from this client)
            prefix_id = default_prefix_id
            logger.debug("Using default prefix ID: %s", prefix_id)

        # Inject Dynamo headers
        request.headers["x-prefix-id"] = prefix_id
        request.headers["x-prefix-total-requests"] = str(total_requests)
        request.headers["x-prefix-osl"] = osl.upper()
        request.headers["x-prefix-iat"] = iat.upper()

        logger.debug("Injected Dynamo headers: prefix_id=%s, total_requests=%d, osl=%s, iat=%s",
                     prefix_id,
                     total_requests,
                     osl.upper(),
                     iat.upper())

    return on_request


def create_httpx_client_with_dynamo_hooks(
    prefix_template: str | None,
    total_requests: int,
    osl: str,
    iat: str,
    timeout: float = 600.0,
) -> "httpx.AsyncClient":
    """
    Create an httpx.AsyncClient with Dynamo prefix header injection.

    This client can be passed to the OpenAI SDK to inject headers at the HTTP level,
    making it framework-agnostic.

    Args:
        prefix_template: Template string with {uuid} placeholder
        total_requests: Expected number of requests for this prefix
        osl: Output sequence length hint (LOW/MEDIUM/HIGH)
        iat: Inter-arrival time hint (LOW/MEDIUM/HIGH)
        timeout: HTTP request timeout in seconds

    Returns:
        An httpx.AsyncClient configured with Dynamo header injection.
    """
    import httpx

    request_hook = _create_dynamo_request_hook(prefix_template, total_requests, osl, iat)

    return httpx.AsyncClient(
        event_hooks={"request": [request_hook]},
        timeout=httpx.Timeout(timeout),
    )


# =============================================================================
# PROVIDER REGISTRATION
# =============================================================================
# Note: Client registrations for each framework (LangChain, LlamaIndex, etc.)
# are in the respective plugin packages under packages/nvidia_nat_<framework>/


@register_llm_provider(config_type=DynamoModelConfig)
async def dynamo_llm(config: DynamoModelConfig, _builder: Builder):
    """Register the Dynamo LLM provider."""
    yield LLMProviderInfo(
        config=config,
        description="A Dynamo-optimized model with automatic prefix headers for KV cache management.",
    )
