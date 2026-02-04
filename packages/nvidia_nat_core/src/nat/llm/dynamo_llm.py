# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

    from nat.profiler.prediction_trie.trie_lookup import PredictionTrieLookup

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.builder.context import Singleton
from nat.builder.llm import LLMProviderInfo
from nat.cli.register_workflow import register_llm_provider
from nat.data_models.optimizable import OptimizableField
from nat.data_models.optimizable import SearchSpace
from nat.llm.openai_llm import OpenAIModelConfig
from nat.llm.utils.constants import LLMHeaderPrefix
from nat.profiler.prediction_trie.data_models import LLMCallPrediction

logger = logging.getLogger(__name__)

# Define valid prefix hint values
PrefixLevel = Literal["LOW", "MEDIUM", "HIGH"]

# =============================================================================
# CATEGORY CONVERSION HELPERS
# =============================================================================


def _output_tokens_to_osl(output_tokens: float) -> PrefixLevel:
    """
    Convert predicted output tokens to OSL category.

    Thresholds:
        - < 256 tokens: LOW (short responses)
        - < 1024 tokens: MEDIUM (typical responses)
        - >= 1024 tokens: HIGH (long responses)
    """
    if output_tokens < 256:
        return "LOW"
    if output_tokens < 1024:
        return "MEDIUM"
    return "HIGH"


def _interarrival_ms_to_iat(interarrival_ms: float) -> PrefixLevel:
    """
    Convert predicted interarrival time to IAT category.

    Thresholds:
        - < 100ms: LOW (rapid bursts, high worker stickiness)
        - < 500ms: MEDIUM (normal pacing)
        - >= 500ms: HIGH (slow requests, more exploration)
    """
    if interarrival_ms < 100:
        return "LOW"
    if interarrival_ms < 500:
        return "MEDIUM"
    return "HIGH"


# =============================================================================
# CONTEXT MANAGEMENT FOR DYNAMO PREFIX ID
# =============================================================================


class DynamoPrefixContext(metaclass=Singleton):
    """
    Singleton class for managing Dynamo prefix IDs across LLM calls.

    Prefix IDs are unique per depth level in the function call stack, allowing
    different caching behavior at different levels of nested function calls.
    Each depth level gets its own prefix ID that remains constant within a
    single workflow run but changes between runs.

    The prefix ID format is: ``{workflow_run_id}-d{depth}``

    Usage::

        from nat.llm.dynamo_llm import DynamoPrefixContext

        # Automatically gets prefix ID based on current call stack depth
        prefix_id = DynamoPrefixContext.get()

        # Or use as a context manager for explicit control
        with DynamoPrefixContext.scope("eval-q001-abc123"):
            # All LLM calls here will use "eval-q001-abc123" prefix
            ...
    """

    # Maps depth -> prefix_id for the current workflow run
    _prefix_ids_by_depth: ContextVar[dict[int, str] | None] = ContextVar('dynamo_prefix_ids_by_depth', default=None)
    # Optional override that takes precedence over depth-based IDs
    _override_prefix_id: ContextVar[str | None] = ContextVar('dynamo_override_prefix_id', default=None)

    @classmethod
    def _get_current_depth(cls) -> int:
        """Get the current function call stack depth from Context."""
        try:
            ctx = Context.get()
            return len(ctx.function_path)
        except Exception:
            return 0

    @classmethod
    def _get_or_create_depth_map(cls) -> dict[int, str]:
        """Get or create the depth -> prefix_id mapping for this context."""
        depth_map = cls._prefix_ids_by_depth.get()
        if depth_map is None:
            depth_map = {}
            cls._prefix_ids_by_depth.set(depth_map)
        return depth_map

    @classmethod
    def set(cls, prefix_id: str) -> None:
        """
        Set an override prefix ID that takes precedence over depth-based IDs.

        Use this when you need explicit control over the prefix ID, such as
        during batch evaluation where each question should have a specific ID.

        Args:
            prefix_id: The prefix ID to use (overrides depth-based generation)
        """
        cls._override_prefix_id.set(prefix_id)
        logger.debug("Set override Dynamo prefix ID: %s", prefix_id)

    @classmethod
    def clear(cls) -> None:
        """Clear all prefix ID state (both override and depth-based)."""
        cls._override_prefix_id.set(None)
        cls._prefix_ids_by_depth.set(None)
        logger.debug("Cleared Dynamo prefix ID context")

    @classmethod
    def get(cls) -> str:
        """
        Get the Dynamo prefix ID for the current context.

        Returns the override prefix ID if set, otherwise returns a depth-based
        prefix ID that is unique per workflow run and call stack depth.

        Returns:
            The prefix ID string, never None.
        """
        # Check for override first
        override = cls._override_prefix_id.get()
        if override:
            return override

        # Get depth-based prefix ID
        depth = cls._get_current_depth()
        depth_map = cls._get_or_create_depth_map()

        if depth not in depth_map:
            # Generate new prefix ID for this depth
            try:
                ctx = Context.get()
                workflow_id = ctx.workflow_run_id
            except Exception:
                workflow_id = None

            if not workflow_id:
                logger.warning("No workflow_run_id in context; using unique prefix ID.")
                workflow_id = uuid.uuid4().hex[:16]

            prefix_id = f"{workflow_id}-d{depth}"
            depth_map[depth] = prefix_id
            logger.debug("Generated Dynamo prefix ID for depth %d: %s", depth, prefix_id)

        return depth_map[depth]

    @classmethod
    def is_set(cls) -> bool:
        """Check if a Dynamo prefix ID is available (always True, IDs are auto-generated)."""
        return True

    @classmethod
    @contextmanager
    def scope(cls, prefix_id: str) -> Iterator[None]:
        """
        Context manager for scoped override prefix ID usage.

        Sets an override prefix ID on entry and restores the previous state on exit,
        ensuring proper cleanup even if exceptions occur. Supports nesting.

        Args:
            prefix_id: The override prefix ID for this scope

        Yields:
            None

        Usage:
            with DynamoPrefixContext.scope("eval-q001"):
                # All LLM calls here will use "eval-q001" prefix
                await llm.ainvoke(...)
        """
        previous_override = cls._override_prefix_id.get()
        cls.set(prefix_id)
        try:
            yield
        finally:
            cls._override_prefix_id.set(previous_override)


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

    prefix_osl: PrefixLevel = OptimizableField(
        default="MEDIUM",
        description="Output Sequence Length hint for the Dynamo router. "
        "LOW means short responses (decode_cost=1.0), "
        "MEDIUM means typical (decode_cost=2.0), "
        "HIGH means long responses (decode_cost=3.0).",
        space=SearchSpace(values=["LOW", "MEDIUM", "HIGH"]),
    )

    prefix_iat: PrefixLevel = OptimizableField(
        default="MEDIUM",
        description="Inter-Arrival Time hint for the Dynamo router. "
        "LOW means rapid bursts (iat_factor=1.5, high stickiness), "
        "MEDIUM means normal (iat_factor=1.0), "
        "HIGH means slow requests (iat_factor=0.6, more exploration).",
        space=SearchSpace(values=["LOW", "MEDIUM", "HIGH"]),
    )

    request_timeout: float = Field(
        default=600.0,
        gt=0.0,
        description="HTTP request timeout in seconds for LLM requests.",
    )

    prediction_trie_path: str | None = Field(
        default=None,
        description="Path to prediction_trie.json file. When set, predictions are "
        "looked up and injected as headers for each LLM call.",
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
            "prediction_trie_path",
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
    headers dynamically. The prefix ID is obtained from DynamoPrefixContext which
    provides depth-aware prefix IDs - each level in the function call stack gets
    its own unique prefix ID that remains constant within a workflow run.

    Args:
        prefix_template: Template string with {uuid} placeholder (unused, for API compat).
        total_requests: Expected number of requests for this prefix.
        osl: Output sequence length hint (LOW/MEDIUM/HIGH).
        iat: Inter-arrival time hint (LOW/MEDIUM/HIGH).

    Returns:
        An async function suitable for use as an httpx event hook.
    """
    # Note: prefix_template is kept for API compatibility but no longer used.
    # Prefix IDs are now managed by DynamoPrefixContext with depth-awareness.
    _ = prefix_template  # Suppress unused parameter warning

    async def on_request(request):
        """Inject Dynamo prefix headers before each request."""
        # Get depth-aware prefix ID from context
        prefix_id = DynamoPrefixContext.get()
        logger.debug("Using depth-aware prefix ID: %s", prefix_id)

        # Inject Dynamo headers
        request.headers[f"{LLMHeaderPrefix.DYNAMO.value}-id"] = prefix_id
        request.headers[f"{LLMHeaderPrefix.DYNAMO.value}-total-requests"] = str(total_requests)
        request.headers[f"{LLMHeaderPrefix.DYNAMO.value}-osl"] = osl.upper()
        request.headers[f"{LLMHeaderPrefix.DYNAMO.value}-iat"] = iat.upper()

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
    prediction_lookup: "PredictionTrieLookup | None" = None,
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
        prediction_lookup: Optional PredictionTrieLookup for dynamic header injection

    Returns:
        An httpx.AsyncClient configured with Dynamo header injection.
    """
    import httpx

    hooks: list[Callable] = []

    # Add Dynamo prefix hook
    prefix_hook = _create_dynamo_request_hook(prefix_template, total_requests, osl, iat)
    hooks.append(prefix_hook)

    # Add dynamic prediction hook if lookup provided
    if prediction_lookup is not None:
        prediction_hook = _create_dynamic_prediction_hook(prediction_lookup)
        hooks.append(prediction_hook)

    return httpx.AsyncClient(
        event_hooks={"request": hooks},
        timeout=httpx.Timeout(timeout),
    )


def _create_prediction_request_hook(
    prediction: LLMCallPrediction, ) -> Callable[["httpx.Request"], Coroutine[Any, Any, None]]:
    """
    Create an httpx event hook that overrides x-prefix-* headers from static prediction data.

    This hook converts numeric prediction values to categorical values (LOW/MEDIUM/HIGH)
    and overrides the x-prefix-* headers set by the Dynamo prefix hook.

    Args:
        prediction: The prediction data to inject

    Returns:
        An async function suitable for use as an httpx event hook.
    """
    # Pre-compute categorical values from prediction
    total_requests = int(prediction.remaining_calls.mean)
    osl = _output_tokens_to_osl(prediction.output_tokens.p90)
    iat = _interarrival_ms_to_iat(prediction.interarrival_ms.mean)

    async def on_request(request):
        """Override x-prefix-* headers with prediction-derived values."""
        request.headers[f"{LLMHeaderPrefix.DYNAMO.value}-total-requests"] = str(total_requests)
        request.headers[f"{LLMHeaderPrefix.DYNAMO.value}-osl"] = osl
        request.headers[f"{LLMHeaderPrefix.DYNAMO.value}-iat"] = iat

        logger.debug(
            "Overrode prefix headers from static prediction: total_requests=%d, osl=%s, iat=%s",
            total_requests,
            osl,
            iat,
        )

    return on_request


def _create_dynamic_prediction_hook(
    trie_lookup: "PredictionTrieLookup", ) -> Callable[["httpx.Request"], Coroutine[Any, Any, None]]:
    """
    Create an httpx event hook that dynamically looks up predictions per request.

    This hook reads the current function path and call index from context,
    looks up the prediction in the trie, and overrides the x-prefix-* headers
    with values derived from the prediction. The numeric prediction values
    are converted to categorical values (LOW/MEDIUM/HIGH) for consistency
    with static configuration.

    When a prediction is found, this hook overrides:
        - x-prefix-total-requests: from remaining_calls.mean
        - x-prefix-osl: converted from output_tokens.p90
        - x-prefix-iat: converted from interarrival_ms.mean

    Args:
        trie_lookup: The PredictionTrieLookup instance to query

    Returns:
        An async function suitable for use as an httpx event hook.
    """

    async def on_request(request: "httpx.Request") -> None:
        """Look up prediction from context and override x-prefix-* headers."""
        from nat.llm.prediction_context import get_call_tracker

        try:
            ctx = Context.get()
            path = ctx.function_path

            # Get call index for current parent function
            call_index = 1  # default
            active_fn = ctx.active_function
            if active_fn and active_fn.function_id != "root":
                tracker = get_call_tracker()
                call_index = tracker.counts.get(active_fn.function_id, 1)

            # Look up prediction
            prediction = trie_lookup.find(path, call_index)

            if prediction:
                # Convert numeric predictions to categorical values and override headers
                total_requests = int(prediction.remaining_calls.mean)
                osl = _output_tokens_to_osl(prediction.output_tokens.p90)
                iat = _interarrival_ms_to_iat(prediction.interarrival_ms.mean)

                request.headers[f"{LLMHeaderPrefix.DYNAMO.value}-total-requests"] = str(total_requests)
                request.headers[f"{LLMHeaderPrefix.DYNAMO.value}-osl"] = osl
                request.headers[f"{LLMHeaderPrefix.DYNAMO.value}-iat"] = iat

                logger.debug(
                    "Overrode prefix headers from prediction: path=%s, call_index=%d, "
                    "total_requests=%d, osl=%s (tokens=%d), iat=%s (ms=%d)",
                    path,
                    call_index,
                    total_requests,
                    osl,
                    int(prediction.output_tokens.p90),
                    iat,
                    int(prediction.interarrival_ms.mean),
                )
            else:
                logger.debug("No prediction found for path=%s, call_index=%d; using static values", path, call_index)

        except Exception as e:
            # Don't fail the request if prediction lookup fails
            logger.warning("Failed to override prefix headers from prediction: %s", e)

    return on_request


def create_httpx_client_with_prediction_headers(
    prediction: LLMCallPrediction,
    prefix_template: str | None,
    total_requests: int,
    osl: str,
    iat: str,
    timeout: float = 600.0,
) -> "httpx.AsyncClient":
    """
    Create an httpx.AsyncClient with both Dynamo prefix and prediction headers.

    Args:
        prediction: Prediction data for this LLM call
        prefix_template: Template string with {uuid} placeholder
        total_requests: Expected number of requests for this prefix
        osl: Output sequence length hint (LOW/MEDIUM/HIGH)
        iat: Inter-arrival time hint (LOW/MEDIUM/HIGH)
        timeout: HTTP request timeout in seconds

    Returns:
        An httpx.AsyncClient configured with header injection.
    """
    import httpx

    hooks: list[Callable] = []

    # Add Dynamo prefix hook
    prefix_hook = _create_dynamo_request_hook(prefix_template, total_requests, osl, iat)
    hooks.append(prefix_hook)

    # Add prediction hook
    prediction_hook = _create_prediction_request_hook(prediction)
    hooks.append(prediction_hook)

    return httpx.AsyncClient(
        event_hooks={"request": hooks},
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
