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
Dynamo LLM provider with automatic prefix hint injection for KV cache optimization.

This module provides a specialized OpenAI-compatible LLM that sends Dynamo prefix hints
for optimal KV cache management and request routing. The prefix parameters are optimizable
via the NAT optimizer.

The implementation uses a custom httpx transport to inject hints at the HTTP level,
making it framework-agnostic (works with LangChain, LlamaIndex, etc.).

Transport Mechanisms
--------------------

This module supports two transport mechanisms for routing hints, used simultaneously
for maximum compatibility:

1. **HTTP Headers** (``x-prefix-*``): For the generalized Thompson Sampling setup
   that uses custom ``frontend.py`` which reads headers directly.

2. **nvext.agent_hints** (in request body): For the optimized Thompson Sampling setup
   that uses the default Dynamo frontend with custom ``processor.py`` which reads
   agent_hints from the preprocessed request. *This is the preferred mechanism.*

Dynamo Prefix Parameters
-------------------------

prefix_osl (Output Sequence Length)
    Expected output tokens for response length hinting. By default, the raw
    integer value is sent. When ``prefix_use_raw_values`` is False, values are
    converted to categories:

    - < 256 tokens: LOW (decode_cost=1.0, short responses)
    - < 1024 tokens: MEDIUM (decode_cost=2.0, typical responses)
    - >= 1024 tokens: HIGH (decode_cost=3.0, long responses)

    Accepts categorical strings (LOW/MEDIUM/HIGH) for backward compatibility,
    which are converted to representative token counts (128/512/2048).

prefix_iat (Inter-Arrival Time)
    Expected inter-arrival time in milliseconds. By default, the raw integer
    value is sent. When ``prefix_use_raw_values`` is False, values are converted
    to categories:

    - < 100ms: LOW (iat_factor=1.5, rapid bursts, high worker stickiness)
    - < 500ms: MEDIUM (iat_factor=1.0, normal pacing)
    - >= 500ms: HIGH (iat_factor=0.6, slow requests, more exploration)

    Accepts categorical strings (LOW/MEDIUM/HIGH) for backward compatibility,
    which are converted to representative millisecond values (50/250/750).

prefix_total_requests
    Expected requests per conversation:

    - Higher values increase KV cache affinity and worker stickiness
    - Lower values allow more load balancing
"""

import json
import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING
from typing import Literal

import httpx

if TYPE_CHECKING:
    from nat.profiler.prediction_trie.trie_lookup import PredictionTrieLookup

from pydantic import Field
from pydantic import field_validator

from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.builder.context import Singleton
from nat.builder.llm import LLMProviderInfo
from nat.cli.register_workflow import register_llm_provider
from nat.data_models.optimizable import OptimizableField
from nat.data_models.optimizable import SearchSpace
from nat.llm.openai_llm import OpenAIModelConfig
from nat.llm.utils.constants import LLMHeaderPrefix

logger = logging.getLogger(__name__)

# Define valid prefix hint values
PrefixLevel = Literal["LOW", "MEDIUM", "HIGH"]

# Representative token counts for categorical levels (midpoint of ranges):
# LOW: 128 tokens (midpoint of 0-256 range)
# MEDIUM: 512 tokens (midpoint of 256-1024 range)
# HIGH: 2048 tokens (midpoint of 1024-4096 range)
_OSL_CATEGORY_TO_INT: dict[str, int] = {"LOW": 128, "MEDIUM": 512, "HIGH": 2048}
# Representative interarrival times for categorical levels (midpoint of ranges):
# LOW: 50ms (midpoint of 0-100ms range)
# MEDIUM: 250ms (midpoint of 100-500ms range)
# HIGH: 750ms (midpoint of 500-1000ms range)
_IAT_CATEGORY_TO_INT: dict[str, int] = {"LOW": 50, "MEDIUM": 250, "HIGH": 750}

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
    A Dynamo LLM provider with automatic prefix hint injection for KV cache optimization.

    This is a specialized OpenAI-compatible LLM that sends Dynamo prefix hints
    for optimal KV cache management and request routing. Prefix hints are enabled
    by default using the template "nat-dynamo-{uuid}". The prefix routing parameters
    (prefix_total_requests, prefix_osl, prefix_iat) are optimizable via the NAT optimizer.

    Hints are sent via both HTTP headers (``x-prefix-*``) and ``nvext.agent_hints``
    in the request body for compatibility with different Dynamo setups:

    - **Generalized Thompson Sampling** (custom frontend.py): Reads HTTP headers
    - **Optimized Thompson Sampling** (default frontend + processor.py): Reads nvext.agent_hints

    To disable prefix hints, set prefix_template to null/None in your config.
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

    prefix_osl: int = OptimizableField(
        default=512,
        ge=1,
        description="Expected output tokens for response length hinting (Output Sequence Length). "
        "Raw integer value is sent by default. Accepts categorical strings "
        "(LOW/MEDIUM/HIGH) for backward compatibility (mapped to 128/512/2048).",
        space=SearchSpace(low=64, high=4096, step=64),
    )

    prefix_iat: int = OptimizableField(
        default=250,
        ge=1,
        description="Expected inter-arrival time in milliseconds for request pacing. "
        "Raw integer value is sent by default. Accepts categorical strings "
        "(LOW/MEDIUM/HIGH) for backward compatibility (mapped to 50/250/750).",
        space=SearchSpace(low=10, high=1000, step=50),
    )

    request_timeout: float = Field(
        default=600.0,
        gt=0.0,
        description="HTTP request timeout in seconds for LLM requests.",
    )

    prefix_use_raw_values: bool = Field(
        default=True,
        description="When True, send raw integer values for OSL (output tokens) and IAT (interarrival ms) "
        "in headers and nvext.agent_hints. When False, convert to categorical LOW/MEDIUM/HIGH.",
    )

    prediction_trie_path: str | None = Field(
        default=None,
        description="Path to prediction_trie.json file. When set, predictions are "
        "looked up and used to override both HTTP headers and nvext.agent_hints for each LLM call.",
    )

    disable_headers: bool = Field(
        default=True,
        description="If True, do not inject Dynamo prefix hints as HTTP headers. "
        "Hints will still be injected via nvext.agent_hints in the request body if prefix_template is set.",
    )

    # =========================================================================
    # VALIDATORS (backward compatibility: categorical strings -> integers)
    # =========================================================================

    @field_validator("prefix_osl", mode="before")
    @classmethod
    def _coerce_prefix_osl(cls, v: object) -> int:
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            upper = v.upper()
            if upper in _OSL_CATEGORY_TO_INT:
                return _OSL_CATEGORY_TO_INT[upper]
            raise ValueError(f"Invalid OSL value '{v}'. Must be an integer >= 1 "
                             f"or one of: {', '.join(_OSL_CATEGORY_TO_INT.keys())}")
        raise TypeError(f"prefix_osl must be int or str, got {type(v)}")

    @field_validator("prefix_iat", mode="before")
    @classmethod
    def _coerce_prefix_iat(cls, v: object) -> object:
        """Convert categorical IAT strings (LOW/MEDIUM/HIGH) to representative millisecond values."""
        if isinstance(v, str) and v.upper() in _IAT_CATEGORY_TO_INT:
            return _IAT_CATEGORY_TO_INT[v.upper()]
        return v

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
            "prefix_use_raw_values",
            "request_timeout",
            "prediction_trie_path",
            "disable_headers",
        })


# =============================================================================
# CUSTOM TRANSPORT FOR DYNAMO HINT INJECTION
# =============================================================================


class _DynamoTransport(httpx.AsyncBaseTransport):
    """
    Custom transport wrapper that injects both HTTP headers and nvext.agent_hints.

    This approach is more reliable than event hooks because it modifies the request
    BEFORE httpx's internal state machine processes it. It supports both transport
    mechanisms simultaneously for maximum compatibility:

    - HTTP headers (``x-prefix-*``): For generalized Thompson Sampling setup
    - nvext.agent_hints: For optimized Thompson Sampling setup (preferred)
    """

    def __init__(
        self,
        transport: httpx.AsyncBaseTransport,
        total_requests: int,
        osl: int,
        iat: int,
        prediction_lookup: "PredictionTrieLookup | None" = None,
        use_raw_values: bool = True,
        disable_headers: bool = True,
    ):
        self._transport = transport
        self._total_requests = total_requests
        self._osl = osl
        self._iat = iat
        self._prediction_lookup = prediction_lookup
        self._use_raw_values = use_raw_values
        self._disable_headers = disable_headers

    async def handle_async_request(self, request: "httpx.Request") -> "httpx.Response":
        # Get prefix ID from context (supports depth-awareness and overrides)
        prefix_id = DynamoPrefixContext.get()

        # Get latency sensitivity from context (defaults to MEDIUM)
        try:
            ctx = Context.get()
            latency_sensitivity = str(ctx.latency_sensitivity.value)
        except Exception:
            # If context not available or latency_sensitivity not implemented yet, default to MEDIUM
            latency_sensitivity = "MEDIUM"

        # Initialize with static config values (always integers)
        total_requests = self._total_requests
        osl_raw = self._osl
        iat_raw = self._iat

        # Check for prediction override
        if self._prediction_lookup is not None:
            try:
                from nat.llm.prediction_context import get_call_tracker

                ctx = Context.get()
                path = ctx.function_path

                # Get call index for current parent function
                call_index = 1
                active_fn = ctx.active_function
                if active_fn and active_fn.function_id != "root":
                    tracker = get_call_tracker()
                    call_index = tracker.counts.get(active_fn.function_id, 1)

                # Look up prediction
                prediction = self._prediction_lookup.find(path, call_index)

                if prediction:
                    # Override with prediction-derived values
                    total_requests = int(prediction.remaining_calls.mean)
                    osl_raw = int(prediction.output_tokens.p90)
                    iat_raw = int(prediction.interarrival_ms.mean)

                    logger.debug(
                        "Overriding hints from prediction: path=%s, call_index=%d, "
                        "total_requests=%d, osl_raw=%d, iat_raw=%d",
                        path,
                        call_index,
                        total_requests,
                        osl_raw,
                        iat_raw,
                    )
                else:
                    logger.debug(
                        "No prediction found for path=%s, call_index=%d; using static values",
                        path,
                        call_index,
                    )

            except Exception:
                logger.exception("Failed to lookup prediction")

        # Compute final values for headers/body
        if self._use_raw_values:
            osl_value: int | str = osl_raw
            iat_value: int | str = iat_raw
        else:
            osl_value = _output_tokens_to_osl(osl_raw)
            iat_value = _interarrival_ms_to_iat(iat_raw)

        headers = dict(request.headers)
        if not self._disable_headers:
            # Headers always need strings
            headers[f"{LLMHeaderPrefix.DYNAMO}-id"] = prefix_id
            headers[f"{LLMHeaderPrefix.DYNAMO}-total-requests"] = str(total_requests)
            headers[f"{LLMHeaderPrefix.DYNAMO}-osl"] = str(osl_value)
            headers[f"{LLMHeaderPrefix.DYNAMO}-iat"] = str(iat_value)
            headers[f"{LLMHeaderPrefix.DYNAMO}-latency-sensitivity"] = latency_sensitivity

        # Modify body to inject nvext.agent_hints (if JSON POST request)
        content = request.content
        if request.method == "POST" and content:
            try:
                body = json.loads(content.decode("utf-8", errors="replace"))
                if isinstance(body, dict):
                    # Build agent_hints dict (int or str depending on raw mode)
                    agent_hints = {
                        "prefix_id": prefix_id,
                        "total_requests": total_requests,
                        "osl": osl_value,
                        "iat": iat_value,
                        "latency_sensitivity": latency_sensitivity,
                    }

                    # Add/merge nvext.agent_hints
                    if "nvext" not in body:
                        body["nvext"] = {}
                    if not isinstance(body["nvext"], dict):
                        body["nvext"] = {}

                    existing = body["nvext"].get("agent_hints", {})
                    if not isinstance(existing, dict):
                        existing = {}

                    # Our hints take precedence over existing
                    body["nvext"]["agent_hints"] = {**existing, **agent_hints}

                    # Re-encode
                    content = json.dumps(body).encode("utf-8")
                    headers["content-length"] = str(len(content))

                    logger.debug("Injected nvext.agent_hints: %s (body size: %d bytes)",
                                 body["nvext"]["agent_hints"],
                                 len(content))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.debug("Could not inject nvext.agent_hints: %s", e)

        # Create new request with modified headers and content
        new_request = httpx.Request(
            method=request.method,
            url=request.url,
            headers=headers,
            content=content,
            extensions=request.extensions,
        )

        logger.debug("Injected Dynamo hints: prefix_id=%s, total_requests=%d, osl=%s, iat=%s, latency_sensitivity=%s",
                     prefix_id,
                     total_requests,
                     osl_value,
                     iat_value,
                     latency_sensitivity)

        return await self._transport.handle_async_request(new_request)

    async def aclose(self) -> None:
        """Close the underlying transport."""
        await self._transport.aclose()


# =============================================================================
# HTTPX CLIENT CREATION
# =============================================================================


def create_httpx_client_with_dynamo_hooks(
    prefix_template: str | None,
    total_requests: int,
    osl: int,
    iat: int,
    timeout: float = 600.0,
    prediction_lookup: "PredictionTrieLookup | None" = None,
    use_raw_values: bool = True,
    disable_headers: bool = True,
) -> "httpx.AsyncClient":
    """
    Create an httpx.AsyncClient with Dynamo hint injection via custom transport.

    This client can be passed to the OpenAI SDK to inject hints at the HTTP level,
    making it framework-agnostic. Hints are injected via both HTTP headers and
    nvext.agent_hints in the request body for maximum compatibility:

    - HTTP headers (``x-prefix-*``): For generalized Thompson Sampling setup
    - nvext.agent_hints: For optimized Thompson Sampling setup (preferred)

    Args:
        prefix_template: Template string with {uuid} placeholder (unused, kept for API compat)
        total_requests: Expected number of requests for this prefix
        osl: Expected output tokens (raw integer value)
        iat: Expected inter-arrival time in milliseconds (raw integer value)
        timeout: HTTP request timeout in seconds
        prediction_lookup: Optional PredictionTrieLookup for dynamic hint injection
        use_raw_values: When True send raw integers; when False convert to LOW/MEDIUM/HIGH
        disable_headers: If True, do not inject hints as HTTP headers (still injects nvext.agent_hints)

    Returns:
        An httpx.AsyncClient configured with Dynamo hint injection.
    """
    import httpx

    # Note: prefix_template is kept for API compatibility but no longer used.
    # Prefix IDs are now managed by DynamoPrefixContext with depth-awareness.
    _ = prefix_template

    # Create base transport and wrap with custom transport
    base_transport = httpx.AsyncHTTPTransport()
    dynamo_transport = _DynamoTransport(
        transport=base_transport,
        total_requests=total_requests,
        osl=osl,
        iat=iat,
        prediction_lookup=prediction_lookup,
        use_raw_values=use_raw_values,
        disable_headers=disable_headers,
    )

    return httpx.AsyncClient(
        transport=dynamo_transport,
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
