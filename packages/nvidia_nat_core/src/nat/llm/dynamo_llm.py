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
Dynamo LLM provider with automatic nvext.agent_hints and nvnext.cache_control injection for KV cache optimization.

This module provides a specialized OpenAI-compatible LLM that sends Dynamo routing
hints for optimal KV cache management and request routing. The hint parameters are
optimizable via the NAT optimizer.

The implementation uses a custom httpx transport to inject hints at the HTTP level,
making it framework-agnostic (works with LangChain, LlamaIndex, ADK).

Transport Mechanism
-------------------

All routing hints are injected into **nvext.agent_hints** (dict in the request body).
The default Dynamo frontend passes this through to the preprocessed request, and our
custom ``processor.py`` reads the routing fields directly from ``agent_hints``.

Standard Dynamo fields (``latency_sensitivity``, ``osl``, ``priority``) are consumed
by Dynamo's built-in router and engine scheduler. Custom fields (``prefix_id``,
``total_requests``, ``iat``) are consumed by our custom ``processor.py``.

nvext Hint Parameters
---------------------

nvext_prefix_osl (Output Sequence Length)
    Expected output tokens for response length hinting. Raw integer value is always
    sent in ``nvext.agent_hints``. Accepts categorical strings (LOW/MEDIUM/HIGH) for
    backward compatibility, which are converted to representative token counts
    (128/512/2048).

nvext_prefix_iat (Inter-Arrival Time)
    Expected inter-arrival time in milliseconds. Raw integer value is always sent in
    ``nvext.agent_hints``. Accepts categorical strings (LOW/MEDIUM/HIGH) for backward
    compatibility, which are converted to representative millisecond values
    (50/250/750).

nvext_prefix_total_requests
    Expected requests per conversation:

    - Higher values increase KV cache affinity and worker stickiness
    - Lower values allow more load balancing
"""

import json
import logging
import threading
import uuid
import warnings
from collections.abc import Iterator
from contextlib import asynccontextmanager
from contextlib import contextmanager
from contextvars import ContextVar
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from nat.profiler.prediction_trie.trie_lookup import PredictionTrieLookup

from pydantic import AliasChoices
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

logger = logging.getLogger(__name__)

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

# Fallback when Context is unavailable (e.g. outside a workflow run).
# Mid-range default on the [0, max_sensitivity] scale.
_DEFAULT_LATENCY_SENSITIVITY: int = 2


class CachePinType(StrEnum):
    """Cache pinning strategy for KV cache entries.

    Controls how aggressively the Dynamo KV cache retains entries for a prefix:

    - EPHEMERAL: Cache entries auto-expire after a computed TTL of inactivity.
      TTL is ``total_requests * iat`` (the estimated total conversation
      duration in milliseconds), giving the expected time span over which
      this prefix's cache entries should be retained before eviction.
    """

    EPHEMERAL = "ephemeral"


class CacheControlMode(StrEnum):
    """Controls when ``nvext.cache_control`` is injected into requests.

    - ALWAYS: Inject on every request (refreshes TTL each turn).
    - FIRST_ONLY: Inject only on the first request per prefix_id, pinning
      the system prompt when it is first established in the KV cache.
      Subsequent requests benefit from prefix matching without re-pinning
      the growing conversation context.
    """

    ALWAYS = "always"
    FIRST_ONLY = "first_only"


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
    A Dynamo LLM provider with automatic nvext.agent_hints and nvext.cache_control injection for KV cache optimization.

    This is a specialized OpenAI-compatible LLM that sends Dynamo routing hints
    for optimal KV cache management and request routing. Hints are injected when
    ``enable_nvext_hints`` is True. The hint parameters (nvext_prefix_total_requests,
    nvext_prefix_osl, nvext_prefix_iat) are optimizable via the NAT optimizer.

    All hints are sent via ``nvext.agent_hints`` in the request body. Standard Dynamo
    fields (``latency_sensitivity``, ``osl``, ``priority``) are consumed by Dynamo's
    built-in router and engine scheduler. Custom fields (``prefix_id``,
    ``total_requests``, ``iat``) are consumed by the custom ``processor.py``.

    To disable hints, set ``enable_nvext_hints: false`` in your config (the default).
    """

    # =========================================================================
    # NVEXT HINT PARAMETERS
    # =========================================================================

    enable_nvext_hints: bool = Field(
        default=False,
        description="When True, inject nvext.agent_hints and nvext.cache_control "
        "into requests via a custom httpx transport. "
        "When False (default), no routing hints are injected.",
    )

    nvext_prefix_id_template: str | None = Field(
        default="nat-dynamo-{uuid}",
        description="Template for prefix ID. The {uuid} placeholder will be replaced with a unique ID. "
        "Currently unused by the transport (prefix IDs come from DynamoPrefixContext), "
        "but retained for configuration reference.",
    )

    nvext_prefix_total_requests: int = OptimizableField(
        default=10,
        ge=1,
        le=50,
        description=("Expected number of requests for this conversation/prefix. "
                     "Higher values increase worker stickiness and KV cache locality. "
                     "Lower values allow more load balancing across workers."),
        space=SearchSpace(low=1, high=20, step=5))

    nvext_prefix_osl: int = OptimizableField(
        default=512,
        ge=1,
        description="Expected output tokens for response length hinting (Output Sequence Length). "
        "Raw integer value is sent in nvext.agent_hints. Accepts categorical strings "
        "(LOW/MEDIUM/HIGH) for backward compatibility (mapped to 128/512/2048).",
        space=SearchSpace(low=64, high=4096, step=64),
    )

    nvext_prefix_iat: int = OptimizableField(
        default=250,
        ge=1,
        description="Expected inter-arrival time in milliseconds for request pacing. "
        "Raw integer value is sent in nvext.agent_hints. Accepts categorical strings "
        "(LOW/MEDIUM/HIGH) for backward compatibility (mapped to 50/250/750).",
        space=SearchSpace(low=10, high=1000, step=50),
    )

    request_timeout: float = Field(
        default=600.0,
        gt=0.0,
        description="HTTP request timeout in seconds for LLM requests.",
    )

    nvext_prediction_trie_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("nvext_prediction_trie_path", "prediction_trie_path"),
        description="Path to prediction_trie.json file. When set, predictions are "
        "looked up and used to override nvext.agent_hints for each LLM call.",
    )

    nvext_cache_pin_type: CachePinType | None = Field(
        default=CachePinType.EPHEMERAL,
        description="Cache pinning strategy for KV cache entries. "
        "When set, injects nvext.cache_control with the pin type and a TTL "
        "computed as total_requests * iat (estimated conversation duration in ms). "
        "Set to null/None to disable cache control hints.",
    )

    nvext_cache_control_mode: CacheControlMode = Field(
        default=CacheControlMode.ALWAYS,
        description="Controls when nvext.cache_control is injected. "
        "'always' injects on every request (refreshes TTL each turn). "
        "'first_only' injects only on the first request per prefix_id, "
        "pinning the system prompt when it is first established in the KV cache.",
    )

    nvext_max_sensitivity: int = Field(
        default=1000,
        ge=1,
        validation_alias=AliasChoices("nvext_max_sensitivity", "max_sensitivity"),
        description="Maximum latency sensitivity value used to compute request priority. "
        "Priority is the integer complement: priority = max_sensitivity - latency_sensitivity. "
        "Lower priority values indicate higher priority requests.",
    )

    # =========================================================================
    # VALIDATORS (backward compatibility: categorical strings -> integers)
    # =========================================================================

    @field_validator("nvext_prefix_osl", mode="before")
    @classmethod
    def _coerce_nvext_prefix_osl(cls, v: object) -> int:
        """Convert categorical OSL strings (LOW/MEDIUM/HIGH) to representative token counts."""
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            upper = v.upper()
            if upper in _OSL_CATEGORY_TO_INT:
                return _OSL_CATEGORY_TO_INT[upper]
            raise ValueError(f"Invalid OSL value '{v}'. Must be an integer >= 1 "
                             f"or one of: {', '.join(_OSL_CATEGORY_TO_INT.keys())}")
        raise TypeError(f"nvext_prefix_osl must be int or str, got {type(v)}")

    @field_validator("nvext_prefix_iat", mode="before")
    @classmethod
    def _coerce_nvext_prefix_iat(cls, v: object) -> int:
        """Convert categorical IAT strings (LOW/MEDIUM/HIGH) to representative millisecond values."""
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            upper = v.upper()
            if upper in _IAT_CATEGORY_TO_INT:
                return _IAT_CATEGORY_TO_INT[upper]
            raise ValueError(f"Invalid IAT value '{v}'. Must be an integer >= 1 "
                             f"or one of: {', '.join(_IAT_CATEGORY_TO_INT.keys())}")
        raise TypeError(f"nvext_prefix_iat must be int or str, got {type(v)}")

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
            "enable_nvext_hints",
            "nvext_prefix_id_template",
            "nvext_prefix_total_requests",
            "nvext_prefix_osl",
            "nvext_prefix_iat",
            "request_timeout",
            "nvext_prediction_trie_path",
            "nvext_cache_pin_type",
            "nvext_cache_control_mode",
            "nvext_max_sensitivity",
        })


# =============================================================================
# CUSTOM TRANSPORT FOR DYNAMO HINT INJECTION
# =============================================================================


class _DynamoTransport(httpx.AsyncBaseTransport):
    """
    Custom transport wrapper that injects all routing hints into nvext.agent_hints.

    This approach is more reliable than event hooks because it modifies the request
    BEFORE httpx's internal state machine processes it.

    All hints are placed in a single ``nvext.agent_hints`` dict:

    - Standard Dynamo fields (``latency_sensitivity``, ``osl``, ``priority``): consumed
      by Dynamo's built-in router and engine scheduler.
    - Custom routing fields (``prefix_id``, ``total_requests``, ``iat``): consumed by
      the custom ``processor.py`` for Thompson Sampling worker selection.
    """

    def __init__(
        self,
        transport: httpx.AsyncBaseTransport,
        total_requests: int,
        osl: int,
        iat: int,
        prediction_lookup: "PredictionTrieLookup | None" = None,
        cache_pin_type: CachePinType | None = CachePinType.EPHEMERAL,
        cache_control_mode: CacheControlMode = CacheControlMode.ALWAYS,
        max_sensitivity: int = 1000,
    ):
        self._transport = transport
        self._total_requests = total_requests
        self._osl = osl
        self._iat = iat
        self._prediction_lookup = prediction_lookup
        self._cache_pin_type = cache_pin_type
        self._cache_control_mode = cache_control_mode
        self._max_sensitivity = max_sensitivity
        # Per-prefix call counter so call_index advances across requests
        # for the same prefix_id (keyed by prefix_id string).
        self._call_counts: dict[str, int] = {}
        self._call_counts_lock = threading.Lock()

        if cache_pin_type is not None:
            warnings.warn(
                f"nvext.cache_control is configured (type={cache_pin_type.value}). cache_control requires "
                "sglang >v0.5.9 with hierarchical cache enabled. Parameters will be "
                "sent but may be silently ignored by the backend. "
                "See https://github.com/sgl-project/sglang/pull/18941",
                stacklevel=2,
            )

    async def handle_async_request(self, request: "httpx.Request") -> "httpx.Response":
        # Get prefix ID from context (supports depth-awareness and overrides)
        prefix_id = DynamoPrefixContext.get()

        # Get latency sensitivity from context.
        # Context.latency_sensitivity is typed as int; coerce
        # defensively in case a subclass or mock returns a float.
        try:
            ctx = Context.get()
            latency_sensitivity = int(ctx.latency_sensitivity)
        except Exception:
            latency_sensitivity = _DEFAULT_LATENCY_SENSITIVITY

        # Initialize with static config values (always integers)
        total_requests = self._total_requests
        osl_raw = self._osl
        iat_raw = self._iat

        # Read the tentative per-prefix call index for prediction trie lookups.
        # The counter is committed to _call_counts only after the request is
        # confirmed eligible for injection (see below), so non-injectable requests
        # (non-POST, empty body, invalid JSON, non-dict body) do not consume the
        # FIRST_ONLY slot.
        with self._call_counts_lock:
            call_index = self._call_counts.get(prefix_id, 0) + 1

        # Check for prediction override
        if self._prediction_lookup is not None:
            try:
                ctx = Context.get()
                path = ctx.function_path

                # Look up prediction
                prediction = self._prediction_lookup.find(path, call_index)

                if prediction:
                    # Override with prediction-derived values
                    total_requests = int(prediction.remaining_calls.mean)
                    osl_raw = int(prediction.output_tokens.p90)
                    iat_raw = int(prediction.interarrival_ms.mean)

                    # Auto-assign latency sensitivity from profiler data
                    # Only if prediction has it AND no manual @latency_sensitive decorator is active
                    if prediction.latency_sensitivity is not None:
                        try:
                            ctx = Context.get()
                            if not ctx.has_manual_latency_sensitivity:
                                latency_sensitivity = prediction.latency_sensitivity
                        except Exception:
                            pass

                    logger.debug(
                        "Overriding hints from prediction: path=%s, call_index=%d, "
                        "total_requests=%d, osl_raw=%d, iat_raw=%d, latency_sensitivity=%s",
                        path,
                        call_index,
                        total_requests,
                        osl_raw,
                        iat_raw,
                        latency_sensitivity,
                    )
                else:
                    logger.debug(
                        "No prediction found for path=%s, call_index=%d; using static values",
                        path,
                        call_index,
                    )

            except Exception:
                logger.exception("Failed to lookup prediction")

        headers = dict(request.headers)

        # Modify body to inject nvext.agent_hints (if JSON POST request).
        #
        # All routing hints live in a single nvext.agent_hints dict:
        #   Standard Dynamo AgentHints fields (dynamo/lib/llm/src/protocols/openai/nvext.rs):
        #     latency_sensitivity  — queue ordering in Dynamo's built-in router
        #     osl                  — output token hint for resource estimation (u32 integer)
        #     priority             — engine scheduler priority (vLLM: lower=higher; SGLang: configurable)
        #   Custom processor.py fields:
        #     prefix_id            — KV cache prefix identity for worker stickiness
        #     total_requests       — expected session length for reuse_budget computation
        #     iat                  — inter-arrival time in ms (always raw integer)
        content = request.content
        if request.method == "POST" and content:
            try:
                body = json.loads(content.decode("utf-8", errors="replace"))
                if isinstance(body, dict):
                    # ---- Validate all agent_hints fields before injection ----
                    #
                    # Config-level Pydantic validation covers static values for osl, iat, and
                    # total_requests. Prediction trie overrides bypass Pydantic, so we guard
                    # those here too. latency_sensitivity comes entirely from Context (not a
                    # config field) so it is only validated here.

                    # total_requests must be a positive integer.
                    if total_requests < 1:
                        raise ValueError(f"total_requests must be >= 1, got {total_requests}")

                    # osl_raw must be a positive integer (Dynamo AgentHints.osl is u32).
                    if osl_raw < 1:
                        raise ValueError(f"osl must be >= 1, got {osl_raw}")

                    # iat_raw must be positive (used as TTL denominator and router weight).
                    if iat_raw < 1:
                        raise ValueError(f"iat must be >= 1, got {iat_raw}")

                    # latency_sensitivity must be in [0, max_sensitivity].
                    if latency_sensitivity < 0:
                        raise ValueError(f"latency_sensitivity ({latency_sensitivity}) must be >= 0")
                    if latency_sensitivity > self._max_sensitivity:
                        raise ValueError(f"latency_sensitivity ({latency_sensitivity}) exceeds "
                                         f"max_sensitivity ({self._max_sensitivity}). "
                                         f"Increase max_sensitivity or lower latency_sensitivity.")

                    # priority is fully derived from validated inputs — no separate check needed.
                    # (lower number = higher priority for vLLM; SGLang is configurable)
                    priority = self._max_sensitivity - latency_sensitivity

                    if "nvext" not in body:
                        body["nvext"] = {}
                    if not isinstance(body["nvext"], dict):
                        body["nvext"] = {}

                    agent_hints = {
                        "latency_sensitivity": float(latency_sensitivity),
                        "osl": osl_raw,
                        "priority": priority,
                        "prefix_id": prefix_id,
                        "total_requests": total_requests,
                        "iat": iat_raw,
                    }
                    existing = body["nvext"].get("agent_hints", {})
                    if not isinstance(existing, dict):
                        existing = {}
                    body["nvext"]["agent_hints"] = {**existing, **agent_hints}

                    # Commit the per-prefix counter now that the request is
                    # confirmed eligible for injection.
                    with self._call_counts_lock:
                        self._call_counts[prefix_id] = call_index

                    # Inject cache_control for KV cache lifetime management.
                    # TTL = total_requests * iat_raw (ms): estimated total conversation
                    # duration before the cache entry should auto-expire.
                    # Formatted as "<N>m" (whole minutes) or "<N>s", rounded up.
                    #
                    # When cache_control_mode is FIRST_ONLY, only inject on the
                    # first request per prefix_id — pinning the system prompt when
                    # it is first established in the KV cache.
                    should_pin = (self._cache_pin_type is not None
                                  and (self._cache_control_mode == CacheControlMode.ALWAYS or
                                       (self._cache_control_mode == CacheControlMode.FIRST_ONLY and call_index == 1)))
                    if should_pin:
                        ttl_ms = total_requests * iat_raw
                        ttl_seconds = max(1, -(-ttl_ms // 1000))  # ceil division
                        if ttl_seconds >= 60 and ttl_seconds % 60 == 0:
                            ttl_str = f"{ttl_seconds // 60}m"
                        else:
                            ttl_str = f"{ttl_seconds}s"
                        body["nvext"]["cache_control"] = {
                            "type": self._cache_pin_type.value,
                            "ttl": ttl_str,
                        }

                    content = json.dumps(body).encode("utf-8")
                    headers["content-length"] = str(len(content))

                    logger.debug("Injected nvext.agent_hints=%s (body size: %d bytes)",
                                 body["nvext"].get("agent_hints"),
                                 len(content))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.debug("Could not inject nvext.agent_hints: %s", e)

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
                     osl_raw,
                     iat_raw,
                     latency_sensitivity)

        return await self._transport.handle_async_request(new_request)

    async def aclose(self) -> None:
        """Close the underlying transport."""
        await self._transport.aclose()


# =============================================================================
# HTTPX CLIENT CREATION
# =============================================================================


@asynccontextmanager
async def _create_httpx_client_with_dynamo_hooks(config: DynamoModelConfig) -> "httpx.AsyncClient":
    """
    Create an httpx.AsyncClient, when `config.enable_nvext_hints` is True, Dynamo hint injection via custom transport
    is added.

    This client can be passed to the OpenAI SDK or wrapped in an AsyncOpenAI client
    for use with LiteLLM/ADK. All hints are injected into ``nvext.agent_hints``
    in the request body.

    Args:
        config: LLM Config

    Returns:
        An httpx.AsyncClient configured with Dynamo hint injection.
    """
    import httpx

    from nat.llm.utils.http_client import async_http_client

    http_client_kwargs = {}
    if config.enable_nvext_hints:
        from nat.profiler.prediction_trie import load_prediction_trie
        from nat.profiler.prediction_trie.trie_lookup import PredictionTrieLookup

        prediction_lookup: PredictionTrieLookup | None = None
        if config.nvext_prediction_trie_path:
            try:
                trie_path = Path(config.nvext_prediction_trie_path)
                trie = load_prediction_trie(trie_path)
                prediction_lookup = PredictionTrieLookup(trie)
                logger.info("Loaded prediction trie from %s", config.nvext_prediction_trie_path)
            except FileNotFoundError:
                logger.warning("Prediction trie file not found: %s", config.nvext_prediction_trie_path)
            except Exception:
                logger.exception("Failed to load prediction trie")

        # Create base transport and wrap with custom transport
        base_transport = httpx.AsyncHTTPTransport(verify=config.verify_ssl)
        dynamo_transport = _DynamoTransport(
            transport=base_transport,
            total_requests=config.nvext_prefix_total_requests,
            osl=config.nvext_prefix_osl,
            iat=config.nvext_prefix_iat,
            prediction_lookup=prediction_lookup,
            cache_pin_type=config.nvext_cache_pin_type,
            cache_control_mode=config.nvext_cache_control_mode,
            max_sensitivity=config.nvext_max_sensitivity,
        )

        http_client_kwargs["transport"] = dynamo_transport
        logger.info(
            "Dynamo agent hints enabled: total_requests=%d, osl=%s, iat=%s, prediction_trie=%s",
            config.nvext_prefix_total_requests,
            config.nvext_prefix_osl,
            config.nvext_prefix_iat,
            "loaded" if config.nvext_prediction_trie_path else "disabled",
        )

    async with async_http_client(llm_config=config, **http_client_kwargs) as client:
        yield client


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
        description="A Dynamo-optimized model with automatic nvext.agent_hints injection for KV cache management.",
    )
