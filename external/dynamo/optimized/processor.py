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
Optimized Processor for Thompson Sampling Router Architecture.

This processor uses the "Processor-as-Backend" pattern with DYNAMIC DISCOVERY
to intercept requests from the default Dynamo frontend and apply custom Thompson
Sampling routing.

## Dynamic Discovery Mode (Forward-Compatible)

Instead of using the deprecated `--static-endpoint` flag on the frontend, this
processor registers a model card in ETCD so the frontend can discover it via
its ModelWatcher. This is the forward-compatible approach.

### Requirements:
- Processor must be started with `--model-path` and `--model-name` arguments
- Model path must point to a valid model directory with tokenizer files
- Model name must match what the frontend expects (e.g., "llama-3.3-70b")

### Endpoint Registration Pattern

1. **This Processor registers as `dynamo.backend.generate`** - Dynamically with instance ID
2. **Processor calls `register_llm()`** - Advertises model card in ETCD
3. **Frontend's ModelWatcher discovers us** - Routes requests to our endpoint
4. **SGLang Worker registers as `dynamo.worker.generate`** - We forward to actual workers

## Request Flow

```
Frontend (discovers backends via ETCD ModelWatcher)
    → routes to dynamo.backend.generate-{instance_id}
    → THIS PROCESSOR (discovered via model card!)
        → extracts hints from nvext annotations
        → queries Thompson Sampling router → worker_id
        → forwards to dynamo.worker.generate (actual SGLang workers)
```

Key differences from generalized/processor.py:
- Uses dynamic discovery (no --static-endpoint on frontend)
- Registers model card via register_llm() for ETCD discovery
- Registers as `dynamo.backend.generate` (not `dynamo.processor.process`)
- Forwards to `dynamo.worker.generate` (not `dynamo.backend.generate`)
- Receives PreprocessedRequest instead of ChatCompletionRequest
- Extracts hints from nvext annotations (prefix_id:value format)
- Uses Prometheus metrics instead of CSV logging
- No tokenization (handled by frontend preprocessor)
"""

import argparse
import asyncio
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import uvloop
from dynamo.runtime import DistributedRuntime
from dynamo.runtime import dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.llm import ModelInput, ModelType, register_llm
from pydantic import BaseModel

# Prometheus metrics - import lazily to ensure proper multiprocess setup
_prometheus_initialized = False
_metrics = {}

configure_dynamo_logging()
logger = logging.getLogger(__name__)


def _init_prometheus_metrics():
    """Initialize Prometheus metrics lazily."""
    global _prometheus_initialized, _metrics
    if _prometheus_initialized:
        return _metrics

    try:
        from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, REGISTRY

        _metrics["requests_total"] = Counter(
            "thompson_processor_requests_total",
            "Total requests processed by the Thompson Sampling processor",
            registry=REGISTRY,
        )
        _metrics["request_latency"] = Histogram(
            "thompson_processor_request_latency_seconds",
            "Request latency in seconds",
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
            registry=REGISTRY,
        )
        _metrics["tokens_in"] = Counter(
            "thompson_processor_tokens_in_total",
            "Total input tokens processed",
            registry=REGISTRY,
        )
        _metrics["tokens_out"] = Counter(
            "thompson_processor_tokens_out_total",
            "Total output tokens generated",
            registry=REGISTRY,
        )
        _metrics["routing_decisions"] = Counter(
            "thompson_processor_routing_decisions_total",
            "Routing decisions by worker",
            ["worker_id"],
            registry=REGISTRY,
        )
        _metrics["router_errors"] = Counter( # i.e errors when picking a worker
            "thompson_processor_router_errors_total",
            "Router communication errors",
            registry=REGISTRY,
        )
        _metrics["engine_errors"] = Counter( # i.e errors when streaming from the engine
            "thompson_processor_engine_errors_total",
            "Backend engine errors",
            registry=REGISTRY,
        )
        _metrics["active_requests"] = Gauge(
            "thompson_processor_active_requests",
            "Currently active requests being processed",
            registry=REGISTRY,
        )
        _prometheus_initialized = True
        logger.info("Prometheus metrics initialized for processor")
    except ImportError:
        logger.warning("prometheus_client not available, metrics disabled")
        _prometheus_initialized = True  # Don't retry

    return _metrics


# ----------------------- request / response models ----------------------- #
class RouterRequest(BaseModel):
    """Request to the Thompson Sampling router."""
    tokens: list[int]
    prefix_id: str = "<no_reuse>"
    reuse_budget: int = 0  # remaining *after this request*
    expected_osl: str | None = "MEDIUM"
    interarrival: str | None = "MEDIUM"


class RouterFeedbackRequest(BaseModel):
    """Feedback to the router after request completion."""
    decision_id: str
    latency_ms: float
    success: bool | None = True
    tokens_in: int | None = None
    tokens_out: int | None = None
    finish_reason: str | None = None


# -------------------------- processor handler -------------------------- #
class ProcessorRequestHandler:
    """
    Processor that receives PreprocessedRequest from the default Dynamo frontend,
    extracts routing hints from nvext annotations, and coordinates with the
    Thompson Sampling router for intelligent worker selection.
    """

    def __init__(
        self,
        runtime: DistributedRuntime,
        enable_router: bool = True,
    ):
        self.runtime = runtime
        self.enable_router = enable_router

        self.router_pick_client = None
        self.router_feedback_client = None
        self.engine_client = None

        # Prefix-level state: {prefix_id: {"total": int, "processed": int}}
        self._prefix_state: dict[str, dict[str, int]] = {}
        self._prefix_lock = asyncio.Lock()

        # Prometheus metrics
        self._metrics = {}

    async def initialize(self):
        """Initialize processor by connecting to router and backend."""
        # Initialize Prometheus metrics
        self._metrics = _init_prometheus_metrics()

        if self.enable_router:
            ns = self.runtime.namespace("dynamo").component("router")
            self.router_pick_client = await ns.endpoint("find_worker").client()
            self.router_feedback_client = await ns.endpoint("feedback").client()
            logger.info("Router clients created, waiting for instances...")
            await self.router_pick_client.wait_for_instances()
            logger.info("Router clients initialized successfully")

        # Engine client - connects to actual workers at dynamo.worker.generate
        # (We register as "backend" to intercept frontend requests, but actual SGLang
        # workers register as "worker" so we can forward to them after routing)
        self.engine_client = await self.runtime.namespace("dynamo").component("worker").endpoint("generate").client()
        logger.info("Engine client created, waiting for worker instances...")
        await self.engine_client.wait_for_instances()
        logger.info("Processor initialized successfully (routing to dynamo.worker.generate)")

    # ---- annotation extraction ----
    @staticmethod
    def _extract_annotation(annotations: list[str], key: str, default: str | None = None) -> str | None:
        """Extract value from annotations list (format: 'key:value')."""
        prefix = f"{key}:"
        for ann in annotations:
            if ann.startswith(prefix):
                return ann[len(prefix):]
        return default

    def _extract_hints(self, request: dict[str, Any]) -> tuple[str, int, str, str]:
        """
        Extract routing hints from PreprocessedRequest annotations.

        Returns: (prefix_id, total_requests, osl, iat)
        """
        annotations = request.get("annotations", [])
        if not isinstance(annotations, list):
            annotations = []

        # Extract from annotations
        prefix_id = self._extract_annotation(annotations, "prefix_id")
        if not prefix_id:
            prefix_id = f"auto-{uuid.uuid4().hex}"

        total_str = self._extract_annotation(annotations, "total_requests", "1")
        try:
            total_requests = max(1, int(total_str))
        except (ValueError, TypeError):
            total_requests = 1

        osl = self._extract_annotation(annotations, "osl", "MEDIUM")
        osl = osl.upper() if osl else "MEDIUM"
        if osl not in ("LOW", "MEDIUM", "HIGH"):
            osl = "MEDIUM"

        iat = self._extract_annotation(annotations, "iat", "MEDIUM")
        iat = iat.upper() if iat else "MEDIUM"
        if iat not in ("LOW", "MEDIUM", "HIGH"):
            iat = "MEDIUM"

        return prefix_id, total_requests, osl, iat

    async def _update_prefix_state(self, prefix_id: str, total_requests: int) -> int:
        """
        Updates prefix counters and returns remaining_after (reuse_budget).
        """
        async with self._prefix_lock:
            s = self._prefix_state.get(prefix_id)
            if s is None:
                s = {"total": total_requests, "processed": 0}
                self._prefix_state[prefix_id] = s
            else:
                s["total"] = max(s["total"], total_requests)

            s["processed"] += 1
            remaining_after = max(s["total"] - s["processed"], 0)

            if remaining_after == 0:
                # Drop state immediately when finished
                self._prefix_state.pop(prefix_id, None)

        return remaining_after

    async def _pick_worker(
        self,
        token_ids: list[int],
        prefix_id: str,
        reuse_budget: int,
        osl: str,
        iat: str,
    ) -> tuple[int | None, str | None]:
        """Pick a worker via the router."""
        if not self.router_pick_client:
            return None, None

        req = RouterRequest(
            tokens=token_ids,
            prefix_id=prefix_id,
            reuse_budget=max(int(reuse_budget), 0),
            expected_osl=osl,
            interarrival=iat,
        )
        try:
            stream = await self.router_pick_client.generate(req.model_dump())

            worker_id: int | None = None
            decision_id: str | None = None
            async for chunk in stream:
                data = chunk.data()
                if "error" in data:
                    logger.error("Router error: %s", data["error"])
                    if self._metrics.get("router_errors"):
                        self._metrics["router_errors"].inc()
                    break
                wid = data.get("worker_id", -1)
                if wid == -1:
                    break
                worker_id = int(wid)
                decision_id = data.get("decision_id")
                break

            if worker_id is not None and self._metrics.get("routing_decisions"):
                self._metrics["routing_decisions"].labels(worker_id=str(worker_id)).inc()

            if worker_id is None:
                logger.warning("Router stream ended without worker_id; falling back to engine load balancing.")

            return worker_id, decision_id

        except Exception as e:
            logger.error("Failed to pick worker: %s", e)
            if self._metrics.get("router_errors"):
                self._metrics["router_errors"].inc()
            return None, None

    async def _send_feedback_safely(
        self,
        decision_id: str | None,
        latency_ms: float,
        success: bool,
        tokens_in: int,
        tokens_out: int,
        finish_reason: str | None,
    ):
        """Send feedback to router (fire-and-forget style)."""
        if not decision_id or not self.router_feedback_client:
            return
        try:
            fb = RouterFeedbackRequest(
                decision_id=decision_id,
                latency_ms=float(latency_ms),
                success=bool(success),
                tokens_in=int(tokens_in),
                tokens_out=int(tokens_out),
                finish_reason=finish_reason or "",
            )
            stream = await self.router_feedback_client.generate(fb.model_dump())
            async for _ in stream:
                pass
        except Exception:
            logger.exception("Failed to send router feedback")

    async def _stream_from_engine(
        self,
        request: dict[str, Any],
        worker_id: int | None,
        decision_id: str | None,
        tokens_in: int,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream response from the backend engine.
        Yields response chunks and sends feedback on completion.
        """
        t0 = time.perf_counter()
        tokens_out = 0
        finish_reason: str | None = None

        try:
            # Route to specific worker or use engine's load balancing
            if worker_id is not None:
                stream = await self.engine_client.direct(request, worker_id)
            else:
                stream = await self.engine_client.generate(request)

            async for chunk in stream:
                data = chunk.data()

                if "error" in data:
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    await self._send_feedback_safely(
                        decision_id, latency_ms, False, tokens_in, tokens_out, "error"
                    )
                    if self._metrics.get("engine_errors"):
                        self._metrics["engine_errors"].inc()
                    yield {"error": data["error"]}
                    return

                # Count output tokens
                if "token_ids" in data and isinstance(data["token_ids"], list):
                    tokens_out += len(data["token_ids"])

                # Pass through the chunk
                yield data

                if "finish_reason" in data and data["finish_reason"] is not None:
                    finish_reason = data["finish_reason"]
                    latency_ms = (time.perf_counter() - t0) * 1000.0

                    # Send feedback
                    await self._send_feedback_safely(
                        decision_id, latency_ms, True, tokens_in, tokens_out, finish_reason
                    )

                    # Update metrics
                    if self._metrics.get("request_latency"):
                        self._metrics["request_latency"].observe(latency_ms / 1000.0)
                    if self._metrics.get("tokens_in"):
                        self._metrics["tokens_in"].inc(tokens_in)
                    if self._metrics.get("tokens_out"):
                        self._metrics["tokens_out"].inc(tokens_out)

                    return

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            await self._send_feedback_safely(
                decision_id, latency_ms, False, tokens_in, tokens_out, "exception"
            )
            if self._metrics.get("engine_errors"):
                self._metrics["engine_errors"].inc()
            logger.exception("Engine stream exception")
            yield {"error": str(e)}
            return

    # ---- main generation ----
    async def generate(self, raw: dict[str, Any]):
        """
        Processor endpoint: receives PreprocessedRequest from frontend.

        Expected format (from Dynamo preprocessor):
        {
            "token_ids": [...],
            "annotations": ["prefix_id:xyz", "total_requests:10", ...],
            "sampling_options": {...},
            "stop_conditions": {...},
            ...
        }
        """
        # Track active requests
        if self._metrics.get("active_requests"):
            self._metrics["active_requests"].inc()

        try:
            # Increment request counter
            if self._metrics.get("requests_total"):
                self._metrics["requests_total"].inc()

            # Extract routing hints from annotations
            prefix_id, total_requests, osl, iat = self._extract_hints(raw)

            # Get token IDs from preprocessed request
            token_ids = raw.get("token_ids", [])
            if not isinstance(token_ids, list):
                token_ids = []

            tokens_in = len(token_ids)
            logger.info(
                "Processing request: prefix=%s total=%d osl=%s iat=%s tokens=%d",
                prefix_id, total_requests, osl, iat, tokens_in
            )

            # Compute reuse_budget := remaining AFTER this request
            reuse_budget = await self._update_prefix_state(prefix_id, total_requests)

            # Pick worker via router
            worker_id, decision_id = await self._pick_worker(
                token_ids, prefix_id, reuse_budget, osl, iat
            )

            logger.info(
                "Routing decision: worker=%s decision=%s reuse_budget=%d",
                worker_id, decision_id, reuse_budget
            )

            # Stream from engine
            async for resp in self._stream_from_engine(raw, worker_id, decision_id, tokens_in):
                yield resp

        finally:
            if self._metrics.get("active_requests"):
                self._metrics["active_requests"].dec()


# -------------------------- worker entry point -------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Optimized Thompson Sampling Processor")
    p.add_argument(
        "--enable-router",
        action="store_true",
        default=True,
        help="Enable Thompson Sampling router integration",
    )
    p.add_argument(
        "--no-router",
        action="store_false",
        dest="enable_router",
        help="Disable router (use engine load balancing only)",
    )
    p.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model directory (for loading tokenizer and model card)",
    )
    p.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Served model name (must match frontend's --model-name)",
    )
    return p.parse_args()


@dynamo_worker(static=False)  # Dynamic mode for ETCD discovery by frontend
async def worker(runtime: DistributedRuntime):
    args = parse_args()

    # DYNAMIC DISCOVERY MODE:
    # Instead of using --static-endpoint on the frontend, we register a model card
    # in ETCD so the frontend can discover us via its ModelWatcher.
    #
    # This is the forward-compatible approach since --static-endpoint is deprecated.
    #
    # Flow:
    #   1. We register as dynamo.backend.generate (dynamically with instance ID)
    #   2. We call register_llm() to advertise ourselves in ETCD
    #   3. Frontend's ModelWatcher discovers us and routes requests to us
    #   4. We forward to actual workers at dynamo.worker.generate

    component = runtime.namespace("dynamo").component("backend")
    await component.create_service()

    # Create the endpoint FIRST (needed for register_llm)
    endpoint = component.endpoint("generate")

    # Register the model card with ETCD so the frontend can discover us
    # We accept preprocessed tokens (ModelInput.Tokens) and serve chat/completions
    logger.info(f"Registering model card: model_name={args.model_name}, model_path={args.model_path}")
    await register_llm(
        model_input=ModelInput.Tokens,  # We accept tokenized input from frontend
        model_type=ModelType.Chat | ModelType.Completions,  # Chat and completions endpoints
        endpoint=endpoint,
        model_path=args.model_path,
        model_name=args.model_name,
    )
    logger.info("Model card registered successfully - frontend can now discover us via ETCD")

    # Initialize the request handler
    # Note: We use the same runtime for both serving AND client connections now,
    # since we're fully dynamic. The runtime will discover workers dynamically.
    handler = ProcessorRequestHandler(runtime, enable_router=args.enable_router)
    await handler.initialize()

    # Serve as "backend.generate" - frontend will route to us after ETCD discovery
    await endpoint.serve_endpoint(handler.generate)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())  # pylint: disable=no-value-for-parameter
