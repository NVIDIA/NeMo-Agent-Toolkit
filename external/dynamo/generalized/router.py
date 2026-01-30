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

import argparse
import asyncio
import logging
import math
import os
import random
import threading
import time
import uuid
from typing import Any

import uvloop
from dynamo.runtime import DistributedRuntime
from dynamo.runtime import dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging
from pydantic import BaseModel
from pydantic import field_validator

configure_dynamo_logging()
logger = logging.getLogger(__name__)

WorkerId = int


# ---------------------- request / response models ---------------------- #
class RouterRequest(BaseModel):
    tokens: list[int]
    session_id: str
    latency_priority: str  # LOW | MEDIUM | HIGH

    @field_validator('latency_priority')
    @classmethod
    def validate_priority(cls, v: str) -> str:
        """Validate and normalize priority to LOW/MEDIUM/HIGH."""
        if not v:
            return "MEDIUM"
        normalized = v.strip().upper()
        if normalized not in ("LOW", "MEDIUM", "HIGH"):
            raise ValueError(f"latency_priority must be LOW/MEDIUM/HIGH, got: {v}")
        return normalized


class RouterResponse(BaseModel):
    worker_id: int
    prefix_hit_rate: float  # For compatibility; always 0.0 in simplified version
    decision_id: str | None = None


class FeedbackRequest(BaseModel):
    decision_id: str
    latency_ms: float
    success: bool | None = True
    tokens_in: int | None = None
    tokens_out: int | None = None
    finish_reason: str | None = None


class FeedbackAck(BaseModel):
    ok: bool
    used_baseline: float
    reward: float
    worker_id: int | None = None
    error: str | None = None


# ---------------------- simplified router implementation ---------------------- #
class SimplifiedRouter:
    """
    Priority-aware router with session-based colocation:
      - Reserves capacity on workers for HIGH priority requests
      - Routes based on in-flight request counts and priority
      - Maintains session-to-worker assignments for KV cache reuse
      - Handles feedback endpoint for compatibility (no learning)
    """

    def __init__(
        self,
        runtime: DistributedRuntime,
        min_workers: int = 1,
        high_capacity_ratio: float = 0.3,
        feedback_timeout_seconds: float = 120.0,
        pending_sweep_interval_seconds: float = 5.0,
    ):
        self.runtime = runtime
        self.min_workers = min_workers
        self.high_capacity_ratio = float(high_capacity_ratio)
        self.feedback_timeout_seconds = float(feedback_timeout_seconds)
        self.pending_sweep_interval_seconds = float(pending_sweep_interval_seconds)

        # Clients / helpers (initialized later)
        self.engine_client = None

        # Concurrency primitives
        self._lock = threading.Lock()
        self._pending_lock = threading.Lock()

        # Worker groupings by priority (computed during initialization)
        self.high_priority_workers: set[int] = set()
        self.low_priority_workers: set[int] = set()
        self.flexible_workers: set[int] = set()  # Can serve MEDIUM or overflow

        # Per-worker in-flight request tracking: worker_id -> {priority: count}
        self.worker_load: dict[int, dict[str, int]] = {}

        # Session-to-worker assignment for colocation: session_id -> worker_id
        self.session_assignments: dict[str, int] = {}

        # Pending decisions for feedback tracking: decision_id -> metadata
        self.pending_decisions: dict[str, dict[str, Any]] = {}
        self._last_pending_sweep = 0.0

    # --------------------- initialization --------------------- #
    async def initialize(self):
        """Initialize router by discovering backend workers and partitioning them."""
        engine = self.runtime.namespace("dynamo").component("backend")
        logger.info("Getting engine client for dynamo/backend/generate")
        self.engine_client = await engine.endpoint("generate").client()

        min_workers = int(self.min_workers)
        if min_workers < 0:
            raise ValueError(f"min_workers must be >= 0, got {min_workers}")

        timeout_s = float(os.environ.get("DYNAMO_ROUTER_WAIT_FOR_WORKERS_TIMEOUT_S", "600"))
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("DYNAMO_ROUTER_WAIT_FOR_WORKERS_TIMEOUT_S must be a finite number > 0 "
                             f"(got {timeout_s!r})")
        deadline = time.monotonic() + timeout_s
        backoff_s = 0.5

        logger.info(
            "Waiting for backend workers (min_workers=%d, timeout_s=%.1f)...",
            min_workers,
            timeout_s,
        )
        if min_workers == 0:
            instance_ids_raw = list(self.engine_client.instance_ids())
            logger.info("Backend workers discovered (min_workers=0): %s", instance_ids_raw)
        else:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out after {timeout_s}s waiting for >= {min_workers} backend worker(s) to register")

                try:
                    await asyncio.wait_for(
                        self.engine_client.wait_for_instances(),
                        timeout=min(remaining, 10.0),
                    )
                except TimeoutError:
                    pass

                instance_ids_raw = list(self.engine_client.instance_ids())
                if len(instance_ids_raw) >= min_workers:
                    try:
                        instance_ids = [int(w) for w in instance_ids_raw]
                    except Exception:
                        instance_ids = instance_ids_raw
                    logger.info("Backend workers discovered: %s", instance_ids)
                    break

                await asyncio.sleep(backoff_s)
                backoff_s = min(backoff_s * 1.5, 5.0)

        self._partition_workers()
        logger.info(
            "SimplifiedRouter initialized: high_priority=%s, low_priority=%s, flexible=%s",
            sorted(self.high_priority_workers),
            sorted(self.low_priority_workers),
            sorted(self.flexible_workers),
        )

    def _partition_workers(self):
        """Partition workers into priority groups based on capacity ratio."""
        if self.engine_client is None:
            raise RuntimeError("Engine client not initialized")

        worker_ids = sorted([int(w) for w in self.engine_client.instance_ids()])
        num_workers = len(worker_ids)

        if num_workers == 0:
            return

        with self._lock:
            # Initialize load tracking
            for wid in worker_ids:
                self.worker_load[wid] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

            if num_workers == 1:
                # Single worker handles all priorities
                self.flexible_workers = set(worker_ids)
                logger.info("Single worker mode: worker %d handles all priorities", worker_ids[0])
            elif num_workers == 2:
                # Two workers: one for HIGH, one for LOW/MEDIUM
                self.high_priority_workers = {worker_ids[0]}
                self.low_priority_workers = {worker_ids[1]}
                logger.info("Two worker mode: HIGH=%s, LOW/MEDIUM=%s", worker_ids[0], worker_ids[1])
            else:
                # Multiple workers: partition based on high_capacity_ratio
                num_high = max(1, int(num_workers * self.high_capacity_ratio))
                self.high_priority_workers = set(worker_ids[:num_high])
                self.low_priority_workers = set(worker_ids[num_high:])
                # Flexible workers can handle overflow from either side
                # Use middle workers as flexible if we have enough
                if num_workers >= 4:
                    mid_start = num_high
                    mid_end = mid_start + max(1, num_workers // 4)
                    self.flexible_workers = set(worker_ids[mid_start:mid_end])
                    self.low_priority_workers -= self.flexible_workers
                logger.info(
                    "Multi-worker mode: HIGH=%s, LOW=%s, FLEXIBLE=%s",
                    sorted(self.high_priority_workers),
                    sorted(self.low_priority_workers),
                    sorted(self.flexible_workers),
                )

    # --------------------- helper methods --------------------- #
    @staticmethod
    def _norm_priority(s: str | None) -> str:
        """Normalize priority string to HIGH/MEDIUM/LOW."""
        if not s:
            return "MEDIUM"
        s = str(s).strip().upper()
        return s if s in ("LOW", "MEDIUM", "HIGH") else "MEDIUM"

    def _get_worker_candidates(self, priority: str) -> list[int]:
        """Get list of workers that can serve the given priority."""
        if priority == "HIGH":
            candidates = list(self.high_priority_workers | self.flexible_workers)
        elif priority == "LOW":
            candidates = list(self.low_priority_workers | self.flexible_workers)
        else:  # MEDIUM
            candidates = list(self.high_priority_workers | self.low_priority_workers | self.flexible_workers)

        # If no candidates (shouldn't happen), use all workers
        if not candidates:
            candidates = list(self.worker_load.keys())

        return candidates

    def _get_worker_total_load(self, wid: int) -> int:
        """Get total in-flight requests for a worker across all priorities."""
        with self._lock:
            load = self.worker_load.get(wid, {})
            return sum(load.values())

    def _increment_load(self, wid: int, priority: str):
        """Increment in-flight request count for worker and priority."""
        with self._lock:
            if wid not in self.worker_load:
                self.worker_load[wid] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
            self.worker_load[wid][priority] += 1

    def _decrement_load(self, wid: int, priority: str):
        """Decrement in-flight request count for worker and priority."""
        with self._lock:
            if wid in self.worker_load and priority in self.worker_load[wid]:
                self.worker_load[wid][priority] = max(0, self.worker_load[wid][priority] - 1)

    def _select_worker(self, session_id: str, priority: str) -> int:
        """
        Select worker based on:
        1. Session assignment (if exists and worker has capacity)
        2. Priority group membership
        3. Least loaded worker
        """
        # Check if session already has an assigned worker
        with self._lock:
            assigned_worker = self.session_assignments.get(session_id)

        if assigned_worker is not None:
            # Verify worker still exists and has reasonable capacity
            if assigned_worker in self.worker_load:
                load = self._get_worker_total_load(assigned_worker)
                # Allow some reasonable max load per worker (e.g., 10 concurrent requests)
                # This prevents sticking to overloaded workers
                max_load_threshold = 10
                if load < max_load_threshold:
                    logger.debug(
                        "Reusing worker %d for session %s (load=%d)",
                        assigned_worker,
                        session_id,
                        load,
                    )
                    return assigned_worker
                else:
                    logger.debug(
                        "Session %s worker %d overloaded (load=%d), selecting new worker",
                        session_id,
                        assigned_worker,
                        load,
                    )

        # Select new worker based on priority and load
        candidates = self._get_worker_candidates(priority)

        # Find least loaded worker among candidates
        with self._lock:
            loads = [(wid, self._get_worker_total_load(wid)) for wid in candidates]

        if not loads:
            # Shouldn't happen, but fallback to random worker
            all_workers = list(self.worker_load.keys())
            chosen = random.choice(all_workers) if all_workers else 0
            logger.warning("No worker candidates found, falling back to worker %d", chosen)
            return chosen

        # Sort by load (ascending) and break ties randomly
        random.shuffle(loads)  # Randomize first for tie-breaking
        loads.sort(key=lambda x: x[1])
        chosen = loads[0][0]

        # Assign session to this worker
        with self._lock:
            self.session_assignments[session_id] = chosen

        logger.debug(
            "Selected worker %d for session %s (priority=%s, load=%d)",
            chosen,
            session_id,
            priority,
            loads[0][1],
        )

        return chosen

    def _sweep_pending(self, now: float):
        """Clean up pending decisions that have timed out."""
        if now - self._last_pending_sweep < self.pending_sweep_interval_seconds:
            return
        self._last_pending_sweep = now

        expired: list[tuple[str, dict[str, Any]]] = []
        with self._pending_lock:
            for decision_id, rec in list(self.pending_decisions.items()):
                if now - float(rec.get("start_ts", now)) >= self.feedback_timeout_seconds:
                    expired.append((decision_id, rec))
                    self.pending_decisions.pop(decision_id, None)

        for decision_id, rec in expired:
            wid = int(rec["wid"])
            priority = str(rec["priority"])
            # Decrement load since we're not expecting feedback
            self._decrement_load(wid, priority)
            logger.warning(
                "Timeout feedback: decision=%s wid=%s priority=%s session=%s",
                decision_id,
                wid,
                priority,
                rec.get("session_id"),
            )

    # --------------------- main endpoint: find_worker --------------------- #
    async def generate(self, request: dict):
        """Router endpoint that selects a worker for the request."""
        req = RouterRequest(**request)

        worker_ids = [int(w) for w in self.engine_client.instance_ids()]
        if not worker_ids:
            yield RouterResponse(worker_id=-1, prefix_hit_rate=0.0).model_dump()
            return

        now = time.time()
        self._sweep_pending(now)

        priority = self._norm_priority(req.latency_priority)
        chosen = self._select_worker(req.session_id, priority)

        # Track decision
        decision_id = uuid.uuid4().hex
        self._increment_load(chosen, priority)

        with self._pending_lock:
            self.pending_decisions[decision_id] = {
                "wid": chosen,
                "priority": priority,
                "session_id": req.session_id,
                "start_ts": now,
                "tokens_in": len(req.tokens),
            }

        logger.info(
            "Router picked worker=%s decision=%s session=%s priority=%s tokens=%d",
            chosen,
            decision_id,
            req.session_id,
            priority,
            len(req.tokens),
        )

        resp = RouterResponse(worker_id=chosen, prefix_hit_rate=0.0, decision_id=decision_id)
        yield resp.model_dump()

    # --------------------- feedback endpoint --------------------- #
    async def feedback(self, request: dict):
        """Feedback endpoint for compatibility (no learning, just load tracking)."""
        try:
            fb = FeedbackRequest(**request)
        except Exception as e:
            ack = FeedbackAck(ok=False, used_baseline=0.0, reward=0.0, error=str(e))
            yield ack.model_dump()
            return

        with self._pending_lock:
            decision = self.pending_decisions.pop(fb.decision_id, None)

        if not decision:
            ack = FeedbackAck(ok=False, used_baseline=0.0, reward=0.0, error="unknown_decision")
            yield ack.model_dump()
            return

        wid = int(decision["wid"])
        priority = str(decision["priority"])

        # Decrement load tracking
        self._decrement_load(wid, priority)

        logger.info(
            "Feedback received: wid=%s decision=%s latency=%.3fms success=%s priority=%s",
            wid,
            fb.decision_id,
            fb.latency_ms,
            fb.success,
            priority,
        )

        # Return success (no learning, just acknowledgment)
        ack = FeedbackAck(ok=True, used_baseline=0.0, reward=1.0, worker_id=wid)
        yield ack.model_dump()


# ---------------------- worker entry point ---------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Simplified priority-aware router")
    parser.add_argument("--min-workers", type=int, default=1)
    parser.add_argument("--high-capacity-ratio", type=float, default=0.3,
                        help="Fraction of workers reserved for HIGH priority (0.0-1.0)")
    parser.add_argument("--feedback-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--pending-sweep-interval-seconds", type=float, default=5.0)
    return parser.parse_args()


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    args = parse_args()

    component = runtime.namespace("dynamo").component("router")
    await component.create_service()
    logger.info("Initializing SimplifiedRouter")

    router = SimplifiedRouter(
        runtime,
        min_workers=args.min_workers,
        high_capacity_ratio=args.high_capacity_ratio,
        feedback_timeout_seconds=args.feedback_timeout_seconds,
        pending_sweep_interval_seconds=args.pending_sweep_interval_seconds,
    )
    await router.initialize()

    # Serve both endpoints (find_worker for selection, feedback for compatibility)
    await asyncio.gather(
        component.endpoint("find_worker").serve_endpoint(router.generate),
        component.endpoint("feedback").serve_endpoint(router.feedback),
    )


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
