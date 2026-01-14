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
import csv
import json
import logging
import math
import os
import random
import threading
import time
import uuid
from collections import deque
from functools import wraps
from typing import Any

import numpy as np
import uvloop
from dynamo.runtime import DistributedRuntime
from dynamo.runtime import dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging
from pydantic import BaseModel

# Try to import KV routing classes from dynamo.llm, fallback to stubs if unavailable
try:
    from dynamo.llm import KvIndexer
    from dynamo.llm import OverlapScores
except ImportError:
    logger_init = logging.getLogger(__name__)
    logger_init.warning("dynamo.llm KV classes not available, using fallback implementations")

    class OverlapScores:
        """Fallback: KV cache overlap scores between a request and workers.

        This fallback is used when `dynamo.llm` is unavailable. It always returns empty
        scores, causing the router to fall back to round-robin selection without
        considering KV cache overlap.
        """

        def __init__(self, scores: dict[int, float] | None = None):
            self.scores = scores if scores is not None else {}

    class KvIndexer:
        """Fallback: KV cache indexer for finding overlap between requests and workers.

        This fallback is used when `dynamo.llm` is unavailable. The
        `find_matches_for_request` method always returns empty overlap scores,
        effectively disabling KV-aware routing.
        """

        def __init__(self, engine: Any, block_size: int):
            self.engine = engine
            self.block_size = block_size

        async def find_matches_for_request(self, tokens: list[int], min_overlap: int) -> OverlapScores:  # noqa: ARG002
            """Find overlap scores for each worker. Returns empty scores (round-robin fallback)."""
            return OverlapScores({})


configure_dynamo_logging()
logger = logging.getLogger(__name__)

WorkerId = int


# ---------------------- request / response models ---------------------- #
class RouterRequest(BaseModel):
    tokens: list[int]
    prefix_id: str = "<no_reuse>"
    reuse_budget: int = 0  # remaining *after this request*
    expected_osl: str | None = "MEDIUM"
    interarrival: str | None = "MEDIUM"


class RouterResponse(BaseModel):
    worker_id: int
    prefix_hit_rate: float
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


# ---------------------- helper decorator ---------------------- #
def safe_update(lock_name: str):

    def decorator(fn):

        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            lock = getattr(self, lock_name)
            with lock:
                return fn(self, *args, **kwargs)

        return wrapper

    return decorator


# ---------------------- router implementation ---------------------- #
class WorkloadAwareRouter:
    """
    Contextual Thompson Sampling router with:
      - KV overlap locality
      - Remaining per-prefix requests (reuse_budget)
      - OSL-based decode cost, ISL/prefill cost per worker
      - IAT-based stickiness/opportunity weighting
      - Instant & outstanding load (no TTL decay)
      - **Delayed bandit update using observed latency via `feedback` endpoint**
      - **Timeout penalty** for missing feedback
      - **Debug traces** for offline analysis
    """

    def __init__(
        self,
        runtime: DistributedRuntime,
        block_size: int = 64,
        router_type: str = "kv",
        min_workers: int = 1,
        # Affinity / exploration
        affinity_base: float = 0.30,
        affinity_reuse_weight: float = 0.15,
        affinity_iat_weight: float = 0.20,
        base_ts_weight: float = 0.10,
        sticky_load_floor: float = 0.70,
        # Softmax temperature
        temp_base: float = 1.0,
        temp_min: float = 0.15,
        temp_max: float = 2.0,
        # Switching cost
        switch_cost_base: float = 0.20,
        switch_cost_reuse: float = 0.08,
        switch_cost_iat: float = 0.05,
        # Load / opportunity cost
        queue_penalty_weight: float = 0.50,
        gpu_penalty_weight: float = 1.00,
        outstanding_work_weight: float = 0.45,
        job_gpu_coupling_weight: float = 0.40,
        job_queue_coupling_weight: float = 0.20,
        # Prefill / ISL
        prefill_token_scale: float = 1024.0,
        prefill_weight: float = 1.0,
        # LinTS
        lints_lambda: float = 1.0,
        lints_v: float = 0.25,
        lints_forget: float = 0.995,
        # ---------- Feedback timeout / sweep ----------
        feedback_timeout_seconds: float = 120.0,  # if no feedback by this time -> penalty
        pending_sweep_interval_seconds: float = 5.0,  # how often to sweep pending
        timeout_reward: float = 0.0,  # small penalty (0..1); 0.0 is harsh
        # ---------- Latency EMA (reward normalization) ----------
        latency_ema_alpha: float = 0.2,
        # ---------- Debug traces ----------
        debug_traces: bool = False,
        debug_trace_dir: str = "/tmp/dynamo_router_traces",
        debug_buffer_size: int = 2000,
    ):
        self.runtime = runtime
        self.block_size = block_size
        self.router_type = router_type
        self.min_workers = min_workers

        # clients / helpers (initialized later)
        self.engine_client = None
        self.indexer: KvIndexer | None = None

        # concurrency primitives
        self._init_lock = threading.Lock()
        self._bandit_lock = threading.Lock()
        self._prefix_lock = threading.Lock()
        self._lin_lock = threading.Lock()
        self._pending_lock = threading.Lock()

        # prefix state: pid -> {"worker": int|None, "reuse_remaining": int}
        self.prefix_cache_state: dict[str, dict[str, int | None]] = {}
        # pid -> {"decode_cost","prefill_cost","iat_factor"}
        self.prefix_meta: dict[str, dict[str, float]] = {}

        # Beta bandits and LinTS params
        self.worker_bandits: dict[int, tuple[float, float]] = {}
        self.feature_dim = 9
        self.lin_lambda = float(lints_lambda)
        self.lin_v = float(lints_v)
        self.lin_forget = float(lints_forget)
        self.lin_forget = max(1e-6, min(self.lin_forget, 0.999999))
        self.linA: dict[int, np.ndarray] = {}
        self.linb: dict[int, np.ndarray] = {}

        # knobs
        self.affinity_base = float(affinity_base)
        self.affinity_reuse_weight = float(affinity_reuse_weight)
        self.affinity_iat_weight = float(affinity_iat_weight)
        self.base_ts_weight = float(base_ts_weight)
        self.sticky_load_floor = float(sticky_load_floor)
        self.temp_base = float(temp_base)
        self.temp_min = float(temp_min)
        self.temp_max = float(temp_max)
        self.switch_cost_base = float(switch_cost_base)
        self.switch_cost_reuse = float(switch_cost_reuse)
        self.switch_cost_iat = float(switch_cost_iat)
        self.queue_penalty_weight = float(queue_penalty_weight)
        self.gpu_penalty_weight = float(gpu_penalty_weight)
        self.outstanding_work_weight = float(outstanding_work_weight)
        self.job_gpu_coupling_weight = float(job_gpu_coupling_weight)
        self.job_queue_coupling_weight = float(job_queue_coupling_weight)
        self.prefill_token_scale = float(prefill_token_scale)
        self.prefill_weight = float(prefill_weight)

        # LinTS numerics
        self._jt_base = 1e-9
        self._jt_mult = 10.0
        self._jt_max = 1e-3
        self._eig_floor = 1e-10

        # Feedback timeout / sweep
        self.feedback_timeout_seconds = float(feedback_timeout_seconds)
        self.pending_sweep_interval_seconds = float(pending_sweep_interval_seconds)
        self.timeout_reward = float(max(0.0, min(1.0, timeout_reward)))
        self._last_pending_sweep = 0.0

        # Latency EMA baselines (two modes: raw ms, or ms/token)
        self.latency_ema_alpha = float(latency_ema_alpha)
        # Global (per-mode)
        self.lat_ema_global: dict[bool, float | None] = {False: None, True: None}
        # Per worker (per-mode)
        self.lat_ema_worker: dict[tuple[int, bool], float] = {}
        # Per bucket (per-mode): (wid, osl, prefill_bin, per_tok) -> value
        self.lat_ema_bucket: dict[tuple[int, str, str, bool], float] = {}

        # Pending decisions waiting for feedback:
        # decision_id -> {
        #   "wid": int, "x": np.ndarray, "osl": str, "prefill_bin": str,
        #   "start_ts": float, "prefix_id": str, "tokens_in": int, "reuse_after": int,
        #   "overlap": float, "prefill_cost": float, "decode_cost": float
        # }
        self.pending: dict[str, dict[str, Any]] = {}

        # Debug traces
        self.debug_traces = bool(debug_traces)
        self.debug_trace_dir = str(debug_trace_dir)
        self.recent_traces: deque = deque(maxlen=int(debug_buffer_size))
        if self.debug_traces:
            os.makedirs(self.debug_trace_dir, exist_ok=True)
            logger.info("Router debug traces enabled -> %s", self.debug_trace_dir)

    # --------------------- tracing --------------------- #
    def _emit_trace(self, kind: str, payload: dict[str, Any]):
        if not self.debug_traces:
            return
        item = {"ts": time.time(), "kind": kind, **payload}
        self.recent_traces.append(item)
        try:
            path = os.path.join(self.debug_trace_dir, "router_traces.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(item, separators=(",", ":")) + "\n")
        except Exception as e:
            logger.debug("Trace write failed: %s", e)

    # --------------------- level mappings --------------------- #
    @staticmethod
    def _norm_level(s: str | None, default: str = "MEDIUM") -> str:
        if not s:
            return default
        s = str(s).strip().upper()
        return s if s in ("LOW", "MEDIUM", "HIGH") else default

    @staticmethod
    def _decode_cost(osl: str) -> float:
        return {"LOW": 1.0, "MEDIUM": 2.0, "HIGH": 3.0}[osl]

    @staticmethod
    def _iat_factor(iat: str) -> float:
        return {"LOW": 1.5, "MEDIUM": 1.0, "HIGH": 0.6}[iat]

    # --------------------- init --------------------- #
    async def initialize(self):
        """Initialize router by polling for backend workers."""
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
                    # We'll re-check instance IDs and retry with backoff until the global deadline.
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

        self.indexer = KvIndexer(engine, self.block_size)

        self._initialize_bandits()
        self._initialize_contextual()
        logger.info("WorkloadAwareRouter initialized with %d backend worker(s)",
                    len(list(self.engine_client.instance_ids())))

        # Initialize router CSV logging (no cap)
        self._router_csv_lock = threading.Lock()
        self._router_csv_path = os.environ.get("ROUTER_METRICS_CSV", "router_metrics.csv")
        try:
            csv_dir = os.path.dirname(self._router_csv_path)
            if csv_dir and not os.path.exists(csv_dir):
                os.makedirs(csv_dir, exist_ok=True)
            if not os.path.exists(self._router_csv_path):
                with self._router_csv_lock:
                    with open(self._router_csv_path, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            "ts_epoch_ms",
                            "tokens_len",
                            "prefix_id",
                            "reuse_after",
                            "chosen_worker",
                            "overlap_chosen",
                            "decode_cost",
                            "prefill_cost",
                            "iat_level",
                            "stickiness",
                            "load_mod",
                        ])
        except Exception as e:
            logger.warning("Failed to initialize router CSV %s: %s", self._router_csv_path, e)

    @safe_update("_init_lock")
    def _initialize_bandits(self):
        for wid in self.engine_client.instance_ids():
            self.worker_bandits.setdefault(int(wid), (1.0, 1.0))

    @safe_update("_init_lock")
    def _initialize_contextual(self):
        for wid in self.engine_client.instance_ids():
            wid = int(wid)
            if wid not in self.linA:
                self.linA[wid] = self.lin_lambda * np.eye(self.feature_dim, dtype=np.float64)
                self.linb[wid] = np.zeros(self.feature_dim, dtype=np.float64)

    def _ensure_worker_context(self, worker_id: int):
        if worker_id not in self.linA:
            with self._lin_lock:
                if worker_id not in self.linA:
                    self.linA[worker_id] = self.lin_lambda * np.eye(self.feature_dim, dtype=np.float64)
                    self.linb[worker_id] = np.zeros(self.feature_dim, dtype=np.float64)

    # --------------------- prefix state --------------------- #
    @safe_update("_prefix_lock")
    def _get_prefix(self, pid: str) -> tuple[int | None, int]:
        info = self.prefix_cache_state.get(pid)
        if info:
            return info.get("worker"), int(info.get("reuse_remaining") or 0)
        return None, 0

    @safe_update("_prefix_lock")
    def _set_prefix(
        self,
        pid: str,
        wid: int,
        reuse_remaining: int,
        decode_cost: float,
        prefill_cost: float,
        iat_factor: float,
    ):
        """Record/refresh prefix assignment. Remove immediately if no future requests remain."""
        if reuse_remaining <= 0:
            self.prefix_cache_state.pop(pid, None)
            self.prefix_meta.pop(pid, None)
            return
        self.prefix_cache_state[pid] = {"worker": wid, "reuse_remaining": max(0, int(reuse_remaining))}
        self.prefix_meta[pid] = {
            "decode_cost": float(decode_cost),
            "prefill_cost": float(max(prefill_cost, 0.0)),
            "iat_factor": float(iat_factor),
        }

    def _worker_outstanding(self, wid: int) -> tuple[int, float]:
        """
        Returns (reuse_total, work_total) without time decay:
          Σ reuse_remaining * (decode_cost + prefill_cost) * iat_factor
        """
        reuse_total = 0
        work_total = 0.0
        for pid, info in self.prefix_cache_state.items():
            if info.get("worker") != wid:
                continue
            r = int(info.get("reuse_remaining") or 0)
            reuse_total += r
            meta = self.prefix_meta.get(pid)
            if meta:
                work_total += float(r) * (float(meta.get("decode_cost", 2.0)) +
                                          float(meta.get("prefill_cost", 0.0))) * float(meta.get("iat_factor", 1.0))
        return reuse_total, work_total

    # --------------------- bandits --------------------- #
    def _linTS_sample(self, wid: int, x: np.ndarray) -> float:
        self._ensure_worker_context(wid)
        with self._lin_lock:
            A = np.array(self.linA[wid], dtype=np.float64, copy=True)
            b = np.array(self.linb[wid], dtype=np.float64, copy=True)

        A = 0.5 * (A + A.T)
        eye = np.eye(self.feature_dim, dtype=np.float64)
        jitter = self._jt_base
        L = None
        while True:
            try:
                L = np.linalg.cholesky(A + jitter * eye)
                break
            except np.linalg.LinAlgError:
                jitter = jitter * self._jt_mult if jitter > 0 else self._jt_base
                if jitter > self._jt_max:
                    vals, vecs = np.linalg.eigh(A)
                    vals = np.maximum(vals, self._eig_floor)
                    A_inv = vecs @ (np.diag(1.0 / vals)) @ vecs.T
                    mu = A_inv @ b
                    z = np.random.normal(size=self.feature_dim)
                    noise = vecs @ (z / np.sqrt(vals))
                    theta = mu + (self.lin_v * noise)
                    return float(theta @ x)

        y = np.linalg.solve(L, b)
        mu = np.linalg.solve(L.T, y)
        z = np.random.normal(size=self.feature_dim)
        noise = np.linalg.solve(L.T, z)
        theta = mu + (self.lin_v * noise)
        return float(theta @ x)

    def _update_contextual(self, wid: int, x: np.ndarray, reward: float):
        r = float(max(0.0, min(1.0, reward)))
        with self._lin_lock:
            A = self.linA[wid]
            b = self.linb[wid]
            A *= self.lin_forget
            b *= self.lin_forget
            A += np.outer(x, x)
            ridge = (1.0 - self.lin_forget) * self.lin_lambda
            if ridge > 0.0:
                A += ridge * np.eye(self.feature_dim, dtype=np.float64)
            self.linA[wid] = 0.5 * (A + A.T)
            self.linb[wid] = b + x * r

    def _ts_sample(self, worker_id: int) -> float:
        with self._bandit_lock:
            alpha, beta = self.worker_bandits.get(worker_id, (1.0, 1.0))
        return np.random.beta(alpha, beta)

    def _update_bandit(self, worker_id: int, reward: float):
        with self._bandit_lock:
            alpha, beta = self.worker_bandits.get(worker_id, (1.0, 1.0))
            r = float(max(0.0, min(1.0, reward)))
            self.worker_bandits[worker_id] = (alpha + r, beta + 1.0 - r)

    # --------------------- features / scores --------------------- #
    def _prefill_cost_for_worker(self, tokens: list[int], overlap: float) -> float:
        isl = max(0, len(tokens))
        frac = min(max(float(overlap), 0.0), 1.0)
        uncached = max(0.0, float(isl) * (1.0 - frac))
        return (uncached / self.prefill_token_scale) * self.prefill_weight

    @staticmethod
    def _prefill_bin(prefill_cost: float) -> str:
        if prefill_cost < 0.25:
            return "LOW"
        if prefill_cost < 0.75:
            return "MEDIUM"
        return "HIGH"

    def _feature_vector(
        self,
        wid: int,
        metrics: dict[str, Any] | None,
        scores: "OverlapScores",
        last_w: int | None,
        reuse_after: int,
        decode_cost: float,
        prefill_cost: float,
        iat_factor: float,
    ) -> np.ndarray:
        gpu = 0.0
        queue = 0.0
        if metrics and isinstance(metrics, dict) and "endpoints" in metrics:
            for ep in metrics["endpoints"]:
                if ep.get("worker_id") == wid:
                    gpu = float(ep.get("gpu_cache_usage_perc", 0.0))
                    queue = float(ep.get("num_requests_waiting", 0.0))
                    break
        inv_load = 1.0 / (1.0 + self.gpu_penalty_weight * max(0.0, gpu) + self.queue_penalty_weight * max(0.0, queue))

        overlap = float(scores.scores.get(wid, 0.0))
        affinity = 1.0 if (last_w is not None and wid == last_w) else 0.0
        _, work_out = self._worker_outstanding(wid)

        decode_norm = decode_cost / 3.0
        prefill_norm = math.tanh(prefill_cost)
        iat_norm = iat_factor / 1.5
        outstanding_norm = math.tanh(0.1 * work_out)
        reuse_norm = math.tanh(0.25 * float(max(reuse_after, 0)))

        return np.array([
            1.0,
            inv_load,
            overlap,
            affinity,
            outstanding_norm,
            decode_norm,
            prefill_norm,
            iat_norm,
            reuse_norm,
        ],
                        dtype=np.float64)

    def _load_score(self, wid: int, metrics: dict[str, Any] | None, job_cost_total: float) -> float:
        gpu = 0.0
        queue = 0.0
        if metrics and isinstance(metrics, dict) and "endpoints" in metrics:
            for ep in metrics["endpoints"]:
                if ep.get("worker_id") == wid:
                    gpu = float(ep.get("gpu_cache_usage_perc", 0.0))
                    queue = float(ep.get("num_requests_waiting", 0.0))
                    break
        _, work_out = self._worker_outstanding(wid)
        penalty = (self.gpu_penalty_weight * gpu + self.queue_penalty_weight * queue +
                   self.outstanding_work_weight * max(0.0, work_out) +
                   self.job_gpu_coupling_weight * job_cost_total * gpu +
                   self.job_queue_coupling_weight * job_cost_total * queue)
        return 1.0 / (1.0 + max(0.0, penalty))

    def _softmax(self, scores: list[float], temp: float) -> list[float]:
        t = float(min(max(temp, self.temp_min), self.temp_max))
        m = float(np.max(scores))
        exps = np.exp((np.array(scores) - m) / max(1e-6, t))
        s = float(np.sum(exps))
        if s <= 0.0 or not np.isfinite(s):
            return [1.0 / len(scores)] * len(scores)
        return list((exps / s).astype(float))

    # --------------------- selection --------------------- #
    def _select_worker(
        self,
        worker_ids,
        req: RouterRequest,
        metrics: dict[str, Any] | None,
        scores: OverlapScores,
    ) -> tuple[int, dict[str, float], dict[int, dict[str, float]], list[float], list[float]]:
        osl = self._norm_level(req.expected_osl, "MEDIUM")
        iat = self._norm_level(req.interarrival, "MEDIUM")
        last_w, _ = self._get_prefix(req.prefix_id)

        reuse_after = max(int(req.reuse_budget), 0)
        decode_cost = self._decode_cost(osl)
        iat_factor = self._iat_factor(iat)

        temp = self.temp_base / (1.0 + float(reuse_after) * iat_factor)
        temp = min(max(temp, self.temp_min), self.temp_max)

        raw_scores: list[float] = []
        worker_list: list[int] = [int(w) for w in worker_ids]
        per_worker_ctx: dict[int, dict[str, float]] = {}
        load_mods: list[float] = []
        overlaps: list[float] = []

        for wid in worker_list:
            overlap = float(scores.scores.get(wid, 0.0))
            prefill_cost = self._prefill_cost_for_worker(req.tokens, overlap)
            job_cost_total = decode_cost + prefill_cost

            x = self._feature_vector(
                wid=wid,
                metrics=metrics,
                scores=scores,
                last_w=last_w,
                reuse_after=reuse_after,
                decode_cost=decode_cost,
                prefill_cost=prefill_cost,
                iat_factor=iat_factor,
            )

            val = self._linTS_sample(wid, x)
            explore_w = self.base_ts_weight / (1.0 + float(reuse_after) * iat_factor)
            val += explore_w * self._ts_sample(wid)

            if last_w == wid and (reuse_after > 0):
                val += (self.affinity_base + self.affinity_reuse_weight * float(reuse_after) +
                        self.affinity_iat_weight * iat_factor) * (0.5 + 0.5 * overlap)

            if last_w is not None and wid != last_w and (reuse_after > 0):
                val -= (self.switch_cost_base + self.switch_cost_reuse * float(reuse_after) +
                        self.switch_cost_iat * iat_factor)

            load_mod = self._load_score(wid, metrics, job_cost_total=job_cost_total)
            if last_w == wid and reuse_after > 0:
                load_mod = max(load_mod, self.sticky_load_floor)
            val *= load_mod

            if np.isnan(val) or np.isinf(val):
                val = -1e9

            raw_scores.append(float(val))
            load_mods.append(float(load_mod))
            overlaps.append(float(overlap))
            per_worker_ctx[wid] = {
                "decode_cost": decode_cost,
                "prefill_cost": prefill_cost,
                "iat_factor": iat_factor,
                "overlap": overlap,
                "reuse_after": float(reuse_after),
                "load_mod": load_mod,
            }

        probs = self._softmax(raw_scores, temp)
        r = random.random()
        cum = 0.0
        idx = 0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                idx = i
                break
        chosen = int(worker_list[idx])

        # Append decision metadata to CSV
        try:
            ts_ms = int(time.time() * 1000)
            overlap_chosen = float(scores.scores.get(chosen, 0.0))
            prefill_cost_chosen = float(per_worker_ctx[chosen]["prefill_cost"]) if chosen in per_worker_ctx else 0.0
            load_mod = float(per_worker_ctx[chosen]["load_mod"]) if chosen in per_worker_ctx else 0.0
            stickiness = 1.0 if (last_w is not None and chosen == last_w) else 0.0
            with self._router_csv_lock:
                with open(self._router_csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        ts_ms,
                        len(req.tokens) if getattr(req, "tokens", None) else 0,
                        req.prefix_id,
                        max(int(req.reuse_budget), 0),
                        chosen,
                        f"{overlap_chosen:.6f}",
                        f"{decode_cost:.6f}",
                        f"{prefill_cost_chosen:.6f}",
                        iat,
                        f"{stickiness:.3f}",
                        f"{load_mod:.6f}",
                    ])
        except Exception as e:
            logger.debug("Failed to append router CSV: %s", e)

        return chosen, per_worker_ctx[chosen], per_worker_ctx, raw_scores, probs

    # --------------------- latency baselines & reward --------------------- #
    def _ema_update(self, old: float | None, new: float) -> float:
        a = self.latency_ema_alpha
        return new if old is None else (a * new + (1.0 - a) * old)

    def _get_latency_baseline(self, wid: int, osl: str, prefill_bin: str, per_tok: bool, fallback: float) -> float:
        key_b = (wid, osl, prefill_bin, per_tok)
        key_w = (wid, per_tok)
        if key_b in self.lat_ema_bucket:
            return self.lat_ema_bucket[key_b]
        if key_w in self.lat_ema_worker:
            return self.lat_ema_worker[key_w]
        if self.lat_ema_global[per_tok] is not None:
            return self.lat_ema_global[per_tok]  # type: ignore
        return max(1.0, float(fallback))

    def _update_latency_baselines(self, wid: int, osl: str, prefill_bin: str, metric: float, per_tok: bool) -> float:
        # global
        self.lat_ema_global[per_tok] = self._ema_update(self.lat_ema_global[per_tok], metric)
        # per worker
        key_w = (wid, per_tok)
        self.lat_ema_worker[key_w] = self._ema_update(self.lat_ema_worker.get(key_w), metric)
        # per bucket
        key_b = (wid, osl, prefill_bin, per_tok)
        self.lat_ema_bucket[key_b] = self._ema_update(self.lat_ema_bucket.get(key_b), metric)
        return self.lat_ema_bucket[key_b]

    @staticmethod
    def _latency_metric(latency_ms: float, tokens_out: int | None) -> tuple[float, bool]:
        """Return (metric_value, per_tok_flag). If tokens_out>0 -> ms/token else ms."""
        if tokens_out is not None and int(tokens_out) > 0:
            return float(latency_ms) / float(max(1, int(tokens_out))), True
        return float(latency_ms), False

    @staticmethod
    def _metric_to_reward(metric: float, baseline: float, success: bool) -> float:
        if not success:
            return 0.0
        denom = max(1e-3, baseline)
        ratio = metric / denom  # <1.0 is good
        return float(1.0 / (1.0 + ratio))  # baseline -> 0.5

    # --------------------- timeout sweep --------------------- #
    def _sweep_pending(self, now: float):
        if now - self._last_pending_sweep < self.pending_sweep_interval_seconds:
            return
        self._last_pending_sweep = now
        expired: list[tuple[str, dict[str, Any]]] = []
        with self._pending_lock:
            for did, rec in list(self.pending.items()):
                if now - float(rec.get("start_ts", now)) >= self.feedback_timeout_seconds:
                    expired.append((did, rec))
                    self.pending.pop(did, None)
        for did, rec in expired:
            wid = int(rec["wid"])
            x = rec["x"]
            reward = float(self.timeout_reward)  # small penalty
            self._update_bandit(wid, reward)
            self._update_contextual(wid, x, reward)
            self._emit_trace(
                "timeout",
                {
                    "decision_id": did,
                    "wid": wid,
                    "reward": reward,
                    "age": self.feedback_timeout_seconds,
                    "prefix_id": rec.get("prefix_id"),
                    "osl": rec.get("osl"),
                    "prefill_bin": rec.get("prefill_bin"),
                })
            logger.warning("Timeout feedback: wid=%s decision=%s reward=%.3f", wid, did, reward)

    # --------------------- main endpoint: find_worker --------------------- #
    async def generate(self, request: dict):
        req = RouterRequest(**request)

        worker_ids = [int(w) for w in self.engine_client.instance_ids()]
        if not worker_ids:
            yield RouterResponse(worker_id=-1, prefix_hit_rate=0.0).model_dump()
            return

        now = time.time()
        self._sweep_pending(now)

        # Metrics aggregation API changed - using None for now (router works without it)
        metrics = None  # TODO: Replace with proper metrics query when API is available
        if self.router_type == "kv_load":
            wid, _ = self._get_underloaded(metrics)
            yield RouterResponse(worker_id=wid, prefix_hit_rate=0.0).model_dump()
            return

        scores: OverlapScores = await self.indexer.find_matches_for_request(req.tokens, 0)
        chosen, chosen_ctx, all_ctx, raw_scores, probs = self._select_worker(worker_ids, req, metrics, scores)
        # Non-blocking CSV append for decision metadata
        try:
            ts_ms = int(time.time() * 1000)
            overlap_chosen = float(scores.scores.get(chosen, 0.0))
            decode_cost = float(chosen_ctx.get("decode_cost", 0.0))
            prefill_cost_chosen = float(chosen_ctx.get("prefill_cost", 0.0))
            iat = self._norm_level(req.interarrival, "MEDIUM")
            last_w, _ = self._get_prefix(req.prefix_id)
            stickiness = 1.0 if (last_w is not None and chosen == last_w) else 0.0
            load_mod = float(chosen_ctx.get("load_mod", 0.0))

            row = [
                ts_ms,
                len(req.tokens) if getattr(req, "tokens", None) else 0,
                req.prefix_id,
                max(int(req.reuse_budget), 0),
                chosen,
                f"{overlap_chosen:.6f}",
                f"{decode_cost:.6f}",
                f"{prefill_cost_chosen:.6f}",
                iat,
                f"{stickiness:.3f}",
                f"{load_mod:.6f}",
            ]

            def _append_row():
                try:
                    with self._router_csv_lock:
                        with open(self._router_csv_path, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(row)
                except Exception:
                    pass

            await asyncio.to_thread(_append_row)
        except Exception:
            pass
        last_w, _ = self._get_prefix(req.prefix_id)

        osl = self._norm_level(req.expected_osl, "MEDIUM")
        iat = self._norm_level(req.interarrival, "MEDIUM")
        decode_cost = self._decode_cost(osl)
        overlap_chosen = float(scores.scores.get(chosen, 0.0))
        prefill_cost_chosen = self._prefill_cost_for_worker(req.tokens, overlap_chosen)
        iat_factor = self._iat_factor(iat)

        # Update prefix state (remove immediately when reuse==0)
        self._set_prefix(
            req.prefix_id,
            chosen,
            reuse_remaining=max(int(req.reuse_budget), 0),
            decode_cost=decode_cost,
            prefill_cost=prefill_cost_chosen,
            iat_factor=iat_factor,
        )

        # Build feature x for chosen & store pending decision
        x = self._feature_vector(
            wid=chosen,
            metrics=metrics,
            scores=scores,
            last_w=last_w,
            reuse_after=max(int(req.reuse_budget), 0),
            decode_cost=decode_cost,
            prefill_cost=prefill_cost_chosen,
            iat_factor=iat_factor,
        )
        decision_id = uuid.uuid4().hex
        with self._pending_lock:
            self.pending[decision_id] = {
                "wid": int(chosen),
                "x": x,
                "osl": osl,
                "prefill_bin": self._prefill_bin(prefill_cost_chosen),
                "start_ts": now,
                "prefix_id": req.prefix_id,
                "tokens_in": len(req.tokens),
                "reuse_after": int(req.reuse_budget),
                "overlap": overlap_chosen,
                "prefill_cost": float(prefill_cost_chosen),
                "decode_cost": float(decode_cost),
            }

        # Decision trace
        if self.debug_traces:
            worker_list = [int(w) for w in worker_ids]
            details = {
                wid: {
                    "score": float(raw_scores[i]),
                    "prob": float(probs[i]),
                    **all_ctx[wid],
                }
                for i, wid in enumerate(worker_list)
            }
            self._emit_trace("decision",
                             {
                                 "decision_id": decision_id,
                                 "prefix_id": req.prefix_id,
                                 "chosen": int(chosen),
                                 "workers": details,
                             })

        logger.info(
            "Router picked worker=%s decision=%s prefix=%s "
            "(last=%s reuse_after=%s osl=%s prefill_cost=%.3f iat=%s overlap=%.3f)",
            chosen,
            decision_id,
            req.prefix_id,
            last_w,
            req.reuse_budget,
            osl,
            prefill_cost_chosen,
            iat,
            overlap_chosen,
        )

        resp = RouterResponse(worker_id=chosen, prefix_hit_rate=overlap_chosen, decision_id=decision_id)
        yield resp.model_dump()
        return

    # --------------------- feedback endpoint --------------------- #
    async def feedback(self, request: dict):
        """Ex-post reward update from processor with observed latency."""
        try:
            fb = FeedbackRequest(**request)
        except Exception as e:
            ack = FeedbackAck(ok=False, used_baseline=0.0, reward=0.0, error=str(e))
            yield ack.model_dump()
            return

        with self._pending_lock:
            decision = self.pending.pop(fb.decision_id, None)

        if not decision:
            ack = FeedbackAck(ok=False, used_baseline=0.0, reward=0.0, error="unknown_decision")
            yield ack.model_dump()
            return

        wid: int = int(decision["wid"])
        x: np.ndarray = decision["x"]
        osl: str = str(decision["osl"])
        prefill_bin: str = str(decision["prefill_bin"])
        tokens_out = None if fb.tokens_out is None else int(fb.tokens_out)
        metric, per_tok = self._latency_metric(float(fb.latency_ms), tokens_out)

        # Baseline lookup (hierarchical)
        baseline_before = self._get_latency_baseline(wid, osl, prefill_bin, per_tok, fallback=metric)

        reward = self._metric_to_reward(metric, baseline_before, bool(fb.success))

        # Update EMAs only on successes (keeps baselines clean)
        if fb.success:
            baseline_after = self._update_latency_baselines(wid, osl, prefill_bin, metric, per_tok)
        else:
            baseline_after = baseline_before

        # Update bandits with ex-post reward
        self._update_bandit(wid, reward)
        self._update_contextual(wid, x, reward)

        self._emit_trace(
            "feedback",
            {
                "decision_id": fb.decision_id,
                "wid": wid,
                "latency_ms": float(fb.latency_ms),
                "tokens_out": tokens_out,
                "metric": metric,
                "per_tok": per_tok,
                "baseline_used": baseline_before,
                "baseline_after": baseline_after,
                "reward": reward,
                "success": bool(fb.success),
                "finish_reason": fb.finish_reason or "",
            })

        logger.info(
            "Feedback: wid=%s decision=%s metric=%.3f%s baseline=%.3f reward=%.3f success=%s",
            wid,
            fb.decision_id,
            metric,
            " ms/tok" if per_tok else " ms",
            baseline_before,
            reward,
            fb.success,
        )

        ack = FeedbackAck(ok=True, used_baseline=float(baseline_before), reward=float(reward), worker_id=wid)
        yield ack.model_dump()
        return

    # --------------------- helpers --------------------- #
    def _get_underloaded(self, metrics: dict[str, Any] | None):
        if not metrics or not metrics.get("endpoints"):
            wid = int(random.choice(self.engine_client.instance_ids()))
            return wid, 0.0
        loads = {ep.get("worker_id"): ep.get("gpu_cache_usage_perc", 0.0) for ep in metrics["endpoints"]}
        min_val = min(loads.values())
        candidates = [wid for wid, v in loads.items() if v == min_val]
        return random.choice(candidates), min_val


# ---------------------- worker entry point ---------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Workload-aware router (LinTS + feedback + timeout + traces)")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--router-type", type=str, default="kv")
    parser.add_argument("--min-workers", type=int, default=1)

    # Affinity / exploration
    parser.add_argument("--affinity-base", type=float, default=0.30)
    parser.add_argument("--affinity-reuse-weight", type=float, default=0.15)
    parser.add_argument("--affinity-iat-weight", type=float, default=0.20)
    parser.add_argument("--base-ts-weight", type=float, default=0.10)
    parser.add_argument("--sticky-load-floor", type=float, default=0.70)

    # Softmax temp
    parser.add_argument("--temp-base", type=float, default=1.0)
    parser.add_argument("--temp-min", type=float, default=0.15)
    parser.add_argument("--temp-max", type=float, default=2.0)

    # Switching cost
    parser.add_argument("--switch-cost-base", type=float, default=0.20)
    parser.add_argument("--switch-cost-reuse", type=float, default=0.08)
    parser.add_argument("--switch-cost-iat", type=float, default=0.05)

    # Load / opportunity cost
    parser.add_argument("--queue-penalty-weight", type=float, default=0.50)
    parser.add_argument("--gpu-penalty-weight", type=float, default=1.00)
    parser.add_argument("--outstanding-work-weight", type=float, default=0.45)
    parser.add_argument("--job-gpu-coupling-weight", type=float, default=0.40)
    parser.add_argument("--job-queue-coupling-weight", type=float, default=0.20)

    # Prefill / ISL
    parser.add_argument("--prefill-token-scale", type=float, default=1024.0)
    parser.add_argument("--prefill-weight", type=float, default=1.0)

    # LinTS
    parser.add_argument("--lints-lambda", type=float, default=1.0)
    parser.add_argument("--lints-v", type=float, default=0.25)
    parser.add_argument("--lints-forget", type=float, default=0.7)

    # Feedback timeout & sweep
    parser.add_argument("--feedback-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--pending-sweep-interval-seconds", type=float, default=5.0)
    parser.add_argument("--timeout-reward", type=float, default=0.0)

    # Latency EMA
    parser.add_argument("--latency-ema-alpha", type=float, default=0.2)

    # Traces
    parser.add_argument("--debug-traces", action="store_true", default=False)
    parser.add_argument("--debug-trace-dir", type=str, default="/tmp/dynamo_router_traces")
    parser.add_argument("--debug-buffer-size", type=int, default=2000)

    return parser.parse_args()


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    args = parse_args()

    component = runtime.namespace("dynamo").component("router")
    await component.create_service()
    logger.info("Initializing WorkloadAwareRouter (LinTS + feedback + timeout + traces)")

    router = WorkloadAwareRouter(
        runtime,
        block_size=args.block_size,
        router_type=args.router_type.lower(),
        min_workers=args.min_workers,
        affinity_base=args.affinity_base,
        affinity_reuse_weight=args.affinity_reuse_weight,
        affinity_iat_weight=args.affinity_iat_weight,
        base_ts_weight=args.base_ts_weight,
        sticky_load_floor=args.sticky_load_floor,
        temp_base=args.temp_base,
        temp_min=args.temp_min,
        temp_max=args.temp_max,
        switch_cost_base=args.switch_cost_base,
        switch_cost_reuse=args.switch_cost_reuse,
        switch_cost_iat=args.switch_cost_iat,
        queue_penalty_weight=args.queue_penalty_weight,
        gpu_penalty_weight=args.gpu_penalty_weight,
        outstanding_work_weight=args.outstanding_work_weight,
        job_gpu_coupling_weight=args.job_gpu_coupling_weight,
        job_queue_coupling_weight=args.job_queue_coupling_weight,
        prefill_token_scale=args.prefill_token_scale,
        prefill_weight=args.prefill_weight,
        lints_lambda=args.lints_lambda,
        lints_v=args.lints_v,
        lints_forget=args.lints_forget,
        feedback_timeout_seconds=args.feedback_timeout_seconds,
        pending_sweep_interval_seconds=args.pending_sweep_interval_seconds,
        timeout_reward=args.timeout_reward,
        latency_ema_alpha=args.latency_ema_alpha,
        debug_traces=args.debug_traces,
        debug_trace_dir=args.debug_trace_dir,
        debug_buffer_size=args.debug_buffer_size,
    )
    await router.initialize()

    # Selection endpoint
    await asyncio.gather(
        component.endpoint("find_worker").serve_endpoint(router.generate),
        component.endpoint("feedback").serve_endpoint(router.feedback),
    )


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
