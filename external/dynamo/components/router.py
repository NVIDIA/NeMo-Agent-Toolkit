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
Optimized Thompson Sampling Router with Prometheus Metrics.

This router implements Contextual Thompson Sampling with:
  - KV overlap locality
  - Remaining per-prefix requests (reuse_budget)
  - OSL-based decode cost, ISL/prefill cost per worker
  - IAT-based stickiness/opportunity weighting
  - Instant & outstanding load (no TTL decay)
  - Delayed bandit update using observed latency via `feedback` endpoint
  - Timeout penalty for missing feedback
  - Prometheus metrics (instead of CSV)
  - Debug traces for offline analysis

Key differences from generalized/router.py:
  - Uses Prometheus metrics instead of CSV logging
  - Removed CSV file I/O
  - Added comprehensive Prometheus gauges, counters, and histograms
"""

import argparse
import asyncio
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
from pathlib import Path
from typing import Any

import numpy as np
import uvloop
import yaml
from dynamo.runtime import DistributedRuntime
from dynamo.runtime import dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

# KV cache overlap scoring — uses RadixTree + ZmqKvEventListener from dynamo.llm.
# Backend-agnostic: works identically with SGLang and vLLM workers.
# Falls back gracefully to empty scores if dynamo.llm primitives are unavailable.
from kv_indexer import KvIndexer
from kv_indexer import OverlapScores
from pydantic import BaseModel

configure_dynamo_logging()
logger = logging.getLogger(__name__)

WorkerId = int


# ---------------------- config loading ---------------------- #
def get_default_config_path() -> Path:
    """Get path to default config.yaml in the same directory as this script."""
    return Path(__file__).parent / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, uses default config.yaml.

    Returns:
        Configuration dictionary with nested structure.
    """
    if config_path is None:
        config_path = get_default_config_path()

    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning("Config file not found: %s, using built-in defaults", config_path)
        return get_builtin_defaults()

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Loaded config from: %s", config_path)
    return config


def get_builtin_defaults() -> dict[str, Any]:
    """Return built-in default configuration (matches config.yaml)."""
    return {
        "infrastructure": {
            "block_size": 64,
            "router_type": "kv",
            "min_workers": 1,
        },
        "affinity": {
            "base": 0.30,
            "reuse_weight": 0.15,
            "iat_weight": 0.20,
            "sticky_load_floor": 0.01,
        },
        "exploration": {
            "base_ts_weight": 0.10,
            "temperature": {
                "base": 1.0,
                "min": 0.15,
                "max": 2.0,
            },
        },
        "switching_cost": {
            "base": 0.20,
            "reuse_penalty": 0.08,
            "iat_penalty": 0.05,
        },
        "load_balancing": {
            "queue_penalty_weight": 0.50,
            "gpu_penalty_weight": 1.00,
            "outstanding_work_weight": 0.45,
            "job_gpu_coupling_weight": 0.40,
            "job_queue_coupling_weight": 0.20,
        },
        "prefill": {
            "token_scale": 1024.0,
            "weight": 1.0,
        },
        "lints": {
            "lambda": 1.0,
            "v": 0.25,
            "forget_rate": 0.995,
        },
        "feedback": {
            "timeout_seconds": 120.0,
            "sweep_interval_seconds": 5.0,
            "timeout_reward": 0.0,
            "latency_ema_alpha": 0.2,
        },
        "debug": {
            "traces_enabled": False,
            "trace_dir": "/tmp/dynamo_router_traces",
            "buffer_size": 2000,
        },
    }


def get_nested(config: dict, dotted_key: str, default: Any = None) -> Any:
    """Get a nested value from config using dot notation.

    Args:
        config: Configuration dictionary
        dotted_key: Key in dot notation, e.g., "affinity.base"
        default: Default value if key not found

    Returns:
        Value at the nested key, or default if not found.
    """
    keys = dotted_key.split(".")
    obj = config
    for k in keys:
        if not isinstance(obj, dict) or k not in obj:
            return default
        obj = obj[k]
    return obj


def set_nested(config: dict, dotted_key: str, value: Any) -> None:
    """Set a nested value in config using dot notation.

    Args:
        config: Configuration dictionary (modified in place)
        dotted_key: Key in dot notation, e.g., "affinity.base"
        value: Value to set
    """
    keys = dotted_key.split(".")
    obj = config
    for k in keys[:-1]:
        if k not in obj:
            obj[k] = {}
        obj = obj[k]
    obj[keys[-1]] = value


def auto_cast(value_str: str) -> Any:
    """Auto-cast a string value to appropriate type.

    Args:
        value_str: String value from CLI

    Returns:
        Value cast to int, float, bool, or str as appropriate.
    """
    # Boolean
    if value_str.lower() in ("true", "yes", "1"):
        return True
    if value_str.lower() in ("false", "no", "0"):
        return False

    # Integer
    try:
        return int(value_str)
    except ValueError:
        pass

    # Float
    try:
        return float(value_str)
    except ValueError:
        pass

    # String
    return value_str


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply CLI argument overrides to configuration.

    Args:
        config: Base configuration dictionary
        args: Parsed CLI arguments

    Returns:
        Configuration with CLI overrides applied.
    """
    # Apply explicit CLI flags
    if args.affinity_base is not None:
        set_nested(config, "affinity.base", args.affinity_base)
        logger.info("CLI override: affinity.base = %s", args.affinity_base)

    if args.temp_base is not None:
        set_nested(config, "exploration.temperature.base", args.temp_base)
        logger.info("CLI override: exploration.temperature.base = %s", args.temp_base)

    if args.lints_v is not None:
        set_nested(config, "lints.v", args.lints_v)
        logger.info("CLI override: lints.v = %s", args.lints_v)

    # Apply generic --override flags
    if args.override:
        for override in args.override:
            if "=" not in override:
                logger.warning("Invalid override format (expected key=value): %s", override)
                continue
            key, value_str = override.split("=", 1)
            value = auto_cast(value_str)
            set_nested(config, key, value)
            logger.info("CLI override: %s = %s", key, value)

    return config


def _init_prometheus_metrics():
    """Initialize Prometheus metrics lazily."""
    import functools

    @functools.lru_cache(maxsize=1)
    def _init() -> dict:
        metrics: dict = {}
        try:
            from prometheus_client import REGISTRY
            from prometheus_client import Counter
            from prometheus_client import Gauge
            from prometheus_client import Histogram

            metrics["decisions_total"] = Counter(
                "thompson_router_decisions_total",
                "Total routing decisions by worker",
                ["worker_id"],
                registry=REGISTRY,
            )
            metrics["kv_overlap"] = Gauge(
                "thompson_router_kv_overlap",
                "KV cache overlap score for last decision by worker",
                ["worker_id"],
                registry=REGISTRY,
            )
            metrics["feedback_latency"] = Histogram(
                "thompson_router_feedback_latency_seconds",
                "Latency from feedback by worker",
                ["worker_id"],
                buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
                registry=REGISTRY,
            )
            metrics["reward"] = Gauge(
                "thompson_router_reward",
                "Last computed reward by worker",
                ["worker_id"],
                registry=REGISTRY,
            )
            metrics["pending_decisions"] = Gauge(
                "thompson_router_pending_decisions",
                "Number of pending decisions awaiting feedback",
                registry=REGISTRY,
            )
            metrics["timeout_penalties"] = Counter(
                "thompson_router_timeout_penalties_total",
                "Total timeout penalties applied",
                registry=REGISTRY,
            )
            metrics["sticky_decisions"] = Counter(
                "thompson_router_sticky_decisions_total",
                "Decisions that stayed on the same worker (sticky)",
                registry=REGISTRY,
            )
            metrics["switch_decisions"] = Counter(
                "thompson_router_switch_decisions_total",
                "Decisions that switched to a different worker",
                registry=REGISTRY,
            )
            metrics["beta_alpha"] = Gauge(
                "thompson_router_beta_alpha",
                "Beta distribution alpha parameter by worker",
                ["worker_id"],
                registry=REGISTRY,
            )
            metrics["beta_beta"] = Gauge(
                "thompson_router_beta_beta",
                "Beta distribution beta parameter by worker",
                ["worker_id"],
                registry=REGISTRY,
            )
            metrics["prefix_state_size"] = Gauge(
                "thompson_router_prefix_state_size",
                "Number of active prefix states",
                registry=REGISTRY,
            )
            metrics["reuse_budget"] = Histogram(
                "thompson_router_reuse_budget",
                "Distribution of reuse_budget values",
                buckets=[0, 1, 2, 5, 10, 20, 50, 100],
                registry=REGISTRY,
            )
            metrics["tokens_per_request"] = Histogram(
                "thompson_router_tokens_per_request",
                "Distribution of input token counts",
                buckets=[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
                registry=REGISTRY,
            )
            logger.info("Prometheus metrics initialized for router")
        except ImportError:
            logger.warning("prometheus_client not available, metrics disabled")

        return metrics

    return _init()


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
    Contextual Thompson Sampling router with Prometheus metrics.
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
        feedback_timeout_seconds: float = 120.0,
        pending_sweep_interval_seconds: float = 5.0,
        timeout_reward: float = 0.0,
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

        # Pending decisions waiting for feedback
        self.pending: dict[str, dict[str, Any]] = {}

        # Debug traces
        self.debug_traces = bool(debug_traces)
        self.debug_trace_dir = str(debug_trace_dir)
        self.recent_traces: deque = deque(maxlen=int(debug_buffer_size))
        if self.debug_traces:
            os.makedirs(self.debug_trace_dir, exist_ok=True)
            logger.info("Router debug traces enabled -> %s", self.debug_trace_dir)

        # Prometheus metrics
        self._metrics = {}

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
        # Initialize Prometheus metrics
        self._metrics = _init_prometheus_metrics()

        # Connect to actual workers at workers.{component}.generate
        # Workers are in the "workers" namespace (hidden from frontend discovery)
        # Component name varies by backend (REQUIRED - no default):
        #   - SGLang: uses "worker" (set via --endpoint workers.worker.generate)
        #   - vLLM: uses "backend" (hardcoded in dynamo.vllm)
        worker_component = os.environ.get("DYNAMO_WORKER_COMPONENT")
        if not worker_component:
            raise ValueError("DYNAMO_WORKER_COMPONENT environment variable is required. "
                             "Set to 'worker' for SGLang or 'backend' for vLLM.")
        engine = self.runtime.namespace("workers").component(worker_component)
        logger.info("Getting engine client for workers/%s/generate", worker_component)
        self.engine_client = await engine.endpoint("generate").client()

        min_workers = int(self.min_workers)
        if min_workers < 0:
            raise ValueError(f"min_workers must be >= 0, got {min_workers}")

        timeout_s = float(os.environ.get("DYNAMO_ROUTER_WAIT_FOR_WORKERS_TIMEOUT_S", "600"))
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("DYNAMO_ROUTER_WAIT_FOR_WORKERS_TIMEOUT_S must be a finite number > 0")

        deadline = time.monotonic() + timeout_s
        backoff_s = 0.5

        logger.info("Waiting for backend workers (min_workers=%d, timeout_s=%.1f)...", min_workers, timeout_s)

        if min_workers == 0:
            instance_ids_raw = list(self.engine_client.instance_ids())
            logger.info("Backend workers discovered (min_workers=0): %s", instance_ids_raw)
        else:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out after {timeout_s}s waiting for >= {min_workers} backend worker(s)")

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

        self.indexer = KvIndexer(engine, self.block_size)

        # Start background metrics scraper (non-blocking HTTP scrapes in a daemon thread).
        discovered_worker_ids = sorted(int(w) for w in self.engine_client.instance_ids())
        self._start_metrics_scraper(discovered_worker_ids, interval=1.0)

        # Register workers' ZMQ KV event streams for overlap scoring.
        # Port allocation: KV_EVENT_BASE_PORT + worker_index (sorted by instance_id).
        kv_event_base_port = int(os.environ.get("KV_EVENT_BASE_PORT", "0"))
        enable_kv_events = os.environ.get("ENABLE_KV_EVENTS", "false").lower() == "true"
        if enable_kv_events and kv_event_base_port > 0:
            discovered_ids = sorted(int(w) for w in self.engine_client.instance_ids())
            for idx, wid in enumerate(discovered_ids):
                endpoint = f"tcp://127.0.0.1:{kv_event_base_port + idx}"
                self.indexer.add_worker(wid, endpoint)
            self.indexer.start_background_drain(interval=0.25)
            logger.info(
                "KvIndexer: %d workers registered, background drain started (base_port=%d)",
                len(discovered_ids),
                kv_event_base_port,
            )
        else:
            logger.info(
                "KvIndexer: KV event overlap disabled (ENABLE_KV_EVENTS=%s, KV_EVENT_BASE_PORT=%s)",
                os.environ.get("ENABLE_KV_EVENTS", "unset"),
                os.environ.get("KV_EVENT_BASE_PORT", "unset"),
            )

        self._initialize_bandits()
        self._initialize_contextual()
        logger.info("WorkloadAwareRouter initialized with %d backend worker(s)",
                    len(list(self.engine_client.instance_ids())))

    @safe_update("_init_lock")
    def _initialize_bandits(self):
        for wid in self.engine_client.instance_ids():
            wid = int(wid)
            self.worker_bandits.setdefault(wid, (1.0, 1.0))
            # Update Prometheus metrics
            if self._metrics.get("beta_alpha"):
                self._metrics["beta_alpha"].labels(worker_id=str(wid)).set(1.0)
            if self._metrics.get("beta_beta"):
                self._metrics["beta_beta"].labels(worker_id=str(wid)).set(1.0)

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
        """Record/refresh prefix assignment."""
        if reuse_remaining <= 0:
            self.prefix_cache_state.pop(pid, None)
            self.prefix_meta.pop(pid, None)
        else:
            self.prefix_cache_state[pid] = {"worker": wid, "reuse_remaining": max(0, int(reuse_remaining))}
            self.prefix_meta[pid] = {
                "decode_cost": float(decode_cost),
                "prefill_cost": float(max(prefill_cost, 0.0)),
                "iat_factor": float(iat_factor),
            }

        # Update prefix state size metric
        if self._metrics.get("prefix_state_size"):
            self._metrics["prefix_state_size"].set(len(self.prefix_cache_state))

    def _worker_outstanding(self, wid: int) -> tuple[int, float]:
        """Returns (reuse_total, work_total) for a worker."""
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

    # Backend-agnostic metric line prefixes.
    # Each canonical metric maps to the exact line prefix(es) for SGLang and vLLM.
    # Using startswith() avoids substring collisions (e.g. pending_prealloc_token_usage).
    _METRIC_PREFIXES: dict[str, list[str]] = {
        "gpu_cache_usage": [
            "sglang:token_usage{",  # SGLang: KV cache fraction (0-1)
            "vllm:kv_cache_usage_perc{",  # vLLM: same semantic, different name
        ],
        "queue_depth": [
            "sglang:num_queue_reqs{",  # SGLang: scheduler queue depth
            "vllm:num_requests_waiting{",  # vLLM: same semantic
        ],
    }

    # ---- cached metrics scraper (non-blocking) ---- #

    def _start_metrics_scraper(self, worker_ids: list[int], interval: float = 1.0) -> None:
        """Start a background thread that periodically scrapes worker metrics.

        The scrape runs in a daemon thread to avoid blocking the asyncio event
        loop.  Results are cached in ``_scraped_metrics`` and read lock-free
        by ``_build_internal_metrics`` on every routing decision.
        """
        if hasattr(self, "_scraper_running") and self._scraper_running:
            return

        self._scraped_metrics: dict[int, dict[str, float]] = {}  # wid -> {gpu, queue}
        self._scraper_running = True
        self._scraper_worker_ids = sorted(worker_ids)
        self._scraper_base_port = int(os.environ.get("WORKER_METRICS_PORT", "0"))

        def _scrape_loop() -> None:
            import urllib.request
            while self._scraper_running:
                for idx, wid in enumerate(self._scraper_worker_ids):
                    if self._scraper_base_port <= 0:
                        break
                    port = self._scraper_base_port + idx
                    gpu = 0.0
                    queue = 0.0
                    try:
                        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=1.0)
                        body = resp.read().decode("utf-8", errors="replace")
                        for line in body.splitlines():
                            if line.startswith("#"):
                                continue
                            for prefix in self._METRIC_PREFIXES["gpu_cache_usage"]:
                                if line.startswith(prefix):
                                    gpu = float(line.rsplit(" ", 1)[-1])
                                    break
                            for prefix in self._METRIC_PREFIXES["queue_depth"]:
                                if line.startswith(prefix):
                                    queue = float(line.rsplit(" ", 1)[-1])
                                    break
                    except Exception:
                        pass
                    self._scraped_metrics[wid] = {"gpu": gpu, "queue": queue}
                time.sleep(interval)

        t = threading.Thread(target=_scrape_loop, daemon=True, name="metrics-scraper")
        t.start()
        logger.info("Started background metrics scraper (interval=%.1fs, workers=%d)", interval, len(worker_ids))

    def _build_internal_metrics(self, worker_ids: list[int]) -> dict[str, Any]:
        """Build a metrics dict from cached scrapes + instant pending counts.

        The worker metrics are scraped in a background thread (no event loop
        blocking).  Pending-decision counts provide an instant supplement that
        reacts within the same function call.
        """
        # Count in-flight (pending) decisions per worker.
        pending_per_worker: dict[int, int] = {wid: 0 for wid in worker_ids}
        with self._pending_lock:
            for rec in self.pending.values():
                w = int(rec.get("wid", -1))
                if w in pending_per_worker:
                    pending_per_worker[w] += 1

        sorted_ids = sorted(worker_ids)
        endpoints = []
        for wid in sorted_ids:
            pending = float(pending_per_worker.get(wid, 0))
            cached = getattr(self, "_scraped_metrics", {}).get(wid)

            if cached:
                gpu_usage = cached["gpu"]
                queue_depth = cached["queue"]
            else:
                # Fallback before first scrape completes
                gpu_usage = min(1.0, pending / 20.0)
                queue_depth = pending

            # Blend: use max of scraped queue and pending count
            effective_queue = max(queue_depth, pending)

            endpoints.append({
                "worker_id": wid,
                "num_requests_waiting": effective_queue,
                "gpu_cache_usage_perc": gpu_usage,
            })

        return {"endpoints": endpoints}

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
            new_alpha = alpha + r
            new_beta = beta + 1.0 - r
            self.worker_bandits[worker_id] = (new_alpha, new_beta)

        # Update Prometheus metrics
        if self._metrics.get("beta_alpha"):
            self._metrics["beta_alpha"].labels(worker_id=str(worker_id)).set(new_alpha)
        if self._metrics.get("beta_beta"):
            self._metrics["beta_beta"].labels(worker_id=str(worker_id)).set(new_beta)

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
        self.lat_ema_global[per_tok] = self._ema_update(self.lat_ema_global[per_tok], metric)
        key_w = (wid, per_tok)
        self.lat_ema_worker[key_w] = self._ema_update(self.lat_ema_worker.get(key_w), metric)
        key_b = (wid, osl, prefill_bin, per_tok)
        self.lat_ema_bucket[key_b] = self._ema_update(self.lat_ema_bucket.get(key_b), metric)
        return self.lat_ema_bucket[key_b]

    @staticmethod
    def _latency_metric(latency_ms: float, tokens_out: int | None) -> tuple[float, bool]:
        if tokens_out is not None and int(tokens_out) > 0:
            return float(latency_ms) / float(max(1, int(tokens_out))), True
        return float(latency_ms), False

    @staticmethod
    def _metric_to_reward(metric: float, baseline: float, success: bool) -> float:
        if not success:
            return 0.0
        denom = max(1e-3, baseline)
        ratio = metric / denom
        return float(1.0 / (1.0 + ratio))

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

            # Update pending count metric
            if self._metrics.get("pending_decisions"):
                self._metrics["pending_decisions"].set(len(self.pending))

        for did, rec in expired:
            wid = int(rec["wid"])
            x = rec["x"]
            reward = float(self.timeout_reward)
            self._update_bandit(wid, reward)
            self._update_contextual(wid, x, reward)

            if self._metrics.get("timeout_penalties"):
                self._metrics["timeout_penalties"].inc()

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

        # Track tokens per request
        if self._metrics.get("tokens_per_request"):
            self._metrics["tokens_per_request"].observe(len(req.tokens))
        if self._metrics.get("reuse_budget"):
            self._metrics["reuse_budget"].observe(req.reuse_budget)

        metrics = self._build_internal_metrics(worker_ids)
        if self.router_type == "kv_load":
            wid, _ = self._get_underloaded(metrics)
            yield RouterResponse(worker_id=wid, prefix_hit_rate=0.0).model_dump()
            return

        scores: OverlapScores = await self.indexer.find_matches_for_request(req.tokens, 0)
        chosen, chosen_ctx, all_ctx, raw_scores, probs = self._select_worker(worker_ids, req, metrics, scores)

        last_w, _ = self._get_prefix(req.prefix_id)

        osl = self._norm_level(req.expected_osl, "MEDIUM")
        iat = self._norm_level(req.interarrival, "MEDIUM")
        decode_cost = self._decode_cost(osl)
        overlap_chosen = float(scores.scores.get(chosen, 0.0))
        prefill_cost_chosen = self._prefill_cost_for_worker(req.tokens, overlap_chosen)
        iat_factor = self._iat_factor(iat)

        # Update prefix state
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
            # Update pending count metric
            if self._metrics.get("pending_decisions"):
                self._metrics["pending_decisions"].set(len(self.pending))

        # Update Prometheus metrics
        if self._metrics.get("decisions_total"):
            self._metrics["decisions_total"].labels(worker_id=str(chosen)).inc()
        if self._metrics.get("kv_overlap"):
            self._metrics["kv_overlap"].labels(worker_id=str(chosen)).set(overlap_chosen)

        # Track sticky vs switch decisions
        if last_w is not None:
            if chosen == last_w:
                if self._metrics.get("sticky_decisions"):
                    self._metrics["sticky_decisions"].inc()
            elif self._metrics.get("switch_decisions"):
                self._metrics["switch_decisions"].inc()

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
            "Router picked worker=%s decision=%s prefix=%s (last=%s reuse_after=%s osl=%s "
            "prefill_cost=%.3f iat=%s overlap=%.3f)",
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
            # Update pending count metric
            if self._metrics.get("pending_decisions"):
                self._metrics["pending_decisions"].set(len(self.pending))

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

        # Update EMAs only on successes
        if fb.success:
            baseline_after = self._update_latency_baselines(wid, osl, prefill_bin, metric, per_tok)
        else:
            baseline_after = baseline_before

        # Update bandits with ex-post reward
        self._update_bandit(wid, reward)
        self._update_contextual(wid, x, reward)

        # Update Prometheus metrics
        if self._metrics.get("feedback_latency"):
            self._metrics["feedback_latency"].labels(worker_id=str(wid)).observe(fb.latency_ms / 1000.0)
        if self._metrics.get("reward"):
            self._metrics["reward"].labels(worker_id=str(wid)).set(reward)

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
            wid = int(random.choice(list(self.engine_client.instance_ids())))
            return wid, 0.0
        loads = {ep.get("worker_id"): ep.get("gpu_cache_usage_perc", 0.0) for ep in metrics["endpoints"]}
        min_val = min(loads.values())
        candidates = [wid for wid, v in loads.items() if v == min_val]
        return random.choice(candidates), min_val


# ---------------------- worker entry point ---------------------- #
def parse_args():
    """Parse minimal CLI arguments.

    The router uses a YAML config file for most parameters.
    Only frequently-tuned parameters have dedicated CLI flags.
    Use --override for any other parameter.

    See PARAMETERS.md for full documentation.
    """
    parser = argparse.ArgumentParser(
        description="Optimized Thompson Sampling Router with Prometheus Metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config
  python router.py

  # Use custom config file
  python router.py --config /path/to/config.yaml

  # Override specific values
  python router.py --config config.yaml --affinity-base 0.5 --temp-base 1.5

  # Override any config value
  python router.py --config config.yaml --override load_balancing.gpu_penalty_weight=2.0

See PARAMETERS.md for full parameter documentation.
        """,
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: config.yaml in script directory)",
    )

    # Primary tuning knobs (explicit CLI flags)
    parser.add_argument(
        "--affinity-base",
        type=float,
        default=None,
        help="Primary stickiness control [0.0-1.0] (overrides config)",
    )
    parser.add_argument(
        "--temp-base",
        type=float,
        default=None,
        help="Primary exploration control [0.15-2.0] (overrides config)",
    )
    parser.add_argument(
        "--lints-v",
        type=float,
        default=None,
        help="LinTS exploration variance [0.0-1.0] (overrides config)",
    )

    # Generic override for any config value
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any config value using dot notation (repeatable)",
    )

    return parser.parse_args()


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    # Parse CLI and load config
    args = parse_args()
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)

    component = runtime.namespace("dynamo").component("router")
    logger.info("Initializing Optimized Thompson Sampling Router (Prometheus metrics)")

    # Resolve block_size: env var KV_BLOCK_SIZE (set by startup script from
    # DYNAMO_KV_BLOCK_SIZE) takes precedence over config.yaml so there is a
    # single source of truth shared with workers and the frontend.
    config_block_size = get_nested(config, "infrastructure.block_size", 64)
    env_block_size_str = os.environ.get("KV_BLOCK_SIZE")
    if env_block_size_str is not None:
        env_block_size = int(env_block_size_str)
        if env_block_size != config_block_size:
            logger.warning(
                "KV_BLOCK_SIZE env var (%d) overrides config.yaml block_size (%d). "
                "Update config.yaml to match DYNAMO_KV_BLOCK_SIZE in .env to silence this warning.",
                env_block_size,
                config_block_size,
            )
        block_size = env_block_size
    else:
        block_size = config_block_size

    # Extract config values with nested access
    router = WorkloadAwareRouter(
        runtime,
        # Infrastructure
        block_size=block_size,
        router_type=str(get_nested(config, "infrastructure.router_type", "kv")).lower(),
        min_workers=get_nested(config, "infrastructure.min_workers", 1),
        # Affinity
        affinity_base=get_nested(config, "affinity.base", 0.30),
        affinity_reuse_weight=get_nested(config, "affinity.reuse_weight", 0.15),
        affinity_iat_weight=get_nested(config, "affinity.iat_weight", 0.20),
        sticky_load_floor=get_nested(config, "affinity.sticky_load_floor", 0.70),
        # Exploration
        base_ts_weight=get_nested(config, "exploration.base_ts_weight", 0.10),
        temp_base=get_nested(config, "exploration.temperature.base", 1.0),
        temp_min=get_nested(config, "exploration.temperature.min", 0.15),
        temp_max=get_nested(config, "exploration.temperature.max", 2.0),
        # Switching cost
        switch_cost_base=get_nested(config, "switching_cost.base", 0.20),
        switch_cost_reuse=get_nested(config, "switching_cost.reuse_penalty", 0.08),
        switch_cost_iat=get_nested(config, "switching_cost.iat_penalty", 0.05),
        # Load balancing
        queue_penalty_weight=get_nested(config, "load_balancing.queue_penalty_weight", 0.50),
        gpu_penalty_weight=get_nested(config, "load_balancing.gpu_penalty_weight", 1.00),
        outstanding_work_weight=get_nested(config, "load_balancing.outstanding_work_weight", 0.45),
        job_gpu_coupling_weight=get_nested(config, "load_balancing.job_gpu_coupling_weight", 0.40),
        job_queue_coupling_weight=get_nested(config, "load_balancing.job_queue_coupling_weight", 0.20),
        # Prefill
        prefill_token_scale=get_nested(config, "prefill.token_scale", 1024.0),
        prefill_weight=get_nested(config, "prefill.weight", 1.0),
        # LinTS
        lints_lambda=get_nested(config, "lints.lambda", 1.0),
        lints_v=get_nested(config, "lints.v", 0.25),
        lints_forget=get_nested(config, "lints.forget_rate", 0.995),
        # Feedback
        feedback_timeout_seconds=get_nested(config, "feedback.timeout_seconds", 120.0),
        pending_sweep_interval_seconds=get_nested(config, "feedback.sweep_interval_seconds", 5.0),
        timeout_reward=get_nested(config, "feedback.timeout_reward", 0.0),
        latency_ema_alpha=get_nested(config, "feedback.latency_ema_alpha", 0.2),
        # Debug
        debug_traces=get_nested(config, "debug.traces_enabled", False),
        debug_trace_dir=get_nested(config, "debug.trace_dir", "/tmp/dynamo_router_traces"),
        debug_buffer_size=get_nested(config, "debug.buffer_size", 2000),
    )
    await router.initialize()

    # Serve both endpoints
    await asyncio.gather(
        component.endpoint("find_worker").serve_endpoint(router.generate),
        component.endpoint("feedback").serve_endpoint(router.feedback),
    )


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
