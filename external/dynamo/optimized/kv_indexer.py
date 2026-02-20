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
KV Cache Indexer for the custom Thompson Sampling router.

Provides real-time KV cache overlap scoring by subscribing to workers' ZMQ
KV event streams and maintaining a local radix tree. Backend-agnostic: works
identically with SGLang and vLLM workers.

Data flow::

    Worker (SGLang/vLLM)
      │  publishes KvCacheEvent via ZMQ (Stored/Removed/Cleared)
      ▼
    ZmqKvEventListener (per worker)
      │  receives msgpack frames, exposes as JSON strings
      ▼
    RadixTree.apply_event(worker_id, event_bytes)
      │  updates local radix tree with per-worker block state
      ▼
    compute_block_hash_for_seq(tokens, block_size)
      │  hashes request tokens into block-level hashes
      ▼
    RadixTree.find_matches(block_hashes) → OverlapScores
      │  returns {(worker_id, dp_rank): matching_blocks}
      ▼
    KvIndexer.find_matches_for_request() → OverlapScores (normalised)
      │  converts to {worker_id: fraction} for the router

All building blocks (RadixTree, ZmqKvEventListener, compute_block_hash_for_seq,
OverlapScores) are from ``dynamo.llm`` and use the shared ``KvCacheEvent``
protocol defined in ``dynamo-kv-router/src/protocols.rs``.

Usage::

    from kv_indexer import KvIndexer, OverlapScores

    indexer = KvIndexer(block_size=64)
    indexer.add_worker(worker_id=123, zmq_endpoint="tcp://10.0.0.1:20080")
    scores = await indexer.find_matches_for_request(token_ids, min_overlap=0)
    overlap = scores.scores.get(123, 0.0)  # float in [0, 1]
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import real Dynamo building blocks, with graceful fallback
# ---------------------------------------------------------------------------
_HAS_DYNAMO_KV = False
try:
    from dynamo.llm import RadixTree as _RadixTree
    from dynamo.llm import ZmqKvEventListener as _ZmqKvEventListener

    # The function name varies across Dynamo image versions:
    #   0.9.0 (NGC): compute_block_hash_for_seq_py
    #   main branch:  compute_block_hash_for_seq
    try:
        from dynamo.llm import compute_block_hash_for_seq as _compute_block_hash_for_seq
    except ImportError:
        from dynamo.llm import compute_block_hash_for_seq_py as _compute_block_hash_for_seq

    _HAS_DYNAMO_KV = True
    logger.info("kv_indexer: dynamo.llm KV primitives imported successfully")
except ImportError as exc:
    logger.warning(
        "kv_indexer: dynamo.llm KV primitives not available (%s); "
        "KvIndexer will return empty overlap scores (fallback mode)",
        exc,
    )


# ---------------------------------------------------------------------------
# OverlapScores wrapper
# ---------------------------------------------------------------------------
class OverlapScores:
    """Normalised overlap scores compatible with the router's interface.

    The router accesses ``scores.scores.get(worker_id, 0.0)`` and expects
    a ``float`` in [0, 1] representing the fraction of the request's KV
    blocks that are already cached on a given worker.
    """

    def __init__(self, scores: dict[int, float] | None = None):
        self.scores: dict[int, float] = scores if scores is not None else {}

    def __repr__(self) -> str:
        return f"OverlapScores({self.scores})"


# ---------------------------------------------------------------------------
# KvIndexer
# ---------------------------------------------------------------------------
class KvIndexer:
    """KV cache indexer using Dynamo's RadixTree + ZmqKvEventListener.

    Backend-agnostic: works with both SGLang (``--page-size``) and vLLM
    (``--block-size``) workers as long as ``block_size`` matches the value
    passed to the workers.

    Parameters
    ----------
    engine:
        Dynamo runtime ``Component`` reference (``workers.<component>``).
        Accepted for API compatibility with the original ``KvIndexer``
        constructor but unused — worker discovery is handled via
        ``add_worker()`` or ``discover_workers()``.
    block_size:
        KV cache block size in tokens.  Must match the backend worker
        configuration (``--page-size`` for SGLang, ``--block-size`` for vLLM).
    """

    def __init__(self, engine: Any, block_size: int):
        self._engine = engine  # kept for interface compatibility
        self.block_size = block_size
        self._listeners: dict[int, Any] = {}  # worker_id -> ZmqKvEventListener
        self._radix_tree: Any | None = None
        self._poll_task: asyncio.Task | None = None  # background drain loop

        if _HAS_DYNAMO_KV:
            self._radix_tree = _RadixTree()
            logger.info("KvIndexer initialised with RadixTree (block_size=%d)", block_size)
        else:
            logger.warning("KvIndexer running in fallback mode (no RadixTree); "
                           "overlap scores will always be 0")

    # ------------------------------------------------------------------
    # Worker registration
    # ------------------------------------------------------------------

    def add_worker(self, worker_id: int, zmq_endpoint: str) -> None:
        """Register a worker's ZMQ KV event stream.

        Parameters
        ----------
        worker_id:
            Dynamo worker instance ID (from ETCD discovery).
        zmq_endpoint:
            ZMQ endpoint to subscribe to, e.g. ``tcp://10.0.0.1:20080``.
        """
        if not _HAS_DYNAMO_KV:
            return
        if worker_id in self._listeners:
            logger.debug("Worker %s already registered in KvIndexer; skipping", worker_id)
            return
        listener = _ZmqKvEventListener(zmq_endpoint, "", self.block_size)
        self._listeners[worker_id] = listener
        logger.info("KvIndexer: registered worker %s at %s", worker_id, zmq_endpoint)

    def discover_workers(self, kv_event_base_port: int | None = None) -> None:
        """Auto-discover workers from the engine client and register listeners.

        Uses the ``KV_EVENT_BASE_PORT`` environment variable (or the explicit
        *kv_event_base_port* argument) to compute per-worker ZMQ endpoints.

        Workers are assumed to use sequential ports starting at the base port:
        ``worker_index 0 → base_port``, ``worker_index 1 → base_port + 1``, etc.
        The ordering follows the order returned by ``engine_client.instance_ids()``.
        """
        if not _HAS_DYNAMO_KV:
            return

        if kv_event_base_port is None:
            kv_event_base_port = int(os.environ.get("KV_EVENT_BASE_PORT", "20080"))

        try:
            instance_ids = [int(wid) for wid in self._engine.endpoint("generate").client_sync().instance_ids()]
        except Exception:
            logger.warning("KvIndexer.discover_workers: could not list instances from engine client; "
                           "call add_worker() manually instead")
            return

        for idx, wid in enumerate(sorted(instance_ids)):
            endpoint = f"tcp://127.0.0.1:{kv_event_base_port + idx}"
            self.add_worker(wid, endpoint)

    # ------------------------------------------------------------------
    # Background event drain
    # ------------------------------------------------------------------

    def start_background_drain(self, interval: float = 0.1) -> None:
        """Start an asyncio task that continuously drains KV events.

        This keeps the radix tree up-to-date between routing decisions so
        that ``find_matches_for_request`` doesn't have to drain inline.
        """
        if self._poll_task is not None:
            return
        self._poll_task = asyncio.create_task(self._drain_loop(interval))
        logger.info("KvIndexer: started background drain (interval=%.2fs)", interval)

    async def _drain_loop(self, interval: float) -> None:
        """Internal loop that periodically drains all listeners."""
        while True:
            try:
                await self._drain_events()
            except Exception:
                logger.exception("KvIndexer: error draining KV events")
            await asyncio.sleep(interval)

    async def _drain_events(self) -> int:
        """Poll all listeners and feed events into the radix tree.

        Returns the total number of events applied.
        """
        if not _HAS_DYNAMO_KV or self._radix_tree is None:
            return 0

        total = 0
        for worker_id, listener in self._listeners.items():
            try:
                events = await listener.get_events()
            except Exception:
                logger.exception("KvIndexer: failed to get events from worker %s", worker_id)
                continue

            for event_json in events:
                try:
                    self._radix_tree.apply_event(worker_id, event_json.encode("utf-8"))
                    total += 1
                except Exception:
                    logger.exception(
                        "KvIndexer: failed to apply event from worker %s",
                        worker_id,
                    )
        return total

    # ------------------------------------------------------------------
    # Overlap query (called by the router)
    # ------------------------------------------------------------------

    async def find_matches_for_request(self, tokens: list[int], min_overlap: int) -> OverlapScores:
        """Compute per-worker overlap scores for a token sequence.

        Returns an ``OverlapScores`` object whose ``.scores`` dict maps
        ``worker_id → float`` in [0, 1] representing the fraction of the
        request's KV blocks already cached on that worker.
        """
        if not _HAS_DYNAMO_KV or self._radix_tree is None:
            return OverlapScores({})

        # If no background drain is running, drain inline
        if self._poll_task is None:
            await self._drain_events()

        # Hash token sequence into block-level hashes
        block_hashes = _compute_block_hash_for_seq(tokens, self.block_size)
        if not block_hashes:
            return OverlapScores({})

        total_blocks = len(block_hashes)

        # Query the radix tree
        raw_scores = self._radix_tree.find_matches(block_hashes)

        # raw_scores.scores is dict[(worker_id, dp_rank), count] from Rust.
        # Normalise to dict[worker_id, float] for the router.
        normalised: dict[int, float] = {}
        for key, count in raw_scores.scores.items():
            # Handle both (worker_id, dp_rank) tuple keys and plain int keys
            if isinstance(key, tuple):
                wid = int(key[0])
            else:
                wid = int(key)
            frac = float(count) / float(total_blocks)
            # Keep the best score if a worker appears with multiple dp_ranks
            if frac > normalised.get(wid, 0.0):
                normalised[wid] = frac

        return OverlapScores(normalised)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Cancel background drain and release resources."""
        if self._poll_task is not None:
            self._poll_task.cancel()
            self._poll_task = None
            logger.info("KvIndexer: background drain stopped")
