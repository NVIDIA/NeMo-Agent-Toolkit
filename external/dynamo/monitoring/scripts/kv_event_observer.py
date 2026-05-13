#!/usr/bin/env python3
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
KV Cache Event Observer for Dynamo vLLM Workers

Subscribes to vLLM's ZMQ KV event publisher and logs/monitors block-level
events (stored, evicted) in real-time. Also polls Prometheus metrics to
detect cache hits (which don't generate ZMQ events).

vLLM publishes events in msgpack format via ZMQ multipart messages:
  - Part 0: Topic (bytes, usually empty)
  - Part 1: Sequence number (8 bytes, big-endian int64)
  - Part 2: Payload (msgpack-encoded KVEventBatch)

KVEventBatch structure (msgpack):
  [timestamp, events_list, dp_rank]

Event types (from ZMQ):
  - BlockStored: A new block was committed to prefix cache
  - BlockRemoved: A block was evicted from prefix cache
  - AllBlocksCleared: Entire cache was cleared

Metrics polling (for cache hits):
  - vllm:prefix_cache_hits_total: Cumulative cache hit tokens
  - vllm:prefix_cache_queries_total: Cumulative cache query tokens

Usage:
    # Inside container:
    python /workspace/monitoring/scripts/kv_event_observer.py --port 20080 --verbose

    # With cache hit tracking (polls metrics endpoint):
    python /workspace/monitoring/scripts/kv_event_observer.py -p 20080 -v --metrics-port 18081

    # Output to file:
    python kv_event_observer.py --port 20080 --verbose --output kv_events.jsonl
"""

import argparse
import json
import re
import signal
import sys
import threading
import time
import urllib.request
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from typing import Any

try:
    import zmq
except ImportError:
    print("ERROR: pyzmq not installed. Run: pip install pyzmq")
    sys.exit(1)

try:
    import msgpack
except ImportError:
    print("ERROR: msgpack not installed. Run: pip install msgpack")
    sys.exit(1)


def format_hash(block_hash: Any) -> str:
    """Format a block hash for display."""
    if isinstance(block_hash, bytes):
        return block_hash.hex()[:16]
    elif isinstance(block_hash, int):
        return f"{block_hash:016x}"[:16]
    return str(block_hash)[:16]


@dataclass
class KVCacheStats:
    """Aggregated statistics for KV cache events."""
    stored_blocks: int = 0
    evicted_blocks: int = 0
    cleared_count: int = 0
    cache_hit_tokens: int = 0  # Tokens served from cache (from metrics)
    cache_query_tokens: int = 0  # Total tokens queried (from metrics)
    unique_hashes: set = field(default_factory=set)
    hash_to_blocks: dict = field(default_factory=lambda: defaultdict(list))
    last_event_time: float = 0.0
    last_seq: int = -1

    def record_stored(self, block_hashes: list[Any], parent_hash: Any = None):
        """Record BlockStored event."""
        self.last_event_time = time.time()
        for bh in block_hashes:
            h = format_hash(bh)
            self.stored_blocks += 1
            self.unique_hashes.add(h)

    def record_removed(self, block_hashes: list[Any]):
        """Record BlockRemoved event."""
        self.last_event_time = time.time()
        for bh in block_hashes:
            h = format_hash(bh)
            self.evicted_blocks += 1
            self.unique_hashes.discard(h)

    def record_cleared(self):
        """Record AllBlocksCleared event."""
        self.last_event_time = time.time()
        self.cleared_count += 1
        self.unique_hashes.clear()

    def record_cache_hit(self, hit_tokens: int, query_tokens: int):
        """Record cache hit from metrics delta."""
        self.cache_hit_tokens += hit_tokens
        self.cache_query_tokens += query_tokens

    def summary(self) -> dict:
        """Return summary statistics."""
        hit_rate = (self.cache_hit_tokens / self.cache_query_tokens * 100) if self.cache_query_tokens > 0 else 0
        return {
            "stored_blocks": self.stored_blocks,
            "evicted_blocks": self.evicted_blocks,
            "net_blocks": self.stored_blocks - self.evicted_blocks,
            "cleared_count": self.cleared_count,
            "unique_hashes_current": len(self.unique_hashes),
            "cache_hit_tokens": self.cache_hit_tokens,
            "cache_query_tokens": self.cache_query_tokens,
            "cache_hit_rate": f"{hit_rate:.1f}%",
            "last_seq": self.last_seq,
        }


@dataclass
class EfficiencySample:
    """A single efficiency measurement sample."""
    timestamp: float
    hit_tokens: int
    query_tokens: int


class SlidingWindowEfficiency:
    """Computes KV cache efficiency over a sliding time window.

    Efficiency (work done fraction) measures what fraction of prompt tokens
    were served from the KV cache rather than being recomputed:

        efficiency = cached_tokens / total_queried_tokens * 100

    Only samples within the configured time window are considered. This
    provides a responsive metric that reflects recent cache behavior rather
    than a lifetime average that never recovers from early cold-cache misses.

    Interpretation:
      - 0%:   No cache reuse. All tokens required fresh computation.
      - 100%: Perfect reuse. All tokens served from cache.
    """

    def __init__(self, window_seconds: float = 30.0):
        self.window_seconds = window_seconds
        self._samples: deque[EfficiencySample] = deque()

    def add_sample(self, hit_tokens: int, query_tokens: int, timestamp: float | None = None):
        """Add a measurement sample to the window."""
        ts = timestamp if timestamp is not None else time.time()
        self._samples.append(EfficiencySample(ts, hit_tokens, query_tokens))
        self._evict_old(ts)

    def _evict_old(self, now: float):
        """Remove samples that have fallen outside the window."""
        cutoff = now - self.window_seconds
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()

    def get_efficiency(self) -> float:
        """Get current efficiency as a percentage (0-100).

        Returns 0.0 when the window is empty (no traffic).
        """
        self._evict_old(time.time())
        total_hits = sum(s.hit_tokens for s in self._samples)
        total_queries = sum(s.query_tokens for s in self._samples)
        if total_queries == 0:
            return 0.0
        return (total_hits / total_queries) * 100.0

    @property
    def sample_count(self) -> int:
        """Number of samples currently in the window."""
        self._evict_old(time.time())
        return len(self._samples)

    def reset(self):
        """Clear all samples."""
        self._samples.clear()


class KVEventObserver:
    """Observes KV cache events from a vLLM worker via ZMQ.

    Also optionally polls Prometheus metrics to detect cache hits
    (which don't generate ZMQ events) and computes KV cache efficiency
    over a sliding window using the same histogram metrics as the
    Grafana dashboard recording rule (rate over request_prompt_tokens_sum
    and request_prefill_kv_computed_tokens_sum).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 20080,
        verbose: bool = False,
        output_file: str | None = None,
        metrics_port: int | None = None,
        window_seconds: float = 30.0,
    ):
        self.host = host
        self.port = port
        self.verbose = verbose
        self.output_file = output_file
        self.metrics_port = metrics_port
        self.stats = KVCacheStats()
        self.running = False
        self._output_handle = None

        # Sliding window efficiency tracker (mirrors the recording rule approach)
        self.efficiency = SlidingWindowEfficiency(window_seconds)

        # Counter-based metrics polling state (prefix_cache_hits/queries)
        self._last_hits = 0.0
        self._last_queries = 0.0

        # Histogram-based metrics polling state (matches vllm:cache_hit_rate rule)
        self._last_prompt_sum = 0.0
        self._last_computed_sum = 0.0

        self._metrics_thread = None

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)

    def _parse_metric(self, metrics_text: str, metric_name: str) -> float:
        """Extract and sum all instances of a metric from Prometheus text format.

        Handles metrics with or without labels and sums across all label
        combinations (e.g. multiple model_name or instance labels).
        """
        pattern = rf'^{re.escape(metric_name)}(?:\{{[^}}]*\}})?\s+([0-9.e+-]+)'
        total = 0.0
        for line in metrics_text.split('\n'):
            match = re.match(pattern, line)
            if match:
                total += float(match.group(1))
        return total

    def _poll_metrics(self):
        """Background thread to poll Prometheus metrics.

        Tracks two complementary views:
        1. Counter-based: vllm:prefix_cache_hits_total / queries_total
           (legacy, per-scheduler-step granularity)
        2. Histogram-based: request_prompt_tokens_sum / request_prefill_kv_computed_tokens_sum
           (matches the vllm:cache_hit_rate recording rule used by the Grafana dashboard)

        The histogram approach feeds the SlidingWindowEfficiency tracker so
        the observer's efficiency % matches what the dashboard shows.
        """
        metrics_url = f"http://{self.host}:{self.metrics_port}/metrics"

        while self.running:
            try:
                with urllib.request.urlopen(metrics_url, timeout=2) as resp:
                    metrics_text = resp.read().decode('utf-8')

                # --- Counter-based cache hits (legacy) ---
                hits = self._parse_metric(metrics_text, 'vllm:prefix_cache_hits_total')
                queries = self._parse_metric(metrics_text, 'vllm:prefix_cache_queries_total')

                hit_delta = hits - self._last_hits
                query_delta = queries - self._last_queries

                if hit_delta > 0:
                    self.stats.record_cache_hit(int(hit_delta), int(query_delta))
                    if self.verbose:
                        hit_rate = (hit_delta / query_delta * 100) if query_delta > 0 else 0
                        print(f"✅ [CACHE HIT] tokens={int(hit_delta):4d} "
                              f"queried={int(query_delta):4d} hit_rate={hit_rate:.0f}%")
                elif query_delta > 0:
                    self.stats.record_cache_hit(0, int(query_delta))

                self._last_hits = hits
                self._last_queries = queries

                # --- Histogram-based efficiency (matches recording rule) ---
                # Same formula as vllm:cache_hit_rate:
                #   cached = prompt_tokens - kv_computed_tokens
                #   efficiency = cached / prompt_tokens
                prompt_sum = self._parse_metric(metrics_text, 'vllm:request_prompt_tokens_sum')
                computed_sum = self._parse_metric(metrics_text, 'vllm:request_prefill_kv_computed_tokens_sum')

                prompt_delta = prompt_sum - self._last_prompt_sum
                computed_delta = computed_sum - self._last_computed_sum

                if prompt_delta > 0:
                    cached_delta = prompt_delta - computed_delta
                    self.efficiency.add_sample(int(max(0, cached_delta)), int(prompt_delta))
                    if self.verbose:
                        eff = self.efficiency.get_efficiency()
                        print(f"📊 [EFFICIENCY] {eff:.1f}% "
                              f"(cached={cached_delta:.0f} prompt={prompt_delta:.0f} "
                              f"window={self.efficiency.window_seconds}s "
                              f"samples={self.efficiency.sample_count})")

                self._last_prompt_sum = prompt_sum
                self._last_computed_sum = computed_sum

            except Exception as e:
                if self.verbose:
                    print(f"[Metrics] Poll error: {e}")

            time.sleep(0.5)  # Poll every 500ms

    def connect(self):
        """Connect to the vLLM KV event publisher."""
        endpoint = f"tcp://{self.host}:{self.port}"
        print(f"[KV Observer] Connecting to {endpoint}...")
        self.socket.connect(endpoint)
        # Subscribe to all topics (empty string = all)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)
        print("[KV Observer] ✓ Connected and subscribed")

        if self.output_file:
            self._output_handle = open(self.output_file, "a")
            print(f"[KV Observer] Writing events to: {self.output_file}")

        if self.metrics_port:
            print(f"[KV Observer] Polling metrics at http://{self.host}:{self.metrics_port}/metrics")
            print(f"[KV Observer] Efficiency window: {self.efficiency.window_seconds}s")
            # Initialize baseline metrics
            try:
                metrics_url = f"http://{self.host}:{self.metrics_port}/metrics"
                with urllib.request.urlopen(metrics_url, timeout=2) as resp:
                    metrics_text = resp.read().decode('utf-8')
                self._last_hits = self._parse_metric(metrics_text, 'vllm:prefix_cache_hits_total')
                self._last_queries = self._parse_metric(metrics_text, 'vllm:prefix_cache_queries_total')
                self._last_prompt_sum = self._parse_metric(metrics_text, 'vllm:request_prompt_tokens_sum')
                self._last_computed_sum = self._parse_metric(metrics_text,
                                                             'vllm:request_prefill_kv_computed_tokens_sum')
                print(f"[KV Observer] ✓ Baseline: hits={self._last_hits:.0f} "
                      f"queries={self._last_queries:.0f} "
                      f"prompt_sum={self._last_prompt_sum:.0f} "
                      f"computed_sum={self._last_computed_sum:.0f}")
            except Exception as e:
                print(f"[KV Observer] ⚠ Could not get baseline metrics: {e}")

    def parse_multipart(self, parts: list[bytes]) -> dict | None:
        """Parse a ZMQ multipart message from vLLM.

        Format: [topic, sequence, payload]
        Payload is msgpack-encoded KVEventBatch: [timestamp, events_list, dp_rank]

        Note: The order is [ts, events, dp_rank], NOT [ts, dp_rank, events]!
        """
        if len(parts) < 3:
            if self.verbose:
                print(f"[KV Observer] Warning: Expected 3 parts, got {len(parts)}")
            return None

        topic, seq_bytes, payload = parts[0], parts[1], parts[2]

        try:
            seq = int.from_bytes(seq_bytes, "big", signed=True)
            self.stats.last_seq = seq
        except Exception:
            seq = -1

        try:
            # Decode msgpack payload
            batch = msgpack.unpackb(payload, raw=False, strict_map_key=False)

            # vLLM KVEventBatch format: [timestamp, events_list, dp_rank]
            # Note: events is at index 1, dp_rank at index 2!
            if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                ts = batch[0]
                events = batch[1]  # Events are at index 1
                dp_rank = batch[2]  # dp_rank is at index 2
            elif isinstance(batch, dict):
                ts = batch.get("ts", time.time())
                dp_rank = batch.get("data_parallel_rank", 0)
                events = batch.get("events", [])
            else:
                events = [batch] if batch else []
                ts = time.time()
                dp_rank = 0

            # Ensure events is a list
            if not isinstance(events, list):
                events = [events] if events else []

            return {
                "seq": seq,
                "timestamp": ts,
                "dp_rank": dp_rank,
                "events": events,
                "topic": topic.decode("utf-8", errors="replace") if topic else "",
            }
        except Exception as e:
            if self.verbose:
                print(f"[KV Observer] Parse error: {e}")
                print(f"[KV Observer]   Raw payload: {payload[:100]}...")
            return None

    def handle_event(self, event_data: dict):
        """Handle a parsed event batch."""
        seq = event_data.get("seq", -1)
        ts = event_data.get("timestamp", 0)
        dp_rank = event_data.get("dp_rank", 0)
        events = event_data.get("events", [])

        for event in events:
            # Events can be dicts or tuples/lists
            # vLLM format (list):
            #   BlockRemoved: ['BlockRemoved', [hash_list], medium]
            #   BlockStored:  ['BlockStored', [hash_list], parent_hash, token_ids, block_size, lora_id, medium]
            #   AllBlocksCleared: ['AllBlocksCleared']
            if isinstance(event, dict):
                event_type = event.get("type", event.get("event_type", "unknown"))
                block_hashes = event.get("block_hashes", [])
                parent_hash = event.get("parent_block_hash")
                medium = event.get("medium", "GPU")
                token_ids = event.get("token_ids", [])
                block_size = event.get("block_size", 0)
            elif isinstance(event, (list, tuple)) and len(event) >= 1:
                event_type = str(event[0]) if event else "unknown"

                if event_type == "BlockRemoved" and len(event) >= 2:
                    # ['BlockRemoved', [hashes], medium]
                    block_hashes = event[1] if isinstance(event[1], list) else [event[1]]
                    medium = event[2] if len(event) > 2 else "GPU"
                    parent_hash = None
                    token_ids = []
                    block_size = 0
                elif event_type == "BlockStored" and len(event) >= 2:
                    # ['BlockStored', [hashes], parent_hash, token_ids, block_size, lora_id, medium]
                    block_hashes = event[1] if isinstance(event[1], list) else [event[1]]
                    parent_hash = event[2] if len(event) > 2 else None
                    token_ids = event[3] if len(event) > 3 else []
                    block_size = event[4] if len(event) > 4 else 0
                    medium = event[6] if len(event) > 6 else "GPU"
                elif event_type == "AllBlocksCleared":
                    block_hashes = []
                    parent_hash = None
                    medium = "GPU"
                    token_ids = []
                    block_size = 0
                else:
                    block_hashes = event[1] if len(event) > 1 and isinstance(event[1], list) else []
                    parent_hash = None
                    medium = event[-1] if len(event) > 2 and isinstance(event[-1], str) else "GPU"
                    token_ids = []
                    block_size = 0
            else:
                event_type = str(type(event).__name__)
                block_hashes = []
                parent_hash = None
                medium = "GPU"
                token_ids = []
                block_size = 0

            # Normalize event type (vLLM uses class names like "BlockStored")
            event_type_lower = event_type.lower()

            if "stored" in event_type_lower or "blockstored" in event_type_lower:
                self.stats.record_stored(block_hashes, parent_hash)
                if self.verbose:
                    num_tokens = len(token_ids) if token_ids else block_size
                    for bh in block_hashes:
                        print(
                            f"📦 [STORED  ] seq={seq:6d} hash={format_hash(bh)} tokens={num_tokens:3d} medium={medium}")
            elif "removed" in event_type_lower or "blockremoved" in event_type_lower:
                self.stats.record_removed(block_hashes)
                if self.verbose:
                    for bh in block_hashes:
                        print(f"🗑️  [REMOVED ] seq={seq:6d} hash={format_hash(bh)} medium={medium}")
            elif "cleared" in event_type_lower or "allblockscleared" in event_type_lower:
                self.stats.record_cleared()
                if self.verbose:
                    print(f"🧹 [CLEARED ] seq={seq:6d} All blocks cleared")
            elif self.verbose:
                print(f"❓ [UNKNOWN ] seq={seq:6d} type={event_type} "
                      f"data={event[:3] if isinstance(event, (list, tuple)) else event}")

        # Write to output file
        if self._output_handle:

            def get_event_type(e):
                if isinstance(e, dict):
                    return str(e.get("type", "unknown"))
                elif isinstance(e, (list, tuple)) and len(e) > 0:
                    return str(e[0])
                else:
                    return str(e)

            output = {
                "_timestamp": datetime.now(UTC).isoformat(),
                "seq": seq,
                "ts": ts,
                "dp_rank": dp_rank,
                "events": [{
                    "type": get_event_type(e)
                } for e in events],
            }
            self._output_handle.write(json.dumps(output) + "\n")
            self._output_handle.flush()

    def run(self, duration: float | None = None):
        """Run the observer loop."""
        self.running = True
        start_time = time.time()
        batches_received = 0

        # Start metrics polling thread if configured
        if self.metrics_port:
            self._metrics_thread = threading.Thread(target=self._poll_metrics, daemon=True, name="metrics-poller")
            self._metrics_thread.start()

        print("[KV Observer] Listening for KV events (msgpack multipart)...")
        if self.metrics_port:
            print("[KV Observer] Cache hits will show as ✅ [CACHE HIT]")
        print("[KV Observer] Press Ctrl+C to stop")
        print("-" * 60)

        try:
            while self.running:
                if duration and (time.time() - start_time) >= duration:
                    print(f"\n[KV Observer] Duration limit reached ({duration}s)")
                    break

                try:
                    # Receive multipart message
                    parts = self.socket.recv_multipart()
                    event_data = self.parse_multipart(parts)

                    if event_data:
                        self.handle_event(event_data)
                        batches_received += 1

                        if batches_received % 20 == 0 and not self.verbose:
                            summary = self.stats.summary()
                            eff = self.efficiency.get_efficiency()
                            print(f"[{batches_received:5d} batches] "
                                  f"Stored: {summary['stored_blocks']:4d} | "
                                  f"Removed: {summary['evicted_blocks']:4d} | "
                                  f"Net: {summary['net_blocks']:4d} | "
                                  f"Efficiency: {eff:.1f}% | "
                                  f"Seq: {summary['last_seq']}")
                except zmq.Again:
                    # Timeout, continue loop
                    continue

        except KeyboardInterrupt:
            print("\n[KV Observer] Interrupted")
        finally:
            self.stop()

    def stop(self):
        """Stop and print final statistics."""
        self.running = False

        print("-" * 60)
        print("[KV Observer] Final Statistics:")
        for key, value in self.stats.summary().items():
            print(f"  {key}: {value}")
        eff = self.efficiency.get_efficiency()
        n = self.efficiency.sample_count
        print(f"  window_efficiency: {eff:.1f}% ({n} samples in {self.efficiency.window_seconds}s window)")

        if self._output_handle:
            self._output_handle.close()

        self.socket.close()
        self.context.term()
        print("[KV Observer] Stopped")


def run_self_test():
    """Validate the SlidingWindowEfficiency calculation with known scenarios.

    Test 1 – Zero reuse:   all tokens recomputed → efficiency must be 0%.
    Test 2 – Perfect reuse: all tokens cached    → efficiency must be 100%.

    These are deterministic unit-style tests that exercise the same sliding-
    window logic that the observer uses when polling live histogram metrics.
    They do NOT require a running vLLM worker.
    """
    print("=" * 60)
    print("KV Event Observer – Sliding Window Efficiency Self-Test")
    print("=" * 60)

    passed = 0
    failed = 0

    # ------------------------------------------------------------------
    # Test 1: No KV cache reuse (efficiency = 0%)
    # Every sample: hit_tokens=0, query_tokens>0  →  0 / total = 0%
    # ------------------------------------------------------------------
    print("\n--- Test 1: No KV Cache Reuse (expect 0% efficiency) ---")
    eff = SlidingWindowEfficiency(window_seconds=10.0)
    now = time.time()
    for i in range(10):
        # All tokens were recomputed, zero served from cache
        eff.add_sample(hit_tokens=0, query_tokens=100, timestamp=now + i)
    result = eff.get_efficiency()
    if abs(result) < 0.01:
        print(f"  ✅ PASS: efficiency = {result:.1f}% (expected 0%)")
        passed += 1
    else:
        print(f"  ❌ FAIL: efficiency = {result:.1f}% (expected 0%)")
        failed += 1

    # ------------------------------------------------------------------
    # Test 2: Perfect KV cache reuse (efficiency = 100%)
    # Every sample: hit_tokens == query_tokens  →  total / total = 100%
    # ------------------------------------------------------------------
    print("\n--- Test 2: Perfect KV Cache Reuse (expect 100% efficiency) ---")
    eff = SlidingWindowEfficiency(window_seconds=10.0)
    now = time.time()
    for i in range(10):
        # All tokens served from cache
        eff.add_sample(hit_tokens=100, query_tokens=100, timestamp=now + i)
    result = eff.get_efficiency()
    if abs(result - 100.0) < 0.01:
        print(f"  ✅ PASS: efficiency = {result:.1f}% (expected 100%)")
        passed += 1
    else:
        print(f"  ❌ FAIL: efficiency = {result:.1f}% (expected 100%)")
        failed += 1

    # ------------------------------------------------------------------
    # Test 3: 50% reuse
    # ------------------------------------------------------------------
    print("\n--- Test 3: 50% Reuse (expect 50% efficiency) ---")
    eff = SlidingWindowEfficiency(window_seconds=10.0)
    now = time.time()
    for i in range(10):
        eff.add_sample(hit_tokens=50, query_tokens=100, timestamp=now + i)
    result = eff.get_efficiency()
    if abs(result - 50.0) < 0.01:
        print(f"  ✅ PASS: efficiency = {result:.1f}% (expected 50%)")
        passed += 1
    else:
        print(f"  ❌ FAIL: efficiency = {result:.1f}% (expected 50%)")
        failed += 1

    # ------------------------------------------------------------------
    # Test 4: Window eviction – old zero-reuse samples expire, only
    # recent perfect-reuse samples remain → efficiency should be 100%.
    # This proves the sliding window correctly drops stale data.
    # ------------------------------------------------------------------
    print("\n--- Test 4: Window Eviction (old 0% samples expire → 100%) ---")
    eff = SlidingWindowEfficiency(window_seconds=5.0)
    base = time.time()
    # Old samples (before window): zero reuse — should be evicted
    for i in range(5):
        eff.add_sample(hit_tokens=0, query_tokens=100, timestamp=base - 10 + i)
    # Recent samples (inside window): perfect reuse
    for i in range(5):
        eff.add_sample(hit_tokens=100, query_tokens=100, timestamp=base + i)
    result = eff.get_efficiency()
    if abs(result - 100.0) < 0.01:
        print(f"  ✅ PASS: efficiency = {result:.1f}% (old zero-reuse samples evicted)")
        passed += 1
    else:
        print(f"  ❌ FAIL: efficiency = {result:.1f}% (expected 100% after eviction)")
        failed += 1

    # ------------------------------------------------------------------
    # Test 5: Empty window → 0%
    # ------------------------------------------------------------------
    print("\n--- Test 5: Empty Window (expect 0% efficiency) ---")
    eff = SlidingWindowEfficiency(window_seconds=10.0)
    result = eff.get_efficiency()
    if abs(result) < 0.01:
        print(f"  ✅ PASS: efficiency = {result:.1f}% (no samples)")
        passed += 1
    else:
        print(f"  ❌ FAIL: efficiency = {result:.1f}% (expected 0%)")
        failed += 1

    # ------------------------------------------------------------------
    # Test 6: Weighted mix – requests of different sizes.
    # 300 tokens all cached + 100 tokens none cached → 300/400 = 75%
    # ------------------------------------------------------------------
    print("\n--- Test 6: Weighted Mix (expect 75% efficiency) ---")
    eff = SlidingWindowEfficiency(window_seconds=10.0)
    now = time.time()
    eff.add_sample(hit_tokens=300, query_tokens=300, timestamp=now)
    eff.add_sample(hit_tokens=0, query_tokens=100, timestamp=now + 1)
    result = eff.get_efficiency()
    if abs(result - 75.0) < 0.01:
        print(f"  ✅ PASS: efficiency = {result:.1f}% (300/400 weighted)")
        passed += 1
    else:
        print(f"  ❌ FAIL: efficiency = {result:.1f}% (expected 75%)")
        failed += 1

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)

    return failed == 0


def run_live_test(api_url: str, metrics_url: str, model: str, num_unique: int = 5, num_repeat: int = 10):
    """End-to-end integration test against a running vLLM worker.

    Sends controlled request patterns and verifies the histogram-based
    efficiency from the actual Prometheus metrics:

      Test 1: Unique random prompts  → efficiency should be ~0%
      Test 2: Identical repeated prompt → efficiency should be ~100%

    This reads the same vllm:request_prompt_tokens_sum and
    vllm:request_prefill_kv_computed_tokens_sum histograms that the
    recording rule uses, so a PASS here means the Grafana dashboard
    will show the correct values.

    Requirements:
      - A running vLLM worker with prefix caching enabled
      - The worker's Prometheus metrics endpoint reachable at metrics_url
      - An API endpoint (OpenAI-compatible) reachable at api_url

    NOTE: Run this when the worker is idle (no other traffic), otherwise
    concurrent requests will pollute the deltas.
    """
    import random
    import string
    import uuid

    def _parse_metric_sum(text: str, name: str) -> float:
        pattern = rf'^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.e+-]+)'
        total = 0.0
        for line in text.split('\n'):
            match = re.match(pattern, line)
            if match:
                total += float(match.group(1))
        return total

    def poll_histograms() -> tuple[float, float]:
        """Return (prompt_tokens_sum, kv_computed_tokens_sum) from the worker."""
        url = metrics_url.rstrip('/') + '/metrics'
        with urllib.request.urlopen(url, timeout=5) as resp:
            text = resp.read().decode()
        p = _parse_metric_sum(text, 'vllm:request_prompt_tokens_sum')
        c = _parse_metric_sum(text, 'vllm:request_prefill_kv_computed_tokens_sum')
        return p, c

    def send_completion(prompt: str, max_tokens: int = 1):
        """Send one completion request (max_tokens=1 so we only measure prefill)."""
        url = api_url.rstrip('/') + '/v1/completions'
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }).encode()
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())

    def random_prompt(num_words: int = 80) -> str:
        """Generate a random prompt that shares no prefix with any other."""
        uid = uuid.uuid4().hex
        words = [''.join(random.choices(string.ascii_lowercase, k=random.randint(4, 8))) for _ in range(num_words)]
        return f"[{uid}] " + " ".join(words)

    # A long deterministic prompt for the perfect-reuse test.
    # ~200 tokens → ~12 cache blocks at block_size=16.
    reuse_prompt = ("The following is a detailed technical explanation about how KV cache "
                    "works in transformer-based large language models. The key-value cache "
                    "stores previously computed attention keys and values to avoid redundant "
                    "computation during autoregressive generation. When a new token is generated, "
                    "the model only needs to compute the key and value for the new token, "
                    "while reusing the cached keys and values from all previous tokens. "
                    "This significantly reduces the computational cost of generation, "
                    "especially for long sequences. The KV cache typically grows linearly "
                    "with the sequence length and the number of attention layers. "
                    "Prefix caching extends this concept by sharing KV cache entries across "
                    "multiple requests that share a common prompt prefix, which is extremely "
                    "beneficial for workloads with repeated system prompts or similar queries.")

    print("=" * 60)
    print("KV Cache Efficiency – Live Integration Test")
    print("=" * 60)
    print(f"  API:     {api_url}")
    print(f"  Metrics: {metrics_url}")
    print(f"  Model:   {model}")
    print()
    print("  ⚠  Run this while the worker is IDLE (no other traffic).")

    passed = 0
    failed = 0

    # ==================================================================
    # Test 1: No reuse – every prompt is unique
    # ==================================================================
    print("\n--- Test 1: No KV Cache Reuse (unique prompts → ~0%) ---")
    try:
        p_before, c_before = poll_histograms()
        print(f"  Baseline: prompt_sum={p_before:.0f}  computed_sum={c_before:.0f}")

        print(f"  Sending {num_unique} unique random prompts...")
        for i in range(num_unique):
            send_completion(random_prompt())
            print(f"    [{i + 1}/{num_unique}] ✓")

        time.sleep(2)  # let metrics flush
        p_after, c_after = poll_histograms()

        p_delta = p_after - p_before
        c_delta = c_after - c_before
        cached = p_delta - c_delta
        efficiency = (cached / p_delta * 100) if p_delta > 0 else 0.0

        print("\n  Results:")
        print(f"    prompt_tokens  Δ {p_delta:.0f}")
        print(f"    computed_tokens Δ {c_delta:.0f}")
        print(f"    cached_tokens  Δ {cached:.0f}")
        print(f"    efficiency:      {efficiency:.1f}%")

        if efficiency < 15.0:
            print(f"\n  ✅ PASS (efficiency {efficiency:.1f}% < 15%)")
            passed += 1
        else:
            print(f"\n  ❌ FAIL (efficiency {efficiency:.1f}% >= 15%)")
            failed += 1

    except Exception as e:
        print(f"\n  ❌ ERROR: {e}")
        failed += 1

    # ==================================================================
    # Test 2: Perfect reuse – identical prompt after priming
    # ==================================================================
    print("\n--- Test 2: Perfect KV Cache Reuse (repeated prompt → ~100%) ---")
    try:
        # Prime: send once so blocks are in cache
        print("  Priming cache with initial request...")
        send_completion(reuse_prompt)
        time.sleep(2)

        # Baseline AFTER the prime so it doesn't count as a miss
        p_before, c_before = poll_histograms()
        print(f"  Baseline (post-prime): prompt_sum={p_before:.0f}  computed_sum={c_before:.0f}")

        print(f"  Sending {num_repeat} identical prompts...")
        for i in range(num_repeat):
            send_completion(reuse_prompt)
            print(f"    [{i + 1}/{num_repeat}] ✓")

        time.sleep(2)
        p_after, c_after = poll_histograms()

        p_delta = p_after - p_before
        c_delta = c_after - c_before
        cached = p_delta - c_delta
        efficiency = (cached / p_delta * 100) if p_delta > 0 else 0.0

        print("\n  Results:")
        print(f"    prompt_tokens  Δ {p_delta:.0f}")
        print(f"    computed_tokens Δ {c_delta:.0f}")
        print(f"    cached_tokens  Δ {cached:.0f}")
        print(f"    efficiency:      {efficiency:.1f}%")
        print("    (may be <100% due to block-size alignment)")

        if efficiency > 80.0:
            print(f"\n  ✅ PASS (efficiency {efficiency:.1f}% > 80%)")
            passed += 1
        else:
            print(f"\n  ❌ FAIL (efficiency {efficiency:.1f}% <= 80%)")
            failed += 1

    except Exception as e:
        print(f"\n  ❌ ERROR: {e}")
        failed += 1

    # ==================================================================
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Observe KV cache events from vLLM workers",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="""
Examples:
  # Monitor worker 0 (ZMQ events only):
  python kv_event_observer.py -p 20080 -v

  # Monitor with cache hit detection and efficiency tracking:
  python kv_event_observer.py -p 20080 -v -m 18081

  # Custom efficiency window (default 30s):
  python kv_event_observer.py -p 20080 -v -m 18081 --window 10

  # Monitor worker 1:
  python kv_event_observer.py -p 20081 -v -m 18082

  # Save events to file:
  python kv_event_observer.py -p 20080 -o events.jsonl

  # Run for 60 seconds:
  python kv_event_observer.py -p 20080 -d 60

  # Run self-test (no worker needed):
  python kv_event_observer.py --test

  # Run live integration test against a running worker:
  python kv_event_observer.py --test-live -m 18081 --model my-model

Event types:
  📦 STORED     - Block committed to prefix cache (ZMQ)
  🗑️ REMOVED    - Block evicted from cache (ZMQ)
  ✅ CACHE HIT  - Tokens served from cache (counter metrics)
  📊 EFFICIENCY - Sliding window KV cache efficiency (histogram metrics)
""")
    parser.add_argument("--host", "-H", default="localhost", help="Worker host (default: localhost)")
    parser.add_argument("--port", "-p", type=int, default=20080, help="KV event ZMQ port (default: 20080)")
    parser.add_argument("--metrics-port",
                        "-m",
                        type=int,
                        help="Prometheus metrics port for cache hit detection (e.g., 18081)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print each event")
    parser.add_argument("--output", "-o", help="Output file (JSONL format)")
    parser.add_argument("--duration", "-d", type=float, help="Run duration in seconds")
    parser.add_argument("--window",
                        "-w",
                        type=float,
                        default=30.0,
                        help="Sliding window size in seconds for efficiency calculation (default: 30)")
    parser.add_argument("--test",
                        action="store_true",
                        help="Run self-test to verify efficiency calculation (no worker needed)")
    parser.add_argument("--test-live",
                        action="store_true",
                        help="Run live integration test against a running worker (requires --metrics-port and --model)")
    parser.add_argument("--api-url",
                        default="http://localhost:8000",
                        help="API URL for live test requests (default: http://localhost:8000)")
    parser.add_argument("--model",
                        default=None,
                        help="Model name for live test requests (e.g., deepseek-ai/DeepSeek-R1-Distill-Llama-8B)")

    args = parser.parse_args()

    # Self-test mode: validate the sliding window logic and exit
    if args.test:
        success = run_self_test()
        sys.exit(0 if success else 1)

    # Live integration test: send real requests, check real metrics
    if args.test_live:
        if not args.metrics_port:
            print("ERROR: --metrics-port (-m) is required for --test-live")
            sys.exit(1)
        if not args.model:
            print("ERROR: --model is required for --test-live")
            sys.exit(1)
        metrics_url = f"http://{args.host}:{args.metrics_port}"
        success = run_live_test(args.api_url, metrics_url, args.model)
        sys.exit(0 if success else 1)

    observer = KVEventObserver(
        host=args.host,
        port=args.port,
        verbose=args.verbose,
        output_file=args.output,
        metrics_port=args.metrics_port,
        window_seconds=args.window,
    )

    signal.signal(signal.SIGINT, lambda s, f: setattr(observer, 'running', False))
    signal.signal(signal.SIGTERM, lambda s, f: setattr(observer, 'running', False))

    observer.connect()
    observer.run(duration=args.duration)


if __name__ == "__main__":
    main()
