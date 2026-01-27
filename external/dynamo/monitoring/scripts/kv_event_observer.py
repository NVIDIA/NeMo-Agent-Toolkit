#!/usr/bin/env python3
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


class KVEventObserver:
    """Observes KV cache events from a vLLM worker via ZMQ.
    
    Also optionally polls Prometheus metrics to detect cache hits,
    which don't generate ZMQ events.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 20080,
        verbose: bool = False,
        output_file: str | None = None,
        metrics_port: int | None = None,
    ):
        self.host = host
        self.port = port
        self.verbose = verbose
        self.output_file = output_file
        self.metrics_port = metrics_port
        self.stats = KVCacheStats()
        self.running = False
        self._output_handle = None

        # Metrics polling state
        self._last_hits = 0.0
        self._last_queries = 0.0
        self._metrics_thread = None

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)

    def _parse_metric(self, metrics_text: str, metric_name: str) -> float:
        """Extract a metric value from Prometheus text format."""
        pattern = rf'^{re.escape(metric_name)}\{{[^}}]*\}}\s+([0-9.e+-]+)'
        for line in metrics_text.split('\n'):
            match = re.match(pattern, line)
            if match:
                return float(match.group(1))
        return 0.0

    def _poll_metrics(self):
        """Background thread to poll Prometheus metrics for cache hits."""
        metrics_url = f"http://{self.host}:{self.metrics_port}/metrics"

        while self.running:
            try:
                with urllib.request.urlopen(metrics_url, timeout=2) as resp:
                    metrics_text = resp.read().decode('utf-8')

                hits = self._parse_metric(metrics_text, 'vllm:prefix_cache_hits_total')
                queries = self._parse_metric(metrics_text, 'vllm:prefix_cache_queries_total')

                # Calculate deltas
                hit_delta = hits - self._last_hits
                query_delta = queries - self._last_queries

                if hit_delta > 0:
                    # Cache hit detected!
                    self.stats.record_cache_hit(int(hit_delta), int(query_delta))
                    if self.verbose:
                        hit_rate = (hit_delta / query_delta * 100) if query_delta > 0 else 0
                        print(
                            f"✅ [CACHE HIT] tokens={int(hit_delta):4d} queried={int(query_delta):4d} hit_rate={hit_rate:.0f}%"
                        )
                elif query_delta > 0:
                    # Queries happened but no hits (cache miss)
                    self.stats.record_cache_hit(0, int(query_delta))

                self._last_hits = hits
                self._last_queries = queries

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
            # Initialize baseline metrics
            try:
                metrics_url = f"http://{self.host}:{self.metrics_port}/metrics"
                with urllib.request.urlopen(metrics_url, timeout=2) as resp:
                    metrics_text = resp.read().decode('utf-8')
                self._last_hits = self._parse_metric(metrics_text, 'vllm:prefix_cache_hits_total')
                self._last_queries = self._parse_metric(metrics_text, 'vllm:prefix_cache_queries_total')
                print(f"[KV Observer] ✓ Baseline: hits={self._last_hits:.0f} queries={self._last_queries:.0f}")
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
                print(
                    f"❓ [UNKNOWN ] seq={seq:6d} type={event_type} data={event[:3] if isinstance(event, (list, tuple)) else event}"
                )

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
                            print(f"[{batches_received:5d} batches] "
                                  f"Stored: {summary['stored_blocks']:4d} | "
                                  f"Removed: {summary['evicted_blocks']:4d} | "
                                  f"Net: {summary['net_blocks']:4d} | "
                                  f"Hashes: {summary['unique_hashes_current']} | "
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

        if self._output_handle:
            self._output_handle.close()

        self.socket.close()
        self.context.term()
        print("[KV Observer] Stopped")


def main():
    parser = argparse.ArgumentParser(description="Observe KV cache events from vLLM workers",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="""
Examples:
  # Monitor worker 0 (ZMQ events only):
  python kv_event_observer.py -p 20080 -v
  
  # Monitor with cache hit detection (polls Prometheus metrics):
  python kv_event_observer.py -p 20080 -v -m 18081
  
  # Monitor worker 1:
  python kv_event_observer.py -p 20081 -v -m 18082
  
  # Save events to file:
  python kv_event_observer.py -p 20080 -o events.jsonl
  
  # Run for 60 seconds:
  python kv_event_observer.py -p 20080 -d 60

Event types:
  📦 STORED   - Block committed to prefix cache (ZMQ)
  🗑️ REMOVED  - Block evicted from cache (ZMQ)
  ✅ CACHE HIT - Tokens served from cache (metrics polling)
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

    args = parser.parse_args()

    observer = KVEventObserver(
        host=args.host,
        port=args.port,
        verbose=args.verbose,
        output_file=args.output,
        metrics_port=args.metrics_port,
    )

    signal.signal(signal.SIGINT, lambda s, f: setattr(observer, 'running', False))
    signal.signal(signal.SIGTERM, lambda s, f: setattr(observer, 'running', False))

    observer.connect()
    observer.run(duration=args.duration)


if __name__ == "__main__":
    main()
