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
Dynamo Metrics Collector for NAT Profiler.

This module collects performance metrics from the Dynamo inference stack via Prometheus.
Metrics are collected from four Dynamo components:

- **Frontend** (:8000): User-facing latency, throughput, token statistics
- **Worker** (:8081): KV cache utilization, SGLang backend metrics
- **Router** (:8082): Thompson Sampling routing decisions
- **Processor** (:8083): Thompson Sampling KVE (KV Efficiency) metrics

Core Optimization Metrics
-------------------------

The profiler focuses on three core metrics for Dynamo LLM optimization:

1. **KV Efficiency (KVE)** - Token-agnostic measure of computational savings:

   - Formula: ``KVE = cached_tokens / prompt_tokens``
   - Measures the fraction of total work saved via KV cache reuse
   - A KVE of 0.8 means 80% of prompt tokens were served from cache
   - Source: Thompson Sampling processor (``dynamo_component_thompson_kve_*``)
   - Fallback: SGLang native ``cache_hit_rate`` if KVE counters unavailable
   - Affected by: prefix_id routing, prefix hints (osl, iat), request patterns

2. **Time to First Token (TTFT)** (``ttft_p50``, ``ttft_p95``, ``ttft_p99``):

   - Latency from request arrival to first token generation
   - Critical for user-perceived responsiveness
   - Affected by queue depth, worker selection, KV cache hits

3. **Inter-Token Latency (ITL)** (``itl_p50``, ``itl_p95``, ``itl_p99``):

   - Time between consecutive token generations during streaming
   - Affects smoothness of streaming responses
   - Influenced by batch scheduling and GPU utilization

Adding New Metrics
------------------

To add a new metric from any Dynamo endpoint:

1. **Find the metric name** by curling the endpoint::

       curl -s http://localhost:8081/metrics | grep -i kv
       curl -s http://localhost:8000/metrics | grep -i token

2. **Add the Prometheus query** to ``METRIC_QUERIES``::

       METRIC_QUERIES = {
           ...
           "my_new_metric": "rate(dynamo_component_my_metric_total[{range}])",
       }

   Note: Use ``{range}`` placeholder for time range (replaced with config value).

3. **Add the field** to ``DynamoMetricsResult``::

       class DynamoMetricsResult(BaseModel):
           ...
           my_new_metric: float | None = Field(
               default=None,
               description="Description of my new metric"
           )

4. **Update the collector** if needed (optional - for complex metrics):

   If the metric requires special handling (e.g., combining multiple queries),
   add custom logic in ``DynamoMetricsCollector.collect()``.

Metric Reference by Endpoint
----------------------------

**Frontend (:8000/metrics)**::

    dynamo_frontend_requests_total          # Counter: Total requests
    dynamo_frontend_inflight_requests       # Gauge: Current inflight
    dynamo_frontend_time_to_first_token_seconds_bucket  # Histogram: TTFT
    dynamo_frontend_inter_token_latency_seconds_bucket  # Histogram: ITL
    dynamo_frontend_output_tokens_total     # Counter: Total output tokens

**Worker (:8081/metrics)**::

    dynamo_component_kvstats_gpu_cache_usage_percent    # Gauge: KV cache %
    dynamo_component_kvstats_gpu_prefix_cache_hit_rate  # Gauge: Cache hit rate
    sglang:cache_hit_rate                   # Gauge: SGLang native cache hit
    sglang:gen_throughput                   # Gauge: Generation throughput
    sglang:num_running_reqs                 # Gauge: Running requests
    sglang:num_queue_reqs                   # Gauge: Queued requests

**Router (:8082/metrics)**::

    dynamo_component_requests_total{dynamo_endpoint="find_worker"}
    dynamo_component_request_duration_seconds_bucket

**Processor (:8083/metrics)**::

    dynamo_component_thompson_requests_total
    dynamo_component_thompson_kve_cached_tokens_total
    dynamo_component_thompson_kve_prompt_tokens_total
    dynamo_component_thompson_routing_decisions_total

See ``external/dynamo/monitoring/README.md`` for the complete metrics reference.
"""

import logging
import math
import time
from typing import Any

import httpx
from pydantic import BaseModel
from pydantic import Field

from nat.data_models.profiler import DynamoMetricsConfig

logger = logging.getLogger(__name__)

# =============================================================================
# PROMETHEUS QUERY DEFINITIONS
# =============================================================================

# Metric queries using Prometheus query language (PromQL).
# Use {range} placeholder for time range substitution.
#
# To add a new metric:
# 1. Add the query here with a descriptive key
# 2. Add corresponding field to DynamoMetricsResult
# 3. The collector will automatically fetch and populate it
METRIC_QUERIES: dict[str, str] = {
    # -------------------------------------------------------------------------
    # Inflight Requests (Gauge metrics - no rate needed)
    # -------------------------------------------------------------------------
    "inflight_requests_frontend": "dynamo_frontend_inflight_requests",
    "inflight_requests_worker": "dynamo_component_inflight_requests",
    "queued_requests": "dynamo_frontend_queued_requests",

    # -------------------------------------------------------------------------
    # Throughput (Rate metrics)
    # -------------------------------------------------------------------------
    "requests_per_minute": "rate(dynamo_frontend_requests_total[{range}]) * 60",
    "token_throughput": "rate(dynamo_frontend_output_tokens_total[{range}])",

    # -------------------------------------------------------------------------
    # Time to First Token (TTFT) - Histogram quantiles
    # -------------------------------------------------------------------------
    "ttft_p50": "histogram_quantile(0.50, rate(dynamo_frontend_time_to_first_token_seconds_bucket[{range}]))",
    "ttft_p95": "histogram_quantile(0.95, rate(dynamo_frontend_time_to_first_token_seconds_bucket[{range}]))",
    "ttft_p99": "histogram_quantile(0.99, rate(dynamo_frontend_time_to_first_token_seconds_bucket[{range}]))",

    # -------------------------------------------------------------------------
    # Inter-Token Latency (ITL) - Histogram quantiles
    # -------------------------------------------------------------------------
    "itl_p50": "histogram_quantile(0.50, rate(dynamo_frontend_inter_token_latency_seconds_bucket[{range}]))",
    "itl_p95": "histogram_quantile(0.95, rate(dynamo_frontend_inter_token_latency_seconds_bucket[{range}]))",
    "itl_p99": "histogram_quantile(0.99, rate(dynamo_frontend_inter_token_latency_seconds_bucket[{range}]))",

    # -------------------------------------------------------------------------
    # KV Cache Metrics (Gauge metrics)
    # -------------------------------------------------------------------------
    "kv_cache_usage_percent": "dynamo_component_kvstats_gpu_cache_usage_percent",
    "kv_cache_hit_rate_sglang": "sglang:cache_hit_rate",  # SGLang native (fallback)
    "kv_cache_hit_rate_dynamo": "dynamo_component_kvstats_gpu_prefix_cache_hit_rate",

    # -------------------------------------------------------------------------
    # KV Efficiency (KVE) - TRUE efficiency metric from Thompson Sampling processor
    # KVE = cached_tokens / prompt_tokens (fraction of work saved)
    # This is token-agnostic and measures actual computational savings
    # -------------------------------------------------------------------------
    "kve_cached_tokens_rate": "rate(dynamo_component_thompson_kve_cached_tokens_total[{range}])",
    "kve_prompt_tokens_rate": "rate(dynamo_component_thompson_kve_prompt_tokens_total[{range}])",
    # Block-level KVE metrics for deeper analysis
    "kve_device_blocks_rate": "rate(dynamo_component_thompson_kve_device_blocks_total[{range}])",
    "kve_host_blocks_rate": "rate(dynamo_component_thompson_kve_host_blocks_total[{range}])",
    "kve_disk_blocks_rate": "rate(dynamo_component_thompson_kve_disk_blocks_total[{range}])",

    # -------------------------------------------------------------------------
    # SGLang Worker Metrics (Gauge metrics)
    # -------------------------------------------------------------------------
    "sglang_running_requests": "sglang:num_running_reqs",
    "sglang_queue_depth": "sglang:num_queue_reqs",
    "sglang_gen_throughput": "sglang:gen_throughput",
    "sglang_utilization": "sglang:utilization",

    # -------------------------------------------------------------------------
    # Thompson Sampling Metrics (Rate metrics)
    # -------------------------------------------------------------------------
    "thompson_routing_decisions_rate": "rate(dynamo_component_thompson_routing_decisions_total[{range}])",
    "thompson_requests_rate": "rate(dynamo_component_thompson_requests_total[{range}])",
}

# =============================================================================
# DATA MODELS
# =============================================================================


class DynamoCoreMetrics(BaseModel):
    """
    Core optimization metrics for Dynamo LLM inference.

    These three metrics are the primary targets for optimization:

    1. **KV Efficiency (KVE)**: Fraction of computational work saved via KV cache.
       - Formula: ``cached_tokens / prompt_tokens``
       - Target: Maximize (closer to 1.0 = more work saved)
       - Affected by: prefix_id routing, prefix hints (osl, iat), request patterns
       - Token-agnostic measure of actual computational savings

    2. **TTFT (Time to First Token)**: User-perceived initial latency.
       - Target: Minimize (lower is better)
       - Affected by: queue depth, worker selection, KV cache hits

    3. **ITL (Inter-Token Latency)**: Streaming smoothness.
       - Target: Minimize (lower is better)
       - Affected by: batch scheduling, GPU utilization, memory bandwidth

    Usage::

        result = await collector.collect()
        core = result.get_core_metrics()

        print(f"KV Efficiency: {core.kv_efficiency:.2%}")
        print(f"TTFT P95: {core.ttft_p95_seconds:.3f}s")
        print(f"ITL P95: {core.itl_p95_seconds:.3f}s")

        # Check if all core metrics are available
        if core.is_complete():
            print("All core metrics collected successfully")
    """

    # -------------------------------------------------------------------------
    # KV Efficiency - KVE (CORE METRIC #1)
    # Goal: MAXIMIZE - Higher efficiency = more computational work saved
    # Formula: cached_tokens / prompt_tokens
    # -------------------------------------------------------------------------
    kv_efficiency: float | None = Field(
        default=None,
        description="KV Efficiency (0-1): fraction of prompt tokens served from cache. "
        "Computed as cached_tokens / prompt_tokens from Thompson Sampling processor. "
        "Higher values indicate more computational work saved via KV cache reuse. "
        "This is the PRIMARY metric affected by prefix routing hints "
        "(nvext_prefix_id, nvext_prefix_osl, nvext_prefix_iat).",
    )
    kv_efficiency_fallback: float | None = Field(
        default=None,
        description="Fallback KV efficiency from SGLang native cache_hit_rate. "
        "Used when Thompson Sampling KVE counters are unavailable.",
    )

    # -------------------------------------------------------------------------
    # Time to First Token - TTFT (CORE METRIC #2)
    # Goal: MINIMIZE - Lower latency = faster initial response
    # -------------------------------------------------------------------------
    ttft_p50_seconds: float | None = Field(
        default=None,
        description="Time to First Token - 50th percentile (median) in seconds",
    )
    ttft_p95_seconds: float | None = Field(
        default=None,
        description="Time to First Token - 95th percentile in seconds. "
        "Primary latency target for optimization.",
    )
    ttft_p99_seconds: float | None = Field(
        default=None,
        description="Time to First Token - 99th percentile in seconds (tail latency)",
    )

    # -------------------------------------------------------------------------
    # Inter-Token Latency - ITL (CORE METRIC #3)
    # Goal: MINIMIZE - Lower latency = smoother streaming
    # -------------------------------------------------------------------------
    itl_p50_seconds: float | None = Field(
        default=None,
        description="Inter-Token Latency - 50th percentile (median) in seconds",
    )
    itl_p95_seconds: float | None = Field(
        default=None,
        description="Inter-Token Latency - 95th percentile in seconds. "
        "Primary streaming smoothness target.",
    )
    itl_p99_seconds: float | None = Field(
        default=None,
        description="Inter-Token Latency - 99th percentile in seconds (tail latency)",
    )

    def get_effective_kv_efficiency(self) -> float | None:
        """
        Get the best available KV efficiency value.

        Prefers the true KVE (cached_tokens/prompt_tokens) from Thompson Sampling,
        falls back to SGLang native cache_hit_rate if KVE is unavailable.

        Returns:
            KV efficiency (0-1) or None if neither source is available
        """
        if self.kv_efficiency is not None:
            return self.kv_efficiency
        return self.kv_efficiency_fallback

    def is_complete(self) -> bool:
        """
        Check if all core optimization metrics were successfully collected.

        Returns:
            True if KV efficiency (or fallback), ttft_p95, and itl_p95 are all available
        """
        return all([
            self.get_effective_kv_efficiency() is not None,
            self.ttft_p95_seconds is not None,
            self.itl_p95_seconds is not None,
        ])

    def get_optimization_summary(self) -> dict[str, float | None]:
        """
        Get a summary dict of the primary optimization targets.

        Returns:
            Dict with the three key metrics for optimization loops
        """
        return {
            "kv_efficiency": self.get_effective_kv_efficiency(),
            "kv_efficiency_source": "kve" if self.kv_efficiency is not None else "sglang_fallback",
            "ttft_p95_seconds": self.ttft_p95_seconds,
            "itl_p95_seconds": self.itl_p95_seconds,
        }

    def to_optimization_score(
        self,
        kv_weight: float = 0.4,
        ttft_weight: float = 0.4,
        itl_weight: float = 0.2,
        ttft_target_seconds: float = 0.5,
        itl_target_seconds: float = 0.05,
    ) -> float | None:
        """
        Compute a combined optimization score (higher is better).

        This provides a single scalar for optimization algorithms that combines
        the three core metrics with configurable weights.

        Args:
            kv_weight: Weight for KV efficiency (0-1)
            ttft_weight: Weight for TTFT score (0-1)
            itl_weight: Weight for ITL score (0-1)
            ttft_target_seconds: Target TTFT for scoring (score=1.0 at target)
            itl_target_seconds: Target ITL for scoring (score=1.0 at target)

        Returns:
            Combined score (0-1) where higher is better, or None if metrics unavailable

        Note:
            Weights should sum to 1.0. TTFT and ITL scores are computed as
            target/actual (capped at 1.0) so lower latency = higher score.
        """
        if not self.is_complete():
            return None

        # KV efficiency score is already 0-1 (higher is better)
        kv_score = self.get_effective_kv_efficiency() or 0.0

        # TTFT score: target/actual, capped at 1.0 (lower latency = higher score)
        ttft_score = min(1.0, ttft_target_seconds / max(self.ttft_p95_seconds or ttft_target_seconds, 0.001))

        # ITL score: target/actual, capped at 1.0 (lower latency = higher score)
        itl_score = min(1.0, itl_target_seconds / max(self.itl_p95_seconds or itl_target_seconds, 0.001))

        return (kv_weight * kv_score) + (ttft_weight * ttft_score) + (itl_weight * itl_score)


class DynamoMetricsResult(BaseModel):
    """
    Results from Dynamo metrics collection.

    To add a new metric:
    1. Add a field here with appropriate type and description
    2. Add the corresponding Prometheus query to METRIC_QUERIES above
    3. The collector will automatically populate it

    All metrics are optional (None) to handle cases where:
    - The metric endpoint is unavailable
    - Prometheus query returns no data
    - The Dynamo component is not running

    For optimization, use ``get_core_metrics()`` to extract the three primary
    optimization targets (KV Cache Efficiency, TTFT, ITL).
    """

    # =========================================================================
    # CORE OPTIMIZATION METRICS (Primary targets for optimization)
    # =========================================================================

    # -------------------------------------------------------------------------
    # KV Efficiency - KVE (CORE METRIC #1)
    # Dashboard panels: "KV Cache Usage %", "KV Cache Stats"
    # KVE = cached_tokens / prompt_tokens (fraction of work saved)
    # -------------------------------------------------------------------------
    kve_cached_tokens_rate: float | None = Field(
        default=None,
        description="Rate of tokens served from KV cache (tokens/sec). KVE numerator.",
    )
    kve_prompt_tokens_rate: float | None = Field(
        default=None,
        description="Rate of total prompt tokens processed (tokens/sec). KVE denominator.",
    )
    kve_device_blocks_rate: float | None = Field(
        default=None,
        description="Rate of KV blocks served from GPU memory (blocks/sec)",
    )
    kve_host_blocks_rate: float | None = Field(
        default=None,
        description="Rate of KV blocks served from CPU/host memory (blocks/sec)",
    )
    kve_disk_blocks_rate: float | None = Field(
        default=None,
        description="Rate of KV blocks served from disk (blocks/sec)",
    )
    kv_cache_usage_percent: float | None = Field(
        default=None,
        description="GPU KV cache memory utilization (0-100%)",
    )
    kv_cache_hit_rate_sglang: float | None = Field(
        default=None,
        description="[FALLBACK] KV cache hit rate from SGLang native metric (0-1). "
        "Used when Thompson Sampling KVE counters are unavailable.",
    )
    kv_cache_hit_rate_dynamo: float | None = Field(
        default=None,
        description="KV cache hit rate from Dynamo component (0-1), alternative source",
    )

    # -------------------------------------------------------------------------
    # Time to First Token - TTFT (CORE METRIC #2)
    # Dashboard panels: "Time to First Token (P95)", "TTFT Over Time"
    # -------------------------------------------------------------------------
    ttft_p50: float | None = Field(
        default=None,
        description="Time to First Token - 50th percentile (seconds)",
    )
    ttft_p95: float | None = Field(
        default=None,
        description="[CORE] Time to First Token - 95th percentile (seconds). PRIMARY latency target.",
    )
    ttft_p99: float | None = Field(
        default=None,
        description="Time to First Token - 99th percentile (seconds)",
    )

    # -------------------------------------------------------------------------
    # Inter-Token Latency - ITL (CORE METRIC #3)
    # Dashboard panel: "ITL Over Time" - Inter-token latency trends
    # -------------------------------------------------------------------------
    itl_p50: float | None = Field(
        default=None,
        description="Inter-Token Latency - 50th percentile (seconds)",
    )
    itl_p95: float | None = Field(
        default=None,
        description="[CORE] Inter-Token Latency - 95th percentile (seconds). PRIMARY streaming target.",
    )
    itl_p99: float | None = Field(
        default=None,
        description="Inter-Token Latency - 99th percentile (seconds)",
    )

    # =========================================================================
    # SUPPLEMENTARY METRICS (Context and diagnostics)
    # =========================================================================

    # -------------------------------------------------------------------------
    # Inflight Requests
    # Dashboard panel: "Inflight Requests" - Current load across components
    # -------------------------------------------------------------------------
    inflight_requests_frontend: float | None = Field(
        default=None,
        description="Current inflight requests at the frontend (user-facing API)",
    )
    inflight_requests_worker: float | None = Field(
        default=None,
        description="Current inflight requests at the worker (SGLang backend)",
    )
    queued_requests: float | None = Field(
        default=None,
        description="Requests currently queued at the frontend",
    )

    # -------------------------------------------------------------------------
    # Throughput
    # Dashboard panel: "Requests/min" - Throughput
    # -------------------------------------------------------------------------
    requests_per_minute: float | None = Field(
        default=None,
        description="Request throughput in requests per minute",
    )

    # -------------------------------------------------------------------------
    # Token Throughput
    # Dashboard panel: "Token Throughput" - Tokens generated per second
    # -------------------------------------------------------------------------
    token_throughput: float | None = Field(
        default=None,
        description="Output token generation rate (tokens/second)",
    )

    # -------------------------------------------------------------------------
    # SGLang Worker Metrics
    # Additional worker-level metrics for deeper analysis
    # -------------------------------------------------------------------------
    sglang_running_requests: float | None = Field(
        default=None,
        description="Number of requests currently running in SGLang",
    )
    sglang_queue_depth: float | None = Field(
        default=None,
        description="Number of requests queued in SGLang",
    )
    sglang_gen_throughput: float | None = Field(
        default=None,
        description="SGLang generation throughput",
    )
    sglang_utilization: float | None = Field(
        default=None,
        description="SGLang GPU utilization",
    )

    # -------------------------------------------------------------------------
    # Thompson Sampling Metrics
    # Routing efficiency and decision-making metrics
    # -------------------------------------------------------------------------
    thompson_routing_decisions_rate: float | None = Field(
        default=None,
        description="Rate of Thompson Sampling routing decisions per second",
    )
    thompson_requests_rate: float | None = Field(
        default=None,
        description="Rate of requests processed by Thompson Sampling processor",
    )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------
    collection_timestamp: float | None = Field(
        default=None,
        description="Unix timestamp when metrics were collected",
    )
    prometheus_url: str | None = Field(
        default=None,
        description="Prometheus URL used for collection",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Any errors encountered during collection",
    )

    # =========================================================================
    # CORE METRICS EXTRACTION
    # =========================================================================

    def compute_kv_efficiency(self) -> float | None:
        """
        Compute KV Efficiency (KVE) from Thompson Sampling processor metrics.

        KVE = cached_tokens / prompt_tokens

        This measures the fraction of computational work saved via KV cache reuse.
        A KVE of 0.8 means 80% of prompt tokens were served from cache.

        Returns:
            KVE (0-1) if both metrics are available and prompt_tokens > 0, else None
        """
        if self.kve_cached_tokens_rate is None or self.kve_prompt_tokens_rate is None:
            return None
        if self.kve_prompt_tokens_rate <= 0:
            return None
        return self.kve_cached_tokens_rate / self.kve_prompt_tokens_rate

    def get_core_metrics(self) -> DynamoCoreMetrics:
        """
        Extract the three core optimization metrics.

        KV Efficiency is computed as cached_tokens / prompt_tokens from the
        Thompson Sampling processor. Falls back to SGLang native cache_hit_rate
        if KVE counters are unavailable.

        Returns:
            DynamoCoreMetrics with KV efficiency, TTFT, and ITL

        Usage::

            result = await collector.collect()
            core = result.get_core_metrics()

            if core.is_complete():
                score = core.to_optimization_score()
                print(f"Optimization score: {score:.3f}")
        """
        # Compute true KVE from Thompson Sampling processor metrics
        kv_efficiency = self.compute_kv_efficiency()

        return DynamoCoreMetrics(
            kv_efficiency=kv_efficiency,
            kv_efficiency_fallback=self.kv_cache_hit_rate_sglang,
            ttft_p50_seconds=self.ttft_p50,
            ttft_p95_seconds=self.ttft_p95,
            ttft_p99_seconds=self.ttft_p99,
            itl_p50_seconds=self.itl_p50,
            itl_p95_seconds=self.itl_p95,
            itl_p99_seconds=self.itl_p99,
        )

    def has_core_metrics(self) -> bool:
        """
        Check if all three core optimization metrics are available.

        Returns:
            True if kv_cache_hit_rate, ttft_p95, and itl_p95 are all collected
        """
        return self.get_core_metrics().is_complete()


# =============================================================================
# METRICS COLLECTOR
# =============================================================================


class DynamoMetricsCollector:
    """
    Collects Dynamo inference stack metrics from Prometheus.

    Usage::

        from nat.plugins.profiler.inference_optimization.dynamo_metrics import DynamoMetricsCollector
        from nat.data_models.profiler import DynamoMetricsConfig

        config = DynamoMetricsConfig(enable=True, prometheus_url="http://localhost:9090")
        collector = DynamoMetricsCollector(config)
        result = await collector.collect()

        print(f"TTFT P95: {result.ttft_p95}")
        print(f"KV Cache Usage: {result.kv_cache_usage_percent}%")
    """

    def __init__(self, config: DynamoMetricsConfig):
        """
        Initialize the collector with configuration.

        Args:
            config: DynamoMetricsConfig with Prometheus URL and metric toggles
        """
        self.config = config
        self.prometheus_url = config.prometheus_url.rstrip("/")

    async def collect(self) -> DynamoMetricsResult:
        """
        Collect all enabled Dynamo metrics from Prometheus.

        Returns:
            DynamoMetricsResult with collected metric values
        """
        result = DynamoMetricsResult(
            collection_timestamp=time.time(),
            prometheus_url=self.prometheus_url,
        )

        # Build list of metrics to collect based on config toggles
        metrics_to_collect = self._get_enabled_metrics()

        # Log collection parameters
        if self.config.workflow_start_timestamp is not None:
            if self.config.workflow_end_timestamp is not None:
                duration = self.config.workflow_end_timestamp - self.config.workflow_start_timestamp
                lookback_info = f"isolated_window={duration:.1f}s"
            else:
                lookback_info = f"workflow_start={self.config.workflow_start_timestamp:.2f}"
        elif self.config.lookback_seconds > 0:
            lookback_info = f"lookback={self.config.lookback_seconds}s"
        else:
            lookback_info = "lookback=600s (default)"

        logger.info("Collecting %d Dynamo metrics from %s (query_range=%s, %s)",
                    len(metrics_to_collect),
                    self.prometheus_url,
                    self.config.query_range,
                    lookback_info)

        collected_count = 0
        null_count = 0

        # Collect each metric
        async with httpx.AsyncClient(timeout=30.0) as client:
            for metric_name, query_template in metrics_to_collect.items():
                try:
                    # Substitute time range placeholder
                    query = query_template.replace("{range}", self.config.query_range)
                    value = await self._query_prometheus(client, query)

                    if value is not None:
                        setattr(result, metric_name, value)
                        logger.debug("Collected %s = %s", metric_name, value)
                        collected_count += 1
                    else:
                        logger.debug("No data for metric %s", metric_name)
                        null_count += 1

                except Exception as e:
                    error_msg = f"Failed to collect {metric_name}: {e}"
                    logger.warning(error_msg)
                    result.errors.append(error_msg)

        logger.info("Dynamo metrics collection complete: %d collected, %d null, %d errors",
                    collected_count,
                    null_count,
                    len(result.errors))

        # Log summary of key metrics for debugging
        core = result.get_core_metrics()
        if core.ttft_p95_seconds is not None or core.itl_p95_seconds is not None:
            logger.info("Core metrics - TTFT P95: %s, ITL P95: %s, KV Efficiency: %s",
                        core.ttft_p95_seconds,
                        core.itl_p95_seconds,
                        core.kv_efficiency)
        else:
            logger.warning("Core metrics (TTFT, ITL) not available - check Prometheus connectivity and metric names")

        return result

    def _get_enabled_metrics(self) -> dict[str, str]:
        """
        Get the subset of METRIC_QUERIES enabled by config.

        Returns:
            Dict mapping metric names to their Prometheus queries
        """
        enabled: dict[str, str] = {}

        # Map config flags to metric prefixes/names
        metric_groups = {
            "collect_inflight_requests": ["inflight_requests_frontend", "inflight_requests_worker", "queued_requests"],
            "collect_throughput": ["requests_per_minute"],
            "collect_ttft": ["ttft_p50", "ttft_p95", "ttft_p99"],
            "collect_itl": ["itl_p50", "itl_p95", "itl_p99"],
            "collect_kv_cache": [
                # KVE metrics (primary - token-level efficiency)
                "kve_cached_tokens_rate",
                "kve_prompt_tokens_rate",
                "kve_device_blocks_rate",
                "kve_host_blocks_rate",
                "kve_disk_blocks_rate",  # Supplementary KV cache metrics
                "kv_cache_usage_percent",
                "kv_cache_hit_rate_sglang",  # Fallback for KVE
                "kv_cache_hit_rate_dynamo",
            ],
            "collect_token_throughput": ["token_throughput", "sglang_gen_throughput"],
        }

        for config_flag, metric_names in metric_groups.items():
            if getattr(self.config, config_flag, False):
                for name in metric_names:
                    if name in METRIC_QUERIES:
                        enabled[name] = METRIC_QUERIES[name]

        # Always collect SGLang worker metrics for context
        for name in ["sglang_running_requests", "sglang_queue_depth", "sglang_utilization"]:
            if name in METRIC_QUERIES:
                enabled[name] = METRIC_QUERIES[name]

        # Always collect Thompson Sampling metrics when available
        for name in ["thompson_routing_decisions_rate", "thompson_requests_rate"]:
            if name in METRIC_QUERIES:
                enabled[name] = METRIC_QUERIES[name]

        return enabled

    async def _query_prometheus(self, client: httpx.AsyncClient, query: str) -> float | None:
        """
        Execute a Prometheus query and extract the scalar result.

        First attempts an instant query. If no data is returned (e.g., because
        rate() returns 0 after workflow completion), falls back to a range query
        with historical lookback to capture the most recent non-zero value.

        Args:
            client: httpx AsyncClient
            query: PromQL query string

        Returns:
            Float value if successful, None if no data or error
        """
        # First try instant query
        value = await self._query_prometheus_instant(client, query)
        if value is not None:
            return value

        # If instant query failed, try range query with lookback
        # This captures historical data when rate() returns 0 after workflow completes
        logger.debug("Instant query returned no data, trying range query with lookback: %s", query)
        return await self._query_prometheus_range(client, query)

    async def _query_prometheus_instant(self, client: httpx.AsyncClient, query: str) -> float | None:
        """
        Execute a Prometheus instant query.

        Args:
            client: httpx AsyncClient
            query: PromQL query string

        Returns:
            Float value if successful, None if no data or error
        """
        url = f"{self.prometheus_url}/api/v1/query"
        params = {"query": query}

        response = await client.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        if data.get("status") != "success":
            logger.warning("Prometheus instant query failed: %s", data.get("error", "unknown"))
            return None

        results = data.get("data", {}).get("result", [])

        if not results:
            logger.debug("No data for instant query: %s", query)
            return None

        # For instant queries, extract the value from the first result
        # Result format: [{"metric": {...}, "value": [timestamp, "value_string"]}]
        try:
            value_str = results[0]["value"][1]
            value = float(value_str)

            # Handle special float values
            if math.isnan(value):
                logger.debug("Instant query returned NaN for: %s", query)
                return None

            # Zero values from rate() after activity stops are not useful
            if value == 0.0:
                logger.debug("Instant query returned 0.0 for rate-based query: %s", query)
                return None

            return value
        except (KeyError, IndexError, ValueError) as e:
            logger.debug("Failed to parse Prometheus instant result for query '%s': %s", query, e)
            return None

    async def _query_prometheus_range(self, client: httpx.AsyncClient, query: str) -> float | None:
        """
        Execute a Prometheus range query with historical lookback.

        This captures metrics that were recorded during the workflow execution
        but are no longer updating (rate() would return 0 for instant queries).

        The time window is determined by:
        1. If workflow timestamps are set: query from workflow start to workflow end (isolated to this eval)
        2. If lookback_seconds is set: query that many seconds back from now
        3. Otherwise: default to 10 minutes (600 seconds)

        Args:
            client: httpx AsyncClient
            query: PromQL query string

        Returns:
            The most recent non-NaN, non-zero value if found, None otherwise
        """
        url = f"{self.prometheus_url}/api/v1/query_range"

        # Determine time window based on config
        # Priority: workflow timestamps > lookback_seconds > default 600s
        if self.config.workflow_start_timestamp is not None:
            # Use exact workflow time window (no buffer before, small buffer after for scrape delay)
            # No buffer before: avoids any risk of including pre-workflow empty data
            # Small buffer after (15s): accounts for Prometheus scrape interval
            start_time = self.config.workflow_start_timestamp

            if self.config.workflow_end_timestamp is not None:
                # Use actual workflow end time + small buffer for scrape delay
                end_time = self.config.workflow_end_timestamp + 15.0
                logger.debug("Using isolated workflow time window: %.2f to %.2f (%.1f seconds)",
                             start_time,
                             end_time,
                             end_time - start_time)
            else:
                # Fall back to current time if end timestamp not set
                end_time = time.time()
                logger.debug("Using workflow start with current time: %.2f to %.2f (%.1f seconds)",
                             start_time,
                             end_time,
                             end_time - start_time)
        elif self.config.lookback_seconds > 0:
            end_time = time.time()
            start_time = end_time - self.config.lookback_seconds
            logger.debug("Using configured lookback for range query: %.1f seconds", self.config.lookback_seconds)
        else:
            # Default to 10 minutes (600 seconds) for backward compatibility
            end_time = time.time()
            start_time = end_time - 600
            logger.debug("Using default 10-minute lookback for range query")

        # Use 15s step to get reasonable granularity
        step = "15s"

        params = {
            "query": query,
            "start": start_time,
            "end": end_time,
            "step": step,
        }

        try:
            response = await client.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            if data.get("status") != "success":
                logger.warning("Prometheus range query failed: %s", data.get("error", "unknown"))
                return None

            results = data.get("data", {}).get("result", [])

            if not results:
                logger.debug("No data for range query: %s", query)
                return None

            # Range query result format:
            # [{"metric": {...}, "values": [[timestamp, "value_string"], ...]}]
            # Collect all valid (non-NaN, non-zero) values and compute the average
            # This gives a representative measurement across the entire workflow
            valid_values: list[float] = []

            for series in results:
                values = series.get("values", [])
                for timestamp_val, value_str in values:
                    try:
                        value = float(value_str)
                        if not math.isnan(value) and value != 0.0:
                            valid_values.append(value)
                    except (ValueError, TypeError):
                        continue

            if valid_values:
                # Use average for a representative measurement across the workflow
                avg_value = sum(valid_values) / len(valid_values)
                min_value = min(valid_values)
                max_value = max(valid_values)
                logger.debug("Range query found %d valid samples for %s: avg=%.4f, min=%.4f, max=%.4f",
                             len(valid_values),
                             query,
                             avg_value,
                             min_value,
                             max_value)
                return avg_value

            logger.debug("Range query found no valid values for: %s", query)
            return None

        except Exception as e:
            logger.debug("Range query failed for '%s': %s", query, e)
            return None

    async def health_check(self) -> dict[str, Any]:
        """
        Check connectivity to Prometheus and Dynamo endpoints.

        Returns:
            Dict with health status for each component
        """
        health: dict[str, Any] = {
            "prometheus": False,
            "frontend": False,
            "worker": False,
            "errors": [],
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            # Check Prometheus
            try:
                response = await client.get(f"{self.prometheus_url}/-/healthy")
                health["prometheus"] = response.status_code == 200
            except Exception as e:
                health["errors"].append(f"Prometheus: {e}")

            # Check if Dynamo metrics are being scraped
            try:
                # Query for any frontend metric to verify scraping
                url = f"{self.prometheus_url}/api/v1/query"
                response = await client.get(url, params={"query": "up{job=~\".*dynamo.*\"}"})
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("data", {}).get("result", [])
                    health["frontend"] = len(results) > 0
                    health["worker"] = len(results) > 0
            except Exception as e:
                health["errors"].append(f"Dynamo metrics check: {e}")

        return health


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def collect_dynamo_metrics(config: DynamoMetricsConfig) -> DynamoMetricsResult:
    """
    Convenience function to collect Dynamo metrics.

    Args:
        config: DynamoMetricsConfig with collection settings

    Returns:
        DynamoMetricsResult with collected metrics
    """
    collector = DynamoMetricsCollector(config)
    return await collector.collect()


async def collect_core_metrics(
    prometheus_url: str = "http://localhost:9090",
    query_range: str = "30s",
) -> DynamoCoreMetrics:
    """
    Convenience function to collect only the three core optimization metrics.

    This is a simplified interface for optimization loops that only need:
    - KV Cache Efficiency
    - Time to First Token (TTFT)
    - Inter-Token Latency (ITL)

    Args:
        prometheus_url: Prometheus server URL
        query_range: Time range for rate calculations (e.g., '1m', '5m')

    Returns:
        DynamoCoreMetrics with the three core metrics

    Usage::

        from nat.plugins.profiler.inference_optimization.dynamo_metrics import collect_core_metrics

        # Quick collection for optimization
        core = await collect_core_metrics()

        if core.is_complete():
            print(f"KV Efficiency: {core.kv_cache_efficiency:.2%}")
            print(f"TTFT P95: {core.ttft_p95_seconds:.3f}s")
            print(f"ITL P95: {core.itl_p95_seconds:.3f}s")

            # Get combined optimization score
            score = core.to_optimization_score()
            print(f"Combined score: {score:.3f}")
    """
    config = DynamoMetricsConfig(
        enable=True,
        prometheus_url=prometheus_url,
        query_range=query_range,
        # Enable only core metrics for efficiency
        collect_kv_cache=True,
        collect_ttft=True,
        collect_itl=True,
        # Disable supplementary metrics
        collect_inflight_requests=False,
        collect_throughput=False,
        collect_token_throughput=False,
    )
    result = await collect_dynamo_metrics(config)
    return result.get_core_metrics()
