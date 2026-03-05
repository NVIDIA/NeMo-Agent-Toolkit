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

from pydantic import BaseModel
from pydantic import Field


class PromptCachingConfig(BaseModel):
    enable: bool = False
    min_frequency: float = 0.5


class BottleneckConfig(BaseModel):
    enable_simple_stack: bool = False
    enable_nested_stack: bool = False


class ConcurrencySpikeConfig(BaseModel):
    enable: bool = False
    spike_threshold: int = 1


class PrefixSpanConfig(BaseModel):
    enable: bool = False
    min_support: float = 2
    min_coverage: float = 0
    max_text_len: int = 1000
    top_k: int = 10
    chain_with_common_prefixes: bool = False


class PredictionTrieConfig(BaseModel):
    enable: bool = False
    output_filename: str = "prediction_trie.json"
    auto_sensitivity: bool = True
    sensitivity_scale: int = 5
    w_critical: float = 0.5
    w_fanout: float = 0.3
    w_position: float = 0.2
    w_parallel: float = 0.0


class DynamoMetricsConfig(BaseModel):
    """
    Configuration for collecting Dynamo inference stack metrics.

    Core Optimization Metrics
    -------------------------
    The profiler focuses on three core metrics for Dynamo LLM optimization:

    1. **KV Efficiency (KVE)** (``collect_kv_cache``):
       Token-agnostic measure of computational work saved via KV cache.
       Formula: ``KVE = cached_tokens / prompt_tokens``
       A KVE of 0.8 means 80% of prompt tokens were served from cache.
       Affected by prefix routing hints (prefix_id, nvext_prefix_osl, nvext_prefix_iat).

    2. **Time to First Token - TTFT** (``collect_ttft``):
       Latency from request to first token. Lower = faster initial response.
       Affected by queue depth, worker selection, KV cache hits.

    3. **Inter-Token Latency - ITL** (``collect_itl``):
       Time between tokens during streaming. Lower = smoother streaming.
       Affected by batch scheduling, GPU utilization.

    To collect only core metrics for optimization, use::

        config = DynamoMetricsConfig.core_metrics_only()

    Dynamo Endpoints
    ----------------
    - Frontend (:8000/metrics): Latency, throughput, token stats
    - Worker (:8081/metrics): KV cache, SGLang stats
    - Router (:8082/metrics): Thompson Sampling routing
    - Processor (:8083/metrics): Thompson Sampling KVE

    Adding New Metrics
    ------------------
    To add metrics from any Dynamo endpoint:

    1. **Identify the metric** from the endpoint::

           curl localhost:8081/metrics | grep kv

    2. **Add to DynamoMetricsResult** in ``src/nat/profiler/inference_optimization/dynamo_metrics.py``:
       - Add a new field to the Pydantic model
       - Add the Prometheus query in ``METRIC_QUERIES``

    3. **Example - Adding a new metric**::

           # In dynamo_metrics.py METRIC_QUERIES dict:
           "my_new_metric": "rate(dynamo_component_my_metric_total[5m])"

           # In DynamoMetricsResult model:
           my_new_metric: float | None = Field(default=None, description="My new metric")

    Metric Reference by Endpoint
    ----------------------------
    - **Frontend (:8000)**: ``dynamo_frontend_*`` (requests, latency, tokens)
    - **Worker (:8081)**: ``dynamo_component_kvstats_*``, ``sglang:*`` (KV cache, SGLang)
    - **Router (:8082)**: ``dynamo_component_*`` with ``dynamo_component="router"`` label
    - **Processor (:8083)**: ``dynamo_component_thompson_*`` (Thompson Sampling)

    See ``external/dynamo/monitoring/README.md`` for the complete metrics reference.
    """

    enable: bool = Field(default=False, description="Enable Dynamo metrics collection")

    prometheus_url: str = Field(
        default="http://localhost:9090",
        description="Prometheus server URL for querying Dynamo metrics",
    )

    # =========================================================================
    # CORE OPTIMIZATION METRICS (Primary targets)
    # =========================================================================
    collect_kv_cache: bool = Field(
        default=True,
        description="[CORE] Collect KV Efficiency (KVE = cached_tokens/prompt_tokens) - "
        "primary metric for prefix caching optimization. Measures fraction of work saved.",
    )
    collect_ttft: bool = Field(
        default=True,
        description="[CORE] Collect Time to First Token (P50/P95/P99) - primary latency metric",
    )
    collect_itl: bool = Field(
        default=True,
        description="[CORE] Collect Inter-Token Latency (P50/P95/P99) - primary streaming metric",
    )

    # =========================================================================
    # SUPPLEMENTARY METRICS (Context and diagnostics)
    # =========================================================================
    collect_inflight_requests: bool = Field(
        default=True,
        description="Collect current inflight requests across components",
    )
    collect_throughput: bool = Field(
        default=True,
        description="Collect requests per minute throughput",
    )
    collect_token_throughput: bool = Field(
        default=True,
        description="Collect token generation throughput (tokens/sec)",
    )

    # Query time range for rate calculations
    query_range: str = Field(
        default="30s",
        description="Time range for rate calculations in Prometheus queries. "
        "Minimum: '15s' (Prometheus scrapes every 5s, need ≥3 points for reliable rates). "
        "Options: '15s', '30s' (default), '1m', '2m', '5m'. "
        "Should roughly match experiment duration. Too short = noisy. Too long = stale data included.",
    )

    # Historical lookback for range queries (set automatically from workflow duration if 0)
    lookback_seconds: float = Field(
        default=0.0,
        description="Lookback time in seconds for Prometheus range queries when instant queries return no data. "
        "If 0 (default), will be set automatically to the workflow duration + buffer. "
        "This allows capturing TTFT/ITL metrics from the entire eval run, even after the workflow completes.",
    )

    # Workflow time window (set automatically by profiler)
    workflow_start_timestamp: float | None = Field(
        default=None,
        description="Unix timestamp when the workflow started (set automatically by profiler). "
        "Used for precise range query time windows.",
    )
    workflow_end_timestamp: float | None = Field(
        default=None,
        description="Unix timestamp when the workflow ended (set automatically by profiler). "
        "Used for precise range query time windows to isolate metrics to this eval run.",
    )

    @classmethod
    def core_metrics_only(
        cls,
        prometheus_url: str = "http://localhost:9090",
        query_range: str = "30s",
    ) -> "DynamoMetricsConfig":
        """
        Create a config that collects only the three core optimization metrics.

        This is optimized for tight optimization loops where you only need:
        - KV Cache Efficiency
        - TTFT (Time to First Token)
        - ITL (Inter-Token Latency)

        Args:
            prometheus_url: Prometheus server URL
            query_range: Time range for rate calculations

        Returns:
            DynamoMetricsConfig with only core metrics enabled

        Usage::

            config = DynamoMetricsConfig.core_metrics_only()
            # Equivalent to:
            # DynamoMetricsConfig(
            #     enable=True,
            #     collect_kv_cache=True,
            #     collect_ttft=True,
            #     collect_itl=True,
            #     collect_inflight_requests=False,
            #     collect_throughput=False,
            #     collect_token_throughput=False,
            # )
        """
        return cls(
            enable=True,
            prometheus_url=prometheus_url,
            query_range=query_range,
            # Core metrics
            collect_kv_cache=True,
            collect_ttft=True,
            collect_itl=True,
            # Disable supplementary metrics
            collect_inflight_requests=False,
            collect_throughput=False,
            collect_token_throughput=False,
        )


class ProfilerConfig(BaseModel):

    base_metrics: bool = False
    token_usage_forecast: bool = False
    token_uniqueness_forecast: bool = False
    workflow_runtime_forecast: bool = False
    compute_llm_metrics: bool = False
    csv_exclude_io_text: bool = False
    prompt_caching_prefixes: PromptCachingConfig = PromptCachingConfig()
    bottleneck_analysis: BottleneckConfig = BottleneckConfig()
    concurrency_spike_analysis: ConcurrencySpikeConfig = ConcurrencySpikeConfig()
    prefix_span_analysis: PrefixSpanConfig = PrefixSpanConfig()
    prediction_trie: PredictionTrieConfig = PredictionTrieConfig()
    dynamo_metrics: DynamoMetricsConfig = DynamoMetricsConfig()
