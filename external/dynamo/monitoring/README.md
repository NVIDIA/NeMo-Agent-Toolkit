# Dynamo Monitoring Stack

This directory contains a Prometheus + Grafana monitoring setup for the Dynamo LLM inference stack with Thompson Sampling router. Metrics are collected at **2-second resolution** directly from the ai-dynamo Prometheus API for per-request granularity.

## Supported Backends

The monitoring stack supports both **SGLang** and **vLLM** backends:

| Backend | Metric Prefix | Startup Script | Features |
|---------|---------------|----------------|----------|
| SGLang | `sglang:` | `start_dynamo_optimized_thompson_hints_sglang.sh` | Fast inference |
| vLLM | `vllm:` | `start_dynamo_optimized_thompson_hints_vllm.sh` | Native KVBM support |

The Grafana dashboard includes a **Backend** dropdown selector to switch between SGLang and vLLM metrics dynamically.

## Quick Start

The monitoring stack starts **automatically** when you run the Dynamo startup script:

```bash
# Start Dynamo (monitoring starts automatically)
bash start_dynamo_optimized_thompson_hints_vllm.sh

# Or start monitoring manually if needed
cd monitoring
docker compose up -d
```

**Access the dashboards:**
- **Grafana**: http://localhost:3000 (no login required)
- **Prometheus**: http://localhost:9090

**Direct dashboard link:**
```
http://localhost:3000/d/dynamo-overview/dynamo-llm-overview
```

In Grafana, use the **Backend** dropdown to select `sglang` or `vllm` based on your deployment.

## Prerequisites

- Docker and Docker Compose
- Dynamo stack running (see `../start_dynamo_optimized_thompson_hints_sglang.sh` or `../start_dynamo_optimized_thompson_hints_vllm.sh`)

## Accessing Grafana Dashboard

### Local Access

If running on your local machine:

1. Open your browser
2. Navigate to: `http://localhost:3000/d/dynamo-overview/dynamo-llm-overview`
3. No login required (anonymous access enabled)
4. Use the **Backend** dropdown (top left) to select `sglang` or `vllm`
5. Use the **time filter** (top right) to adjust the time range

### Remote Access via SSH Tunnel

If Dynamo and monitoring are running on a remote server (for example, a GPU cluster), use SSH port forwarding:

**Step 1: Create SSH tunnel**
```bash
# Replace <USERNAME> and <REMOTE_HOST> with your credentials
ssh -L 3000:localhost:3000 <USERNAME>@<REMOTE_HOST>

# Example with VPN-accessible server:
ssh -L 3000:localhost:3000 myuser@10.57.201.5
```

**Step 2: Open browser**
Navigate to: `http://localhost:3000/d/dynamo-overview/dynamo-llm-overview`

**Step 3: Set time filter**
- Click the time picker in the top-right corner of Grafana
- Select a preset range (Last 1 hour, Last 6 hours, Last 24 hours)
- Or set a custom range to view historical data from previous benchmark runs

> **Tip**: Data persists across restarts. Zoom out to the last 12-24 hours to see multiple benchmark intervals.

### Viewing Historical Data

Prometheus stores metrics data persistently. To view data from previous runs:

1. Open the Grafana dashboard
2. Use the time picker (top right) to expand the time range
3. Look for intervals of activity separated by gaps
4. Compare KV Efficiency scores across different runs

**Example observation**: With a tool-calling agent (20 tools) on 4xH100 with 2 workers, you might see:
- Worker 18081: 25.4% average KV Efficiency
- Worker 18082: 16.4% average KV Efficiency

### Sharing Dashboard Access

Anyone with SSH access to the remote server can view the same data:

1. Share the SSH tunnel command with team members
2. They can connect and view real-time or historical metrics
3. Useful for collaborative debugging and performance analysis

## Architecture

The monitoring stack collects metrics from all Dynamo components. The architecture uses **model name isolation** to ensure all requests flow through the Thompson Sampling router.

### Request Flow (Model Name Isolation)

```
Client Request (with nvext.annotations)
      ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Default Dynamo Frontend (:8000)                                        │
│    - Tokenization + nvext parsing                                       │
│    - ETCD ModelWatcher (namespace=dynamo)                               │
│    - Routes to processor ONLY (workers use internal model name)         │
└─────────────────────────────────────────────────────────────────────────┘
      ↓ discovers processor (model: llama-3.3-70b)
┌─────────────────────────────────────────────────────────────────────────┐
│  Custom Processor (:18091/metrics)                                      │
│    - Extracts hints: prefix_id, total_requests, osl, iat                │
│    - Queries Thompson Sampling router                                   │
│    - Registered at: dynamo.backend.generate (namespace=dynamo)          │
└─────────────────────────────────────────────────────────────────────────┘
      ↓ queries router
┌─────────────────────────────────────────────────────────────────────────┐
│  Custom Router (:18090/metrics)                                         │
│    - Thompson Sampling + KV overlap scoring                             │
│    - Returns optimal worker_id                                          │
│    - Registered at: dynamo.router.{find_worker,feedback}                │
└─────────────────────────────────────────────────────────────────────────┘
      ↓ returns worker_id
┌─────────────────────────────────────────────────────────────────────────┐
│  vLLM and SGLang Workers (:18081, :18082, ... /metrics)                 │
│    - Registered at: workers.worker.generate (namespace=workers)         │
│    - Model: llama-3.3-70b-internal (hidden from frontend)               │
│    - Each worker uses TP_SIZE GPUs                                      │
└─────────────────────────────────────────────────────────────────────────┘
      ↓
Response + Feedback to Router
```

### Metrics Collection

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Dynamo Stack                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Frontend   │  │  Workers    │  │   Router    │  │  Processor  │         │
│  │  :8000      │  │ :18081-180xx│  │   :18090    │  │   :18091    │         │
│  │  /metrics   │  │  /metrics   │  │  /metrics   │  │  /metrics   │         │
│  │  (latency)  │  │  (KV cache) │  │  (routing)  │  │  (KVE)      │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
└─────────┼────────────────┼────────────────┼────────────────┼────────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Monitoring Stack                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                          Prometheus :9090                              │ │
│  │    Scrapes all endpoints every 2 seconds for per-request granularity:  │ │
│  │    - Frontend (:8000)        - latency, throughput, tokens             │ │
│  │    - Workers (:18081-180xx)  - KV cache, backend stats (per-worker)    │ │
│  │    - Router (:18090)         - Thompson Sampling routing metrics       │ │
│  │    - Processor (:18091)      - Thompson Sampling KVE metrics           │ │
│  └────────────────────────────────┬───────────────────────────────────────┘ │
│                                   │                                         │
│                                   ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                          Grafana :3000                                 │ │
│  │    Dashboard: "Dynamo LLM Overview"                                    │ │
│  │    URL: /d/dynamo-overview/dynamo-llm-overview                         │ │
│  │    Access: Anonymous (no login required)                               │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Model Name Isolation Explained

| Component | Model Name | Namespace | Purpose |
|-----------|------------|-----------|---------|
| Workers | `llama-3.3-70b-internal` | `workers` | Hidden from frontend discovery |
| Processor | `llama-3.3-70b` | `dynamo` | Discovered by frontend |
| Router | N/A | `dynamo` | Internal routing service |

This isolation ensures **ALL requests** go through the Thompson Sampling router, enabling:
- KV overlap-aware worker selection
- Workload hint extraction (`prefix_id`, `osl`, `iat`)
- Per-request feedback for router learning

## Metrics Endpoints

| Component | Port(s) | URL | Description |
|-----------|---------|-----|-------------|
| Frontend | 8000 | `http://localhost:8000/metrics` | User-facing metrics (latency, throughput, tokens) |
| Workers | 18081+ | `http://localhost:18081/metrics` | KV cache, backend stats - one port per worker |
| Router | 18090 | `http://localhost:18090/metrics` | Thompson Sampling routing decisions |
| Processor | 18091 | `http://localhost:18091/metrics` | Thompson Sampling KVE (KV Efficiency) metrics |

### Worker Port Allocation

Worker metrics ports are sequential starting at `DYNAMO_WORKER_METRICS_PORT` (default: 18081):

| Configuration | Workers | GPU Allocation | Metrics Ports |
|---------------|---------|----------------|---------------|
| 8 GPUs, TP=4 | 2 | GPUs 0-3, 4-7 | 18081, 18082 |
| 8 GPUs, TP=2 | 4 | GPUs 0-1, 2-3, 4-5, 6-7 | 18081-18084 |
| 4 GPUs, TP=2 | 2 | GPUs 0-1, 2-3 | 18081, 18082 |

Each worker is identified in Grafana by its metrics port (for example, `instance="localhost:18081"`).

## Key Metrics

### Frontend Metrics (`:8000/metrics`)

User-facing HTTP API metrics for latency, throughput, and token statistics.

| Prefix | Full Metric Name | Type | Description |
|--------|------------------|------|-------------|
| `dynamo_frontend_` | `dynamo_frontend_requests_total` | Counter | Total requests processed |
| `dynamo_frontend_` | `dynamo_frontend_inflight_requests` | Gauge | Currently processing requests |
| `dynamo_frontend_` | `dynamo_frontend_queued_requests` | Gauge | Requests waiting in queue |
| `dynamo_frontend_` | `dynamo_frontend_disconnected_clients` | Counter | Client disconnections |
| `dynamo_frontend_` | `dynamo_frontend_time_to_first_token_seconds` | Histogram | Time until first token generated |
| `dynamo_frontend_` | `dynamo_frontend_inter_token_latency_seconds` | Histogram | Time between consecutive tokens |
| `dynamo_frontend_` | `dynamo_frontend_request_duration_seconds` | Histogram | Total request duration |
| `dynamo_frontend_` | `dynamo_frontend_input_sequence_tokens` | Histogram | Input prompt length distribution |
| `dynamo_frontend_` | `dynamo_frontend_output_sequence_tokens` | Histogram | Output length distribution |
| `dynamo_frontend_` | `dynamo_frontend_output_tokens_total` | Counter | Total output tokens generated |
| `dynamo_frontend_` | `dynamo_frontend_model_context_length` | Gauge | Model context window size |
| `dynamo_frontend_` | `dynamo_frontend_model_kv_cache_block_size` | Gauge | KV cache block size |

### Worker Metrics (`:18081+/metrics`)

Backend worker metrics including KV cache, scheduling, and internal statistics. Both SGLang and vLLM expose similar metrics with different prefixes:
- **SGLang**: Metrics prefixed with `sglang:` (e.g., `sglang:cache_hit_rate`)
- **vLLM**: Metrics prefixed with `vllm:` (e.g., `vllm:cache_hit_rate`)

#### Dynamo Component Metrics

| Prefix | Full Metric Name | Type | Description |
|--------|------------------|------|-------------|
| `dynamo_component_kvstats_` | `dynamo_component_kvstats_gpu_cache_usage_percent` | Gauge | KV cache memory utilization (0-100) |
| `dynamo_component_kvstats_` | `dynamo_component_kvstats_gpu_prefix_cache_hit_rate` | Gauge | Prefix cache hit rate (0-1) |
| `dynamo_component_kvstats_` | `dynamo_component_kvstats_active_blocks` | Gauge | Active KV cache blocks |
| `dynamo_component_kvstats_` | `dynamo_component_kvstats_total_blocks` | Gauge | Total KV cache blocks |
| `dynamo_component_` | `dynamo_component_request_duration_seconds` | Histogram | Backend request processing time |
| `dynamo_component_` | `dynamo_component_requests_total` | Counter | Total requests to worker |
| `dynamo_component_` | `dynamo_component_inflight_requests` | Gauge | Requests currently in worker |
| `dynamo_component_` | `dynamo_component_uptime_seconds` | Gauge | Worker uptime |

#### Backend Native Metrics

Both SGLang and vLLM expose similar native metrics with their respective prefixes. Use the `${backend}` variable in the Grafana dashboard to switch between them.

**Common metrics across both backends:**

| Metric (use `${backend}:` prefix) | Type | Description |
|-----------------------------------|------|-------------|
| `cache_hit_rate` | Gauge | Prefix cache hit rate |
| `token_usage` | Gauge | Current token usage |
| `num_running_reqs` | Gauge | Currently running requests |
| `num_queue_reqs` | Gauge | Queued requests |
| `num_used_tokens` | Gauge | Tokens currently in use |
| `gen_throughput` | Gauge | Generation throughput |

**SGLang-specific metrics:**

| Prefix | Full Metric Name | Type | Description |
|--------|------------------|------|-------------|
| `sglang:` | `sglang:utilization` | Gauge | GPU utilization |
| `sglang:` | `sglang:queue_time_seconds` | Histogram | Time spent in queue |
| `sglang:` | `sglang:per_stage_req_latency_seconds` | Histogram | Per-stage request latency |
| `sglang:` | `sglang:kv_transfer_latency_ms` | Gauge | KV transfer latency |
| `sglang:` | `sglang:kv_transfer_speed_gb_s` | Gauge | KV transfer speed |
| `sglang:` | `sglang:engine_startup_time` | Gauge | Engine startup duration |
| `sglang:` | `sglang:engine_load_weights_time` | Gauge | Model weight loading time |

**vLLM-specific metrics:**

| Prefix | Full Metric Name | Type | Description |
|--------|------------------|------|-------------|
| `vllm:` | `vllm:gpu_cache_usage_perc` | Gauge | GPU KV cache usage percentage |
| `vllm:` | `vllm:cpu_cache_usage_perc` | Gauge | CPU KV cache usage percentage |
| `vllm:` | `vllm:num_requests_running` | Gauge | Currently running requests |
| `vllm:` | `vllm:num_requests_waiting` | Gauge | Waiting requests in queue |
| `vllm:` | `vllm:generation_tokens_total` | Counter | Total generation tokens |
| `vllm:` | `vllm:prompt_tokens_total` | Counter | Total prompt tokens |

### Router Metrics (`:18090/metrics`)

Dynamo component metrics for the Thompson Sampling router (uses standard `dynamo_component_*` prefix).

| Prefix | Full Metric Name | Type | Description |
|--------|------------------|------|-------------|
| `dynamo_component_` | `dynamo_component_requests_total` | Counter | Total routing requests (labeled by endpoint) |
| `dynamo_component_` | `dynamo_component_request_duration_seconds` | Histogram | Routing decision latency |
| `dynamo_component_` | `dynamo_component_request_bytes_total` | Counter | Request payload bytes |
| `dynamo_component_` | `dynamo_component_response_bytes_total` | Counter | Response payload bytes |
| `dynamo_component_` | `dynamo_component_inflight_requests` | Gauge | In-flight routing requests |
| `dynamo_component_` | `dynamo_component_uptime_seconds` | Gauge | Router uptime |
| `dynamo_component_nats_` | `dynamo_component_nats_service_requests_total` | Gauge | NATS service requests |
| `dynamo_component_nats_` | `dynamo_component_nats_service_processing_ms_avg` | Gauge | Average NATS processing time |
| `dynamo_component_nats_` | `dynamo_component_nats_client_connection_state` | Gauge | NATS connection state (0=disconnected, 1=connected) |

**Router Endpoints** (use `dynamo_endpoint` label to filter):
- `find_worker` - Worker selection requests
- `feedback` - Feedback from completed requests

### Thompson Sampling Processor Metrics (`:18091/metrics`)

Custom Thompson Sampling KV Efficiency (KVE) metrics from the processor component.

| Prefix | Full Metric Name | Type | Description |
|--------|------------------|------|-------------|
| `dynamo_component_thompson_` | `dynamo_component_thompson_requests_total` | Counter | Total requests processed |
| `dynamo_component_thompson_` | `dynamo_component_thompson_request_latency_seconds` | Histogram | End-to-end request latency |
| `dynamo_component_thompson_` | `dynamo_component_thompson_tokens_in_total` | Counter | Total input tokens |
| `dynamo_component_thompson_` | `dynamo_component_thompson_tokens_out_total` | Counter | Total output tokens |
| `dynamo_component_thompson_` | `dynamo_component_thompson_routing_decisions_total` | Counter | Routing decisions made |
| `dynamo_component_thompson_` | `dynamo_component_thompson_active_requests` | Gauge | Currently processing requests |
| `dynamo_component_thompson_` | `dynamo_component_thompson_router_errors_total` | Counter | Router communication errors |
| `dynamo_component_thompson_` | `dynamo_component_thompson_engine_errors_total` | Counter | Engine or worker errors |
| `dynamo_component_thompson_kve_` | `dynamo_component_thompson_kve_prompt_tokens_total` | Counter | Total prompt tokens (KVE denominator) |
| `dynamo_component_thompson_kve_` | `dynamo_component_thompson_kve_cached_tokens_total` | Counter | Cached tokens hit (KVE numerator) |
| `dynamo_component_thompson_kve_` | `dynamo_component_thompson_kve_device_blocks_total` | Counter | KV blocks from GPU memory |
| `dynamo_component_thompson_kve_` | `dynamo_component_thompson_kve_host_blocks_total` | Counter | KV blocks from CPU memory (not yet implemented) |
| `dynamo_component_thompson_kve_` | `dynamo_component_thompson_kve_disk_blocks_total` | Counter | KV blocks from disk (not yet implemented) |

**KV Cache Efficiency Score (KVES) Calculation:**

The full KVES formula is:
```
KVES = (TotalWork - ActualWork) / TotalWork ∈ [0,1]
     where 0 = no cache benefit, 1 = full reuse

ActualWork = <w_hit, h> + w_compute * recomputed_prefill_blocks * block_size
TotalWork = cached_prompt_blocks * block_size
w_hit = (w_gpu_hit, w_cpu_hit, w_disk_hit)  # weights per hit source
```

Since full KVES requires GPU, CPU, and disk hit breakdowns, we use a **simplified KVES proxy** based on cache hit rate. CPU and disk hit penalties (`w_cpu_hit`, `w_disk_hit`) are not yet implemented — the corresponding `host_blocks` and `disk_blocks` counters are placeholders left for future integration with the Dynamo team once tiered KV cache eviction surfaces per-tier hit counts.

**Note**: vLLM with KVBM enabled provides richer KV cache metrics than SGLang.

```promql
# KVES Proxy (using SGLang native metric - RECOMMENDED)
sglang:cache_hit_rate

# As percentage
sglang:cache_hit_rate * 100
```

> **Why use the native SGLang metric?** SGLang computes cache hit rate internally but does not include
> `cached_tokens` in its API responses. The `thompson_kve_*` counters from the processor will show 0
> unless the underlying engine provides `usage.prompt_tokens_details.cached_tokens`.

> **Note on Full KVES**: CPU and disk hit penalties are **not yet implemented**. The `w_cpu_hit` and
> `w_disk_hit` weights in the full KVES equation require per-tier hit breakdowns from the inference
> engine, which are not currently exposed. This is left for future integration with the Dynamo team
> once vLLM with KVBM (or equivalent) surfaces GPU→CPU→Disk tiered cache hit counts through its API.

## KV Cache Metrics Status

This section documents the working status of all KV cache-related metrics across the Dynamo stack.

**Backend Selection**: The Grafana dashboard uses a `${backend}` template variable. Select `sglang` or `vllm` from the dropdown to switch all backend-specific queries.

### Working Metrics ✓

| Prefix | Full Metric Name | Status | Description |
|--------|------------------|--------|-------------|
| `sglang:` | `sglang:token_usage` | ✓ **WORKING** | KV cache memory usage as ratio (0-1). Multiply by 100 for percentage. |
| `sglang:` | `sglang:num_used_tokens` | ✓ **WORKING** | Absolute number of tokens currently stored in KV cache. |
| `dynamo_component_kvstats_` | `dynamo_component_kvstats_total_blocks` | ✓ **WORKING** | Total KV cache blocks available (capacity). |
| `sglang:` | `sglang:gen_throughput` | ✓ **WORKING** | Token generation throughput (tokens/sec). |

### Conditionally Working Metrics ⚠

| Prefix | Full Metric Name | Status | Notes |
|--------|------------------|--------|-------|
| `sglang:` | `sglang:cache_hit_rate` | ⚠ **CONDITIONAL** | Shows prefix cache hit rate (0-1). Requires repeated queries with shared prefixes to see non-zero values. May stay at 0 if prefix caching is not effective for workload. |

### Not Implemented / Always Zero Metrics

| Prefix | Full Metric Name | Status | Notes |
|--------|------------------|--------|-------|
| `sglang:` | `sglang:utilization` | ✗ **ALWAYS 0** | Exported but not populated in unified engine mode. Use `sglang:num_running_reqs` and `sglang:gen_throughput` instead to gauge worker activity. |
| `sglang:` | `sglang:is_cuda_graph` | ✗ **ALWAYS 0** | CUDA graph optimization not enabled in current configuration. |
| `sglang:` | `sglang:spec_accept_*` | ✗ **ALWAYS 0** | Speculative decoding metrics - not applicable without draft model. |

### Non-Working Metrics ✗

| Prefix | Full Metric Name | Status | Reason |
|--------|------------------|--------|--------|
| `dynamo_component_kvstats_` | `dynamo_component_kvstats_gpu_cache_usage_percent` | ✗ **NOT WORKING** | Internal Dynamo metric not populated by the SGLang backend. Use `sglang:token_usage * 100` instead. |
| `dynamo_component_kvstats_` | `dynamo_component_kvstats_gpu_prefix_cache_hit_rate` | ✗ **NOT WORKING** | Internal Dynamo metric not populated. Use `sglang:cache_hit_rate` instead. |
| `dynamo_component_kvstats_` | `dynamo_component_kvstats_active_blocks` | ✗ **NOT WORKING** | Internal Dynamo metric not populated by the SGLang backend. |
| `dynamo_component_thompson_kve_` | `dynamo_component_thompson_kve_cached_tokens_total` | ✗ **NOT WORKING** | SGLang API doesn't return `cached_tokens` in response. |
| `dynamo_component_thompson_kve_` | `dynamo_component_thompson_kve_prompt_tokens_total` | ✗ **NOT WORKING** | Counter stays at 0 due to API limitation. |
| `dynamo_component_thompson_kve_` | `dynamo_component_thompson_kve_*_blocks_total` | ✗ **NOT WORKING** | Block-level KVE metrics not populated. |

### Architecture-Specific Metrics (Always Zero for Llama)

| Prefix | Full Metric Name | Status | Reason |
|--------|------------------|--------|--------|
| `sglang:` | `sglang:swa_token_usage` | N/A | Sliding Window Attention - not used by Llama architecture. |
| `sglang:` | `sglang:mamba_usage` | N/A | Mamba architecture metric - not applicable to Llama. |
| `sglang:` | `sglang:kv_transfer_*` | N/A | KV transfer metrics only used in disaggregated prefill and decode modes. |
| `sglang:` | `sglang:pending_prealloc_token_usage` | N/A | Pre-allocation metric - typically 0 in standard operation. |

### Recommended KV Cache Queries

The following queries use `${backend}` variable (set to `sglang` or `vllm` in Grafana):

```promql
# KV Cache Memory Usage % (RECOMMENDED - works with both backends!)
${backend}:token_usage * 100

# Absolute tokens in KV cache
${backend}:num_used_tokens

# Total KV cache capacity (blocks)
dynamo_component_kvstats_total_blocks

# Prefix Cache Hit Rate % (may be 0 without repeated prefix queries)
${backend}:cache_hit_rate * 100

# Token throughput
${backend}:gen_throughput
```

**Direct queries** (without variable):
```promql
# SGLang specific
sglang:token_usage * 100
sglang:cache_hit_rate * 100

# vLLM specific
vllm:token_usage * 100
vllm:cache_hit_rate * 100
```

## Grafana Dashboard

### Dashboard Access

| Property | Value |
|----------|-------|
| Dashboard Name | Dynamo LLM Overview |
| Direct URL | `http://localhost:3000/d/dynamo-overview/dynamo-llm-overview` |
| Authentication | None required (anonymous access enabled) |
| Data Refresh | Every 2 seconds (configurable) |
| Data Retention | Persistent (survives restarts) |

### Backend Selector

The dashboard includes a **Backend** dropdown variable at the top. Select:
- **`sglang`** - For SGLang workers (metrics prefixed with `sglang:`)
- **`vllm`** - For vLLM workers (metrics prefixed with `vllm:`)

All backend-specific panels automatically update based on your selection.

### Time Controls

Use the time picker (top right) to:
- Select preset ranges: Last 5 minutes, Last 1 hour, Last 6 hours, Last 24 hours
- Set custom absolute time ranges for specific benchmark intervals
- Use the refresh dropdown to control auto-refresh frequency

### Dashboard Panels

1. **Inflight Requests** (stat) — Current in-flight request count
   - `dynamo_frontend_inflight_requests`
2. **Requests (1m)** (stat) — Recent request throughput
   - `sum(increase(dynamo_frontend_requests_total[10s]))`
3. **Time to First Token (TTFT)** (time series) — [P50, P95, P99] latency to first generated token
   - `histogram_quantile(0.5, rate(dynamo_frontend_time_to_first_token_seconds_bucket[10s]))`
   - `histogram_quantile(0.95, ...)`
   - `histogram_quantile(0.99, ...)`
4. **Inter-Token Latency (ITL)** (time series) — [P50, P95, P99] latency between tokens
   - `histogram_quantile(0.5, rate(dynamo_frontend_inter_token_latency_seconds_bucket[10s]))`
   - `histogram_quantile(0.95, ...)`
   - `histogram_quantile(0.99, ...)`
5. **Token Throughput** (time series) — Per-worker and aggregate generation throughput
   - `${backend}:gen_throughput` (per worker)
   - `sum(${backend}:gen_throughput)` (aggregate)
   - `rate(dynamo_frontend_output_tokens_total{job="dynamo-frontend"}[10s])` (frontend-side)
6. **Request Flow (Frontend → Processor → Router → Workers)** (time series) — End-to-end request rates through each component
   - `sum(rate(dynamo_frontend_requests_total[10s]))` (frontend)
   - `sum(rate(dynamo_component_requests_total{dynamo_namespace="dynamo",dynamo_component="backend"}[10s]))` (processor)
   - `sum(rate(dynamo_component_requests_total{...dynamo_component="router",dynamo_endpoint="find_worker"}[10s]))` (router)
   - `rate(dynamo_component_requests_total{dynamo_namespace="workers",...,dynamo_endpoint="generate"}[10s])` (per worker)
   - `sum(...)` (aggregate workers)
7. **Worker Queue Depth** (time series) — Pending requests per worker
   - `${backend}:num_queue_reqs`
8. **Worker Activity (Running Requests)** (time series) — Active requests per worker
   - `${backend}:num_running_reqs`
9. **KV Cache Details (Per-Worker)** (time series) — Detailed per-worker cache state
   - `avg_over_time(${backend}:cache_hit_efficiency[1m]) * 100` (KVES proxy %)
   - `avg_over_time(${backend}:token_usage[1m]) * 100` (KV usage %)
   - `last_over_time(${backend}:num_used_tokens[1m])` (tokens used)
   - `last_over_time(dynamo_component_kvstats_total_blocks[1m])` (capacity in blocks)
   - `max(dynamo_frontend_model_kv_cache_block_size{job="dynamo-frontend"})` (block size)
10. **KVES Proxy by Worker** (time series) — Cache hit efficiency per worker (0–1 scale)
    - `${backend}:cache_hit_efficiency`
11. **KV Cache Usage & Tokens** (time series) — Memory utilization and token counts
    - `${backend}:token_usage * 100` (usage %)
    - `${backend}:num_used_tokens` (absolute tokens)

> **Note on KV Cache Metrics**: The dashboard uses backend-native metrics (`${backend}:token_usage`,
> `${backend}:cache_hit_efficiency`, `${backend}:num_used_tokens`) which are reliably populated by both
> SGLang and vLLM. The Dynamo-specific `dynamo_component_kvstats_*` metrics may not be populated
> depending on your backend configuration. See the "KV Cache Metrics Status" section above for details.

## Files

```
monitoring/
├── docker-compose.yml              # Prometheus + Grafana services (ports templated from DYNAMO_* environment variables)
├── prometheus.yml                  # Prometheus scrape config template (placeholders substituted at startup)
├── README.md                       # This file
├── rules/
│   ├── sglang-aliases.yml          # Recording rules mapping SGLang metrics to dashboard queries
│   └── vllm-aliases.yml            # Recording rules mapping vLLM metrics to dashboard queries
├── scripts/
│   └── kv_event_observer.py        # KV cache event observer utility
└── grafana/
    └── provisioning/
        ├── datasources/
        │   └── datasources.yml     # Prometheus datasource config
        └── dashboards/
            ├── dashboards.yml      # Dashboard provider config
            └── json/
                └── dynamo-overview.json  # Pre-built dashboard
```

## Usage

### Automatic Startup (Recommended)

The monitoring stack starts **automatically** when you run the Dynamo startup script:

```bash
# Start Dynamo with monitoring (vLLM backend)
bash start_dynamo_optimized_thompson_hints_vllm.sh

# Or SGLang backend
bash start_dynamo_optimized_thompson_hints_sglang.sh
```

The script will:
1. Start etcd and NATS infrastructure
2. Start Prometheus and Grafana containers
3. Wait for monitoring services to be ready
4. Start Dynamo components (workers, router, processor, frontend)

### Manual Startup

If you need to start monitoring separately:

```bash
cd monitoring
docker compose up -d
```

### Stop Monitoring

```bash
docker compose down
```

### View Logs

```bash
docker compose logs -f prometheus
docker compose logs -f grafana
```

### Reset Data (Start Fresh)

```bash
docker compose down -v  # Removes ALL volumes (Prometheus + Grafana data)
docker compose up -d
```

### Clear Prometheus Data Only

If you're seeing duplicate labels in Grafana (for example, after restarting workers with new IDs), you can clear just the Prometheus data while keeping Grafana settings:

```bash
# Stop the monitoring containers
docker stop dynamo-prometheus dynamo-grafana
docker rm dynamo-prometheus dynamo-grafana

# Remove just the Prometheus data volume (clears all historical metrics)
docker volume rm monitoring_prometheus_data && echo "Prometheus data volume removed (old metrics cleared)"

# Restart the monitoring stack with fresh data
docker compose up -d
```

Alternatively, use the stop script with the `--kill-metrics` flag:

```bash
# From the dynamo directory
bash stop_dynamo.sh --kill-metrics

# Then remove the Prometheus volume
docker volume rm monitoring_prometheus_data

# Restart everything (monitoring will start automatically)
bash start_dynamo_optimized_thompson_hints_vllm.sh
```

## Remote Access via SSH Port Forwarding

If the monitoring stack is running on a remote GPU server (for example, a leased cluster node), use SSH port forwarding to access Grafana and Prometheus from your local machine.

### Step-by-Step Remote Access

**1. Create SSH tunnel to the remote server:**

```bash
# General syntax
ssh -L 3000:localhost:3000 <USERNAME>@<REMOTE_HOST>

# Example with VPN-accessible server
ssh -L 3000:localhost:3000 myuser@10.57.201.5
```

**2. Open the Grafana dashboard in your browser:**

```
http://localhost:3000/d/dynamo-overview/dynamo-llm-overview
```

**3. Configure the time range:**
- Click the time picker (top right corner of Grafana UI)
- Select a preset: Last 1 hour, Last 6 hours, Last 12 hours, Last 24 hours
- Or set a custom absolute time range to view specific benchmark intervals

**4. Select your backend:**
- Use the **Backend** dropdown (top left) to choose `sglang` or `vllm`
- All panels will automatically update to show backend-specific metrics

### Sharing Data with Team Members

Anyone with SSH access to the same server can view the monitoring data:

```bash
# Team member creates their own tunnel
ssh -L 3000:localhost:3000 <THEIR_USERNAME>@<REMOTE_HOST>

# Then opens the same dashboard URL
# http://localhost:3000/d/dynamo-overview/dynamo-llm-overview
```

This enables collaborative analysis - multiple people can view the same data simultaneously to focus on specific signals.

### Forward Multiple Ports

To access both Grafana and Prometheus simultaneously:

```bash
ssh -L 3000:localhost:3000 -L 9090:localhost:9090 <USERNAME>@<REMOTE_HOST>
```

Access:
- Grafana: `http://localhost:3000/d/dynamo-overview/dynamo-llm-overview`
- Prometheus: `http://localhost:9090`

### Background SSH Tunnel

To run the tunnel in the background (stays open after terminal closes):

```bash
ssh -f -N -L 3000:localhost:3000 -L 9090:localhost:9090 <USERNAME>@<REMOTE_HOST>
```

- `-f`: Run in background after authentication
- `-N`: Don't execute remote commands (tunnel only)

To kill a background tunnel:
```bash
# Find the SSH process
ps aux | grep "ssh -f -N -L 3000"

# Kill it
kill <PID>
```

### Viewing Historical Benchmark Data

Prometheus persists all metrics data. To view historical benchmarks:

1. Open the Grafana dashboard
2. Expand the time range using the time picker (top right)
3. Zoom out to 12-24 hours to see multiple benchmark intervals
4. Gaps between data intervals indicate periods when Dynamo was stopped

**Example**: After running multiple benchmark sessions, you might see:
- Interval 1: Baseline configuration
- Interval 2: Optimized parameters (small gap)
- Interval 3: Best KV Efficiency (for example, Worker 18081: 25.4%, Worker 18082: 16.4%)

## Manual Metrics Queries

### Prometheus UI (http://localhost:9090)

Example queries:

```promql
# Request rate (requests/second)
rate(dynamo_frontend_requests_total[1m])

# P95 Time to First Token
histogram_quantile(0.95, rate(dynamo_frontend_time_to_first_token_seconds_bucket[5m]))

# P99 Inter-Token Latency
histogram_quantile(0.99, rate(dynamo_frontend_inter_token_latency_seconds_bucket[5m]))

# Token throughput
rate(dynamo_frontend_output_tokens_total[1m])

# KV cache hit rate (Dynamo)
dynamo_component_kvstats_gpu_prefix_cache_hit_rate

# KV cache hit rate (SGLang native)
sglang:cache_hit_rate

# KV cache usage percentage
dynamo_component_kvstats_gpu_cache_usage_percent

# Thompson routing decisions rate
rate(dynamo_component_thompson_routing_decisions_total[5m])

# KV Efficiency / Cache Hit Rate (using SGLang native - RECOMMENDED)
sglang:cache_hit_rate * 100

# Router endpoint request rate
rate(dynamo_component_requests_total{dynamo_component="router"}[5m])

# Worker queue depth
sglang:num_queue_reqs
```

### curl

```bash
# All frontend metrics
curl -s http://localhost:8000/metrics

# All worker metrics (Worker 0)
curl -s http://localhost:18081/metrics

# All worker metrics (Worker 1, if running multiple workers)
curl -s http://localhost:18082/metrics

# All router metrics
curl -s http://localhost:18090/metrics

# All processor metrics (Thompson Sampling)
curl -s http://localhost:18091/metrics

# Filter specific metrics
curl -s http://localhost:8000/metrics | grep time_to_first_token
curl -s http://localhost:18081/metrics | grep kvstats
curl -s http://localhost:18081/metrics | grep "sglang:"   # SGLang backend
curl -s http://localhost:18081/metrics | grep "vllm:"     # vLLM backend
curl -s http://localhost:18091/metrics | grep thompson
```

## Troubleshooting

### Prometheus can't scrape targets

Check if Dynamo is running:
```bash
# Check frontend health
curl http://localhost:8000/health

# Check worker metrics (Worker 0)
curl http://localhost:18081/metrics

# Check router metrics
curl http://localhost:18090/metrics

# Check processor metrics
curl http://localhost:18091/metrics
```

### Grafana shows "No data"

1. **Verify Prometheus is scraping**: http://localhost:9090/targets
   - All targets should show "UP" state
   - Check for scrape errors in the "Error" column

2. **Check if metrics exist**: http://localhost:9090/graph
   - Query a metric name (for example, `dynamo_frontend_requests_total`)
   - If no data, Dynamo may not be running or generating traffic

3. **Ensure time range is correct in Grafana**:
   - Click the time picker (top right)
   - Select "Last 1 hour" or expand to see historical data
   - If you just started, wait 30-60 seconds for initial data

4. **Check backend selector**:
   - Make sure the Backend dropdown matches your deployment (`sglang` vs `vllm`)
   - Backend mismatch will result in empty panels

### SSH tunnel issues

If you can't access Grafana via SSH tunnel:

```bash
# Verify the tunnel is active
ps aux | grep "ssh -L 3000"

# Test if port 3000 is accessible locally
curl -s http://localhost:3000/api/health

# If "connection refused", recreate the tunnel
ssh -L 3000:localhost:3000 <USERNAME>@<REMOTE_HOST>
```

### Port conflicts

If ports 9090 or 3000 are in use, modify `docker-compose.yml`:
```yaml
# Change Prometheus port
command:
  - '--web.listen-address=:9091'  # Different port

# Change Grafana port
environment:
  - GF_SERVER_HTTP_PORT=3001  # Different port
```

### Stale metrics after restart

If you see old worker instances in Grafana after restarting Dynamo:

```bash
# Clear Prometheus data and restart
docker stop dynamo-prometheus
docker rm dynamo-prometheus
docker volume rm monitoring_prometheus_data
cd monitoring && docker compose up -d
```

## Complete Metrics Reference

### Summary by Component

| Component | Port(s) | Metric Count | Key Prefixes |
|-----------|---------|--------------|--------------|
| Frontend | 8000 | ~22 | `dynamo_frontend_*` |
| Workers | 18081+ | ~50 | `dynamo_component_kvstats_*`, `sglang:*` or `vllm:*` |
| Router | 18090 | ~20 | `dynamo_component_*` (labeled `router`) |
| Processor | 18091 | ~35 | `dynamo_component_thompson_*` |

### All Metric Names by Component

<details>
<summary><b>Frontend (port 8000) - 22 metrics</b></summary>

```
dynamo_frontend_disconnected_clients
dynamo_frontend_inflight_requests
dynamo_frontend_input_sequence_tokens_{bucket,count,sum}
dynamo_frontend_inter_token_latency_seconds_{bucket,count,sum}
dynamo_frontend_model_context_length
dynamo_frontend_model_kv_cache_block_size
dynamo_frontend_model_migration_limit
dynamo_frontend_output_sequence_tokens_{bucket,count,sum}
dynamo_frontend_output_tokens_total
dynamo_frontend_queued_requests
dynamo_frontend_request_duration_seconds_{bucket,count,sum}
dynamo_frontend_requests_total
dynamo_frontend_time_to_first_token_seconds_{bucket,count,sum}
```
</details>

<details>
<summary><b>Worker (ports 18081+) - 50 metrics per worker</b></summary>

**Dynamo Component Metrics:**
```
dynamo_component_inflight_requests
dynamo_component_kvstats_active_blocks
dynamo_component_kvstats_gpu_cache_usage_percent
dynamo_component_kvstats_gpu_prefix_cache_hit_rate
dynamo_component_kvstats_total_blocks
dynamo_component_nats_client_*
dynamo_component_nats_service_*
dynamo_component_request_bytes_total
dynamo_component_request_duration_seconds_{bucket,count,sum}
dynamo_component_requests_total
dynamo_component_response_bytes_total
dynamo_component_uptime_seconds
```

**SGLang Native Metrics:**
```
sglang:cache_hit_rate
sglang:engine_load_weights_time
sglang:engine_startup_time
sglang:gen_throughput
sglang:is_cuda_graph
sglang:kv_transfer_*
sglang:mamba_usage
sglang:num_decode_prealloc_queue_reqs
sglang:num_decode_transfer_queue_reqs
sglang:num_grammar_queue_reqs
sglang:num_paused_reqs
sglang:num_prefill_inflight_queue_reqs
sglang:num_prefill_prealloc_queue_reqs
sglang:num_queue_reqs
sglang:num_retracted_reqs
sglang:num_running_reqs
sglang:num_running_reqs_offline_batch
sglang:num_used_tokens
sglang:pending_prealloc_token_usage
sglang:per_stage_req_latency_seconds_{bucket,count,sum}
sglang:queue_time_seconds_{bucket,count,sum}
sglang:spec_accept_length
sglang:spec_accept_rate
sglang:swa_token_usage
sglang:token_usage
sglang:utilization
```
</details>

<details>
<summary><b>Router (port 18090) - 20 metrics</b></summary>

```
dynamo_component_inflight_requests{dynamo_component="router"}
dynamo_component_nats_client_connection_state
dynamo_component_nats_client_current_connections
dynamo_component_nats_client_in_messages
dynamo_component_nats_client_in_total_bytes
dynamo_component_nats_client_out_messages
dynamo_component_nats_client_out_overhead_bytes
dynamo_component_nats_service_active_endpoints
dynamo_component_nats_service_active_services
dynamo_component_nats_service_errors_total
dynamo_component_nats_service_processing_ms_avg
dynamo_component_nats_service_processing_ms_total
dynamo_component_nats_service_requests_total
dynamo_component_request_bytes_total{dynamo_endpoint="find_worker|feedback"}
dynamo_component_request_duration_seconds_{bucket,count,sum}
dynamo_component_requests_total
dynamo_component_response_bytes_total
dynamo_component_uptime_seconds
```
</details>

<details>
<summary><b>Processor (port 18091) - 35 metrics</b></summary>

**Standard Dynamo Component Metrics:**
```
dynamo_component_inflight_requests
dynamo_component_nats_client_*
dynamo_component_nats_service_*
dynamo_component_request_bytes_total
dynamo_component_request_duration_seconds_{bucket,count,sum}
dynamo_component_requests_total
dynamo_component_response_bytes_total
dynamo_component_uptime_seconds
```

**Thompson Sampling Custom Metrics:**
```
dynamo_component_thompson_active_requests
dynamo_component_thompson_engine_errors_total
dynamo_component_thompson_kve_cached_tokens_total
dynamo_component_thompson_kve_device_blocks_total
dynamo_component_thompson_kve_disk_blocks_total
dynamo_component_thompson_kve_host_blocks_total
dynamo_component_thompson_kve_prompt_tokens_total
dynamo_component_thompson_request_latency_seconds_{bucket,count,sum}
dynamo_component_thompson_requests_total
dynamo_component_thompson_router_errors_total
dynamo_component_thompson_routing_decisions_total
dynamo_component_thompson_tokens_in_total
dynamo_component_thompson_tokens_out_total
```
</details>

