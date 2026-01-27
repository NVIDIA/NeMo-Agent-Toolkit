# Dynamo Monitoring Stack

This directory contains a Prometheus + Grafana monitoring setup for the Dynamo LLM inference stack with Thompson Sampling router.

## Supported Backends

The monitoring stack supports both **SGLang** and **vLLM** backends:

| Backend | Metric Prefix | Startup Script | Features |
|---------|---------------|----------------|----------|
| SGLang | `sglang:` | `start_dynamo_optimized_thompson_hints_sglang.sh` | Fast inference |
| vLLM | `vllm:` | `start_dynamo_optimized_thompson_hints_vllm.sh` | Native KVBM support |

The Grafana dashboard includes a **Backend** dropdown selector to switch between SGLang and vLLM metrics dynamically.

## Quick Start

```bash
# Start the monitoring stack
cd monitoring
docker compose up -d

# Access the dashboards
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)

# In Grafana, use the "Backend" dropdown to select sglang or vllm
```

## Prerequisites

- Docker and Docker Compose
- Dynamo stack running (see `../start_dynamo_optimized_thompson_hints_sglang.sh` or `../start_dynamo_optimized_thompson_hints_vllm.sh`)

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              Dynamo Stack                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Frontend   │  │   Worker    │  │   Router    │  │  Processor  │         │
│  │  :8000      │  │   :8081     │  │   :8082     │  │   :8083     │         │
│  │  /metrics   │  │  /metrics   │  │  /metrics   │  │  /metrics   │         │
│  │  (latency)  │  │  (KV cache) │  │  (routing)  │  │  (KVE)      │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
└─────────┼────────────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Monitoring Stack                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                          Prometheus :9090                              │ │
│  │    Scrapes all 4 endpoints every 5 seconds:                           │ │
│  │    - Frontend (:8000) - latency, throughput, tokens                   │ │
│  │    - Worker (:8081)   - KV cache, NATS, internal stats                │ │
│  │    - Router (:8082)   - Thompson Sampling routing metrics             │ │
│  │    - Processor (:8083) - Thompson Sampling KVE metrics                │ │
│  └────────────────────────────────┬───────────────────────────────────────┘ │
│                                   │                                         │
│                                   ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                          Grafana :3000                                 │ │
│  │    Pre-configured dashboard: "Dynamo LLM Overview"                    │ │
│  │    Login: admin / admin                                                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Metrics Endpoints

| Component | Port | URL | Description |
|-----------|------|-----|-------------|
| Frontend | 8000 | `http://localhost:8000/metrics` | User-facing metrics (latency, throughput) |
| Workers | 18081-180xx | `http://localhost:18081/metrics` | Internal metrics (KV cache, NATS stats) - one port per worker |
| Router | 18090 | `http://localhost:18090/metrics` | Thompson Sampling routing metrics |
| Processor | 18091 | `http://localhost:18091/metrics` | Thompson Sampling KVE metrics |

**Note**: Worker metrics ports are sequential starting at 18081. With 2 workers: 18081, 18082. With 4 workers: 18081-18084.

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

### Worker Metrics (`:8081/metrics`)

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

### Router Metrics (`:8082/metrics`)

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

### Thompson Sampling Processor Metrics (`:8083/metrics`)

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
| `dynamo_component_thompson_` | `dynamo_component_thompson_engine_errors_total` | Counter | Engine/worker errors |
| `dynamo_component_thompson_kve_` | `dynamo_component_thompson_kve_prompt_tokens_total` | Counter | Total prompt tokens (KVE denominator) |
| `dynamo_component_thompson_kve_` | `dynamo_component_thompson_kve_cached_tokens_total` | Counter | Cached tokens hit (KVE numerator) |
| `dynamo_component_thompson_kve_` | `dynamo_component_thompson_kve_device_blocks_total` | Counter | KV blocks from GPU memory |
| `dynamo_component_thompson_kve_` | `dynamo_component_thompson_kve_host_blocks_total` | Counter | KV blocks from CPU memory |
| `dynamo_component_thompson_kve_` | `dynamo_component_thompson_kve_disk_blocks_total` | Counter | KV blocks from disk |

**KV Cache Efficiency Score (KVES) Calculation:**

The full KVES formula is:
```
KVES = (TotalWork - ActualWork) / TotalWork ∈ [0,1]
     where 0 = no cache benefit, 1 = full reuse

ActualWork = <w_hit, h> + w_compute * recomputed_prefill_blocks * block_size
TotalWork = cached_prompt_blocks * block_size
w_hit = (w_gpu_hit, w_cpu_hit, w_disk_hit)  # weights per hit source
```

Since full KVES requires GPU/CPU/disk hit breakdowns, we use a **simplified KVES proxy** based on cache hit rate:

**Note**: vLLM with KVBM enabled provides richer KV cache metrics than SGLang.

```promql
# KVES Proxy (using SGLang native metric - RECOMMENDED)
sglang:cache_hit_rate

# As percentage
sglang:cache_hit_rate * 100
```

> **Why use SGLang's native metric?** SGLang computes cache hit rate internally but doesn't include
> `cached_tokens` in its API responses. The processor's `thompson_kve_*` counters will show 0
> unless the underlying engine provides `usage.prompt_tokens_details.cached_tokens`.

> **Note on Full KVES**: To implement the full KVES equation with CPU/disk hit weights, use
> vLLM with KVBM enabled, which provides GPU→CPU→Disk tiered caching with proper metrics.

## KV Cache Metrics Status

This section documents the working status of all KV cache related metrics across the Dynamo stack.

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
| `dynamo_component_kvstats_` | `dynamo_component_kvstats_gpu_cache_usage_percent` | ✗ **NOT WORKING** | Dynamo's internal metric not populated by SGLang backend. Use `sglang:token_usage * 100` instead. |
| `dynamo_component_kvstats_` | `dynamo_component_kvstats_gpu_prefix_cache_hit_rate` | ✗ **NOT WORKING** | Dynamo's internal metric not populated. Use `sglang:cache_hit_rate` instead. |
| `dynamo_component_kvstats_` | `dynamo_component_kvstats_active_blocks` | ✗ **NOT WORKING** | Dynamo's internal metric not populated by SGLang backend. |
| `dynamo_component_thompson_kve_` | `dynamo_component_thompson_kve_cached_tokens_total` | ✗ **NOT WORKING** | SGLang API doesn't return `cached_tokens` in response. |
| `dynamo_component_thompson_kve_` | `dynamo_component_thompson_kve_prompt_tokens_total` | ✗ **NOT WORKING** | Counter stays at 0 due to API limitation. |
| `dynamo_component_thompson_kve_` | `dynamo_component_thompson_kve_*_blocks_total` | ✗ **NOT WORKING** | Block-level KVE metrics not populated. |

### Architecture-Specific Metrics (Always Zero for Llama)

| Prefix | Full Metric Name | Status | Reason |
|--------|------------------|--------|--------|
| `sglang:` | `sglang:swa_token_usage` | N/A | Sliding Window Attention - not used by Llama architecture. |
| `sglang:` | `sglang:mamba_usage` | N/A | Mamba architecture metric - not applicable to Llama. |
| `sglang:` | `sglang:kv_transfer_*` | N/A | KV transfer metrics only used in disaggregated prefill/decode mode. |
| `sglang:` | `sglang:pending_prealloc_token_usage` | N/A | Preallocation metric - typically 0 in standard operation. |

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

The pre-configured dashboard "Dynamo LLM Overview" includes:

### Backend Selector

The dashboard includes a **Backend** dropdown variable at the top. Select:
- **sglang** - For SGLang workers (metrics prefixed with `sglang:`)
- **vllm** - For vLLM workers (metrics prefixed with `vllm:`)

All backend-specific panels automatically update based on your selection.

### Dashboard Panels

1. **Inflight Requests** - Current load across all components
2. **Requests/min** - Throughput
3. **Time to First Token (P95)** - Latency to start generating
4. **KVES Proxy (Cache Hit Rate %)** - KV Efficiency Score proxy using prefix cache hit rate
5. **TTFT Over Time** - P50/P95/P99 latency trends
6. **ITL Over Time** - Inter-token latency trends
7. **Token Throughput** - Tokens generated per second
8. **KV Cache Usage** - Memory usage % and prefix cache hit rate % over time
9. **KV Cache Tokens & Throughput** - Absolute token count and generation throughput
10. **KV Cache Details (Per-Worker)** - Detailed per-worker metrics including:
    - KVES: Prefix hit rate (%) - `avg_over_time(${backend}:cache_hit_rate[1m]) * 100`
    - KV Usage (%) - `avg_over_time(${backend}:token_usage[1m]) * 100`
    - KV Tokens Used - `last_over_time(${backend}:num_used_tokens[1m])`
    - KV Capacity (blocks) - `last_over_time(dynamo_component_kvstats_total_blocks[1m])`
    - Frontend Block Size - `last_over_time(dynamo_frontend_model_kv_cache_block_size[5m])`
11. **KVES Proxy by Worker** - Color-coded efficiency score per worker (0-1 scale)
12. **KV Cache Memory Usage % by Worker** - Per-worker memory utilization

### Thompson Sampling Panels (Included)

The dashboard includes these Thompson Sampling and worker monitoring panels:

- **Routing Decisions/sec** - `rate(dynamo_component_thompson_routing_decisions_total[5m])`
- **Worker Queue Depth** - `${backend}:num_queue_reqs`
- **Worker Activity** - `${backend}:num_running_reqs`

> **Note on KV Cache Metrics**: The dashboard uses backend-native metrics (`${backend}:token_usage`,
> `${backend}:cache_hit_rate`, `${backend}:num_used_tokens`) which are reliably populated by both
> SGLang and vLLM. The Dynamo-specific `dynamo_component_kvstats_*` metrics may not be populated
> depending on your backend configuration. See the "KV Cache Metrics Status" section above for details.

## Files

```
monitoring/
├── docker-compose.yml              # Prometheus + Grafana services
├── prometheus.yml                  # Prometheus scrape configuration
├── README.md                       # This file
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

### Start Monitoring

```bash
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
bash start_dynamo_optimized_thompson_hints.sh
```

## Remote Access via SSH Port Forwarding

If the monitoring stack is running on a remote server, use SSH port forwarding to access Grafana and Prometheus locally.

### General Syntax

```bash
ssh -L <local_port>:localhost:<remote_port> <username>@<remote_host>
```

### Access Grafana (Port 3000)

```bash
ssh -L 3000:localhost:3000 <username>@<remote_host>
```

Then open http://localhost:3000 in your browser.

### Access Prometheus (Port 9090)

```bash
ssh -L 9090:localhost:9090 <username>@<remote_host>
```

Then open http://localhost:9090 in your browser.

### Forward Multiple Ports

To access both Grafana and Prometheus simultaneously:

```bash
ssh -L 3000:localhost:3000 -L 9090:localhost:9090 <username>@<remote_host>
```

### Background SSH Tunnel

To run the tunnel in the background:

```bash
ssh -f -N -L 3000:localhost:3000 -L 9090:localhost:9090 <username>@<remote_host>
```

- `-f`: Run in background after authentication
- `-N`: Don't execute remote commands (tunnel only)

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

# All worker metrics (Dynamo + SGLang)
curl -s http://localhost:8081/metrics

# All router metrics
curl -s http://localhost:8082/metrics

# All processor metrics (Thompson Sampling)
curl -s http://localhost:8083/metrics

# Filter specific metrics
curl -s http://localhost:8000/metrics | grep time_to_first_token
curl -s http://localhost:8081/metrics | grep kvstats
curl -s http://localhost:8081/metrics | grep "sglang:"
curl -s http://localhost:8083/metrics | grep thompson
```

## Troubleshooting

### Prometheus can't scrape targets

Check if Dynamo is running:
```bash
curl http://localhost:8000/health
curl http://localhost:8081/metrics
```

### Grafana shows "No data"

1. Verify Prometheus is scraping: http://localhost:9090/targets
2. Check if metrics exist: http://localhost:9090/graph (query a metric name)
3. Ensure time range is correct in Grafana

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

## Alternative: File-Based Collection

If you don't want to run Prometheus/Grafana, use the collection script:

```bash
cd /localhome/local-bbednarski/NeMo-Agent-Toolkit/external/dynamo
./collect_metrics.sh ./metrics_output 30  # Collect every 30s
```

This creates timestamped `.prom` files that can be analyzed later or imported into Prometheus.

## Complete Metrics Reference

### Summary by Component

| Component | Port | Metric Count | Key Prefixes |
|-----------|------|--------------|--------------|
| Frontend | 8000 | ~22 | `dynamo_frontend_*` |
| Worker | 8081 | ~50 | `dynamo_component_kvstats_*`, `sglang:*` |
| Router | 8082 | ~20 | `dynamo_component_*` (labeled `router`) |
| Processor | 8083 | ~35 | `dynamo_component_thompson_*` |

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
<summary><b>Worker (port 8081) - 50 metrics</b></summary>

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
<summary><b>Router (port 8082) - 20 metrics</b></summary>

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
<summary><b>Processor (port 8083) - 35 metrics</b></summary>

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

