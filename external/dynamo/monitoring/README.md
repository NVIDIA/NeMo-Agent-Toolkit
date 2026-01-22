# Dynamo Monitoring Stack

This directory contains a Prometheus + Grafana monitoring setup for the Dynamo LLM inference stack with Thompson Sampling router.

## Quick Start

```bash
# Start the monitoring stack
cd monitoring
docker compose up -d

# Access the dashboards
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

## Prerequisites

- Docker and Docker Compose
- Dynamo stack running (see `../start_dynamo_optimized_thompson_hints.sh`)

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
| Worker | 8081 | `http://localhost:8081/metrics` | Internal metrics (KV cache, NATS stats) |
| Router | 8082 | `http://localhost:8082/metrics` | Thompson Sampling routing metrics |
| Processor | 8083 | `http://localhost:8083/metrics` | Thompson Sampling KVE metrics |

## Key Metrics

### Frontend Metrics (`:8000/metrics`)

User-facing HTTP API metrics for latency, throughput, and token statistics.

| Metric | Type | Description |
|--------|------|-------------|
| `dynamo_frontend_requests_total` | Counter | Total requests processed |
| `dynamo_frontend_inflight_requests` | Gauge | Currently processing requests |
| `dynamo_frontend_queued_requests` | Gauge | Requests waiting in queue |
| `dynamo_frontend_disconnected_clients` | Counter | Client disconnections |
| `dynamo_frontend_time_to_first_token_seconds` | Histogram | Time until first token generated |
| `dynamo_frontend_inter_token_latency_seconds` | Histogram | Time between consecutive tokens |
| `dynamo_frontend_request_duration_seconds` | Histogram | Total request duration |
| `dynamo_frontend_input_sequence_tokens` | Histogram | Input prompt length distribution |
| `dynamo_frontend_output_sequence_tokens` | Histogram | Output length distribution |
| `dynamo_frontend_output_tokens_total` | Counter | Total output tokens generated |
| `dynamo_frontend_model_context_length` | Gauge | Model context window size |
| `dynamo_frontend_model_kv_cache_block_size` | Gauge | KV cache block size |

### Worker Metrics (`:8081/metrics`)

SGLang backend worker metrics including KV cache, scheduling, and internal statistics.

#### Dynamo Component Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `dynamo_component_kvstats_gpu_cache_usage_percent` | Gauge | KV cache memory utilization (0-100) |
| `dynamo_component_kvstats_gpu_prefix_cache_hit_rate` | Gauge | Prefix cache hit rate (0-1) |
| `dynamo_component_kvstats_active_blocks` | Gauge | Active KV cache blocks |
| `dynamo_component_kvstats_total_blocks` | Gauge | Total KV cache blocks |
| `dynamo_component_request_duration_seconds` | Histogram | Backend request processing time |
| `dynamo_component_requests_total` | Counter | Total requests to worker |
| `dynamo_component_inflight_requests` | Gauge | Requests currently in worker |
| `dynamo_component_uptime_seconds` | Gauge | Worker uptime |

#### SGLang Native Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `sglang:cache_hit_rate` | Gauge | Prefix cache hit rate |
| `sglang:token_usage` | Gauge | Current token usage |
| `sglang:num_running_reqs` | Gauge | Currently running requests |
| `sglang:num_queue_reqs` | Gauge | Queued requests |
| `sglang:num_used_tokens` | Gauge | Tokens currently in use |
| `sglang:gen_throughput` | Gauge | Generation throughput |
| `sglang:utilization` | Gauge | GPU utilization |
| `sglang:queue_time_seconds` | Histogram | Time spent in queue |
| `sglang:per_stage_req_latency_seconds` | Histogram | Per-stage request latency |
| `sglang:kv_transfer_latency_ms` | Gauge | KV transfer latency |
| `sglang:kv_transfer_speed_gb_s` | Gauge | KV transfer speed |
| `sglang:engine_startup_time` | Gauge | Engine startup duration |
| `sglang:engine_load_weights_time` | Gauge | Model weight loading time |

### Router Metrics (`:8082/metrics`)

Dynamo component metrics for the Thompson Sampling router (uses standard `dynamo_component_*` prefix).

| Metric | Type | Description |
|--------|------|-------------|
| `dynamo_component_requests_total` | Counter | Total routing requests (labeled by endpoint) |
| `dynamo_component_request_duration_seconds` | Histogram | Routing decision latency |
| `dynamo_component_request_bytes_total` | Counter | Request payload bytes |
| `dynamo_component_response_bytes_total` | Counter | Response payload bytes |
| `dynamo_component_inflight_requests` | Gauge | In-flight routing requests |
| `dynamo_component_uptime_seconds` | Gauge | Router uptime |
| `dynamo_component_nats_service_requests_total` | Gauge | NATS service requests |
| `dynamo_component_nats_service_processing_ms_avg` | Gauge | Average NATS processing time |
| `dynamo_component_nats_client_connection_state` | Gauge | NATS connection state (0=disconnected, 1=connected) |

**Router Endpoints** (use `dynamo_endpoint` label to filter):
- `find_worker` - Worker selection requests
- `feedback` - Feedback from completed requests

### Thompson Sampling Processor Metrics (`:8083/metrics`)

Custom Thompson Sampling KV Efficiency (KVE) metrics from the processor component.

| Metric | Type | Description |
|--------|------|-------------|
| `dynamo_component_thompson_requests_total` | Counter | Total requests processed |
| `dynamo_component_thompson_request_latency_seconds` | Histogram | End-to-end request latency |
| `dynamo_component_thompson_tokens_in_total` | Counter | Total input tokens |
| `dynamo_component_thompson_tokens_out_total` | Counter | Total output tokens |
| `dynamo_component_thompson_routing_decisions_total` | Counter | Routing decisions made |
| `dynamo_component_thompson_active_requests` | Gauge | Currently processing requests |
| `dynamo_component_thompson_router_errors_total` | Counter | Router communication errors |
| `dynamo_component_thompson_engine_errors_total` | Counter | Engine/worker errors |
| `dynamo_component_thompson_kve_prompt_tokens_total` | Counter | Total prompt tokens (KVE denominator) |
| `dynamo_component_thompson_kve_cached_tokens_total` | Counter | Cached tokens hit (KVE numerator) |
| `dynamo_component_thompson_kve_device_blocks_total` | Counter | KV blocks from GPU memory |
| `dynamo_component_thompson_kve_host_blocks_total` | Counter | KV blocks from CPU memory |
| `dynamo_component_thompson_kve_disk_blocks_total` | Counter | KV blocks from disk |

**KV Efficiency (KVE) Calculation:**
```promql
# KV Cache Efficiency percentage (using SGLang native metric - RECOMMENDED)
sglang:cache_hit_rate * 100

# Alternative: Using processor counters (may show 0 if SGLang doesn't return cached_tokens in API)
# rate(dynamo_component_thompson_kve_cached_tokens_total[5m]) / rate(dynamo_component_thompson_kve_prompt_tokens_total[5m]) * 100
```

> **Why use SGLang's native metric?** SGLang computes cache hit rate internally but doesn't include
> `cached_tokens` in its API responses. The processor's `thompson_kve_*` counters will show 0
> unless the underlying engine provides `usage.prompt_tokens_details.cached_tokens`.

## Grafana Dashboard

The pre-configured dashboard "Dynamo LLM Overview" includes:

1. **Inflight Requests** - Current load across all components
2. **Requests/min** - Throughput
3. **Time to First Token (P95)** - Latency to start generating
4. **KV Cache Usage %** - GPU memory utilization
5. **TTFT Over Time** - P50/P95/P99 latency trends
6. **ITL Over Time** - Inter-token latency trends
7. **Token Throughput** - Tokens generated per second
8. **KV Cache Stats** - Cache usage and hit rate over time

### Thompson Sampling Panels (Included)

The dashboard includes these Thompson Sampling and SGLang monitoring panels:

- **KV Efficiency / Cache Hit Rate** - `sglang:cache_hit_rate * 100` (SGLang native metric)
- **Routing Decisions/sec** - `rate(dynamo_component_thompson_routing_decisions_total[5m])`
- **SGLang Queue Depth** - `sglang:num_queue_reqs` + `sglang:num_running_reqs`
- **Worker Utilization** - `sglang:utilization` + `sglang:token_usage`

> **Note**: KV Efficiency uses SGLang's native `cache_hit_rate` metric rather than the processor's
> `thompson_kve_*` counters because SGLang doesn't include `cached_tokens` in its API responses.
> The native metric provides the same information: `(cached_tokens / prompt_tokens) * 100`.

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
docker compose down -v  # Removes volumes
docker compose up -d
```

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

