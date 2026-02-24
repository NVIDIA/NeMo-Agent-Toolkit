<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Optimized Thompson Sampling Router Architecture

## Overview

This architecture uses the **default Dynamo frontend** with custom **Processor** and **Router** components to implement Thompson Sampling-based intelligent worker selection with KV cache locality awareness.

### Processor-as-Backend Pattern

**Key insight**: The default Dynamo frontend has its own built-in router (`DYN_ROUTER_MODE`) and routes directly to `dynamo.backend.generate`. To intercept requests and apply custom Thompson Sampling routing:

1. **Processor registers as `dynamo.backend.generate`** - The frontend discovers our processor as the "backend"
2. **SGLang Worker registers as `dynamo.worker.generate`** - Our processor forwards to actual workers after routing
3. **The built-in frontend router becomes irrelevant** - The frontend routes to `dynamo.backend.generate` which is our processor

```text
Frontend (built-in router: round-robin)
    → routes to dynamo.backend.generate
    → OUR PROCESSOR (intercepts!)
        → queries Thompson Sampling router
        → forwards to dynamo.worker.generate (actual SGLang workers)
```

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                CLIENT                                            │
│                                                                                  │
│  POST /v1/chat/completions                                                      │
│  {                                                                               │
│    "model": "llama-3.3-70b",                                                    │
│    "messages": [...],                                                           │
│    "nvext": {                                                                   │
│      "annotations": [                                                           │
│        "prefix_id:my-session-001",                                              │
│        "total_requests:10",                                                     │
│        "osl:MEDIUM",                                                            │
│        "iat:LOW"                                                                │
│      ]                                                                          │
│    }                                                                            │
│  }                                                                               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         DEFAULT DYNAMO FRONTEND                                  │
│                           (python -m dynamo.frontend)                           │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ OpenAI HTTP Server (port 8000)                                          │   │
│  │  • /v1/chat/completions                                                 │   │
│  │  • /v1/models                                                           │   │
│  │  • /health                                                              │   │
│  │  • /metrics (Prometheus)                                                │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ Preprocessor                                                            │   │
│  │  • Tokenization (chat template applied)                                 │   │
│  │  • NVExt parsing → PreprocessedRequest                                  │   │
│  │  • Annotations preserved: prefix_id, total_requests, osl, iat           │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                         │
│                                        │ PreprocessedRequest                     │
│                                        │ (tokens + annotations + extra_args)     │
└────────────────────────────────────────┼────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         CUSTOM PROCESSOR                                         │
│              (registers as: dynamo.backend.generate)                             │
│              (intercepts frontend requests!)                                     │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ 1. Receive PreprocessedRequest from frontend                            │   │
│  │    • Extract annotations: prefix_id, total_requests, osl, iat           │   │
│  │    • Compute reuse_budget = total_requests - processed_for_prefix       │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ 2. Query Router (find_worker endpoint)                                  │   │
│  │    RouterRequest {                                                      │   │
│  │      tokens: [...],                                                     │   │
│  │      prefix_id: "my-session-001",                                       │   │
│  │      reuse_budget: 9,                                                   │   │
│  │      expected_osl: "MEDIUM",                                            │   │
│  │      interarrival: "LOW"                                                │   │
│  │    }                                                                    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ 3. Route to Selected Backend Worker                                     │   │
│  │    • Use worker_id from router to direct request                        │   │
│  │    • Stream response tokens back to frontend                            │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ 4. Send Feedback to Router                                              │   │
│  │    RouterFeedbackRequest {                                              │   │
│  │      decision_id: "abc123",                                             │   │
│  │      latency_ms: 245.5,                                                 │   │
│  │      success: true,                                                     │   │
│  │      tokens_in: 128,                                                    │   │
│  │      tokens_out: 64                                                     │   │
│  │    }                                                                    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  Prometheus Metrics (port 8081):                                                │
│    • thompson_processor_requests_total                                          │
│    • thompson_processor_request_latency_seconds                                 │
│    • thompson_processor_tokens_processed_total                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         CUSTOM ROUTER                                            │
│                    (dynamo/router component)                                     │
│                                                                                  │
│  Endpoints:                                                                      │
│    • find_worker: Select optimal worker for request                             │
│    • feedback: Receive latency feedback to update bandits                       │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ Thompson Sampling Algorithm                                             │   │
│  │                                                                         │   │
│  │  Score(worker) = LinTS(features) + Beta_TS(worker)                      │
│  │                + Affinity(prefix_sticky)                                │   │
│  │                - SwitchCost(if switching)                               │   │
│  │                × LoadModifier(queue, GPU, outstanding)                  │   │
│  │                                                                         │   │
│  │  Features (9-dim):                                                      │   │
│  │    [1, inv_load, kv_overlap, affinity, outstanding,                     │   │
│  │     decode_cost, prefill_cost, iat_factor, reuse_budget]                │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ KV Cache Indexer                                                        │   │
│  │  • Tracks KV cache blocks per worker                                    │   │
│  │  • Computes overlap scores for routing decisions                        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ Bandit State                                                            │   │
│  │  • Beta bandits: (α, β) per worker                                      │   │
│  │  • LinTS: A matrix, b vector per worker                                 │   │
│  │  • Pending decisions awaiting feedback                                  │   │
│  │  • Latency EMA baselines (global, per-worker, per-bucket)               │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  Prometheus Metrics (port 8081):                                                │
│    • thompson_router_decisions_total{worker_id}                                 │
│    • thompson_router_kv_overlap{worker_id}                                      │
│    • thompson_router_feedback_latency_seconds                                   │
│    • thompson_router_reward{worker_id}                                          │
│    • thompson_router_pending_decisions                                          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         BACKEND WORKER (Unified Mode)                            │
│                    (python -m dynamo.sglang)                                     │
│              (registers as: dynamo.worker.generate)                              │
│              (NOT backend.generate - that's our processor!)                      │
│                                                                                  │
│  Default Configuration (start_dynamo_optimized_thompson_hints.sh):              │
│                                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │   Unified Worker                                                          │  │
│  │   GPUs: 0,1,2,3 (DYNAMO_GPU_DEVICES)                                      │  │
│  │   TP: 4 (DYNAMO_TP_SIZE)                                                  │  │
│  │   Endpoint: dynamo.worker.generate (--endpoint flag)                      │  │
│  │                                                                           │  │
│  │   • KV Cache (shared across TP ranks)                                     │  │
│  │   • SGLang Engine                                                         │  │
│  │   • Prometheus Metrics (port 8081)                                        │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
│  Environment Variables for GPU Configuration:                                   │
│    DYNAMO_GPU_DEVICES="0,1,2,3"    # Which GPUs to use (default: 0,1,2,3)      │
│    DYNAMO_TP_SIZE=4                # Tensor parallelism degree (default: 4)    │
│                                                                                  │
│  Metrics exposed:                                                               │
│    • sglang:* metrics on port 8081                                              │
│    • dynamo_component_* metrics                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Scaling to Multiple Workers (8-GPU Example)

For systems with more GPUs, you can run multiple workers. The current startup script
runs a **single unified worker** by default. To scale to multiple workers:

### Option A: Two Workers with TP=4 (8 GPUs total)
```bash
# Worker 1: GPUs 0-3
export DYNAMO_GPU_DEVICES="0,1,2,3"
export DYNAMO_TP_SIZE=4
# (start first worker)

# Worker 2: GPUs 4-7
export DYNAMO_GPU_DEVICES="4,5,6,7"
export DYNAMO_TP_SIZE=4
# (start second worker)
```

### Option B: One Worker with TP=8 (8 GPUs, single worker)
```bash
export DYNAMO_GPU_DEVICES="0,1,2,3,4,5,6,7"
export DYNAMO_TP_SIZE=8
```

> **Note**: The Thompson Sampling router benefits most from multiple workers,
> as it can learn optimal routing between them. With a single worker, the router
> still tracks KV cache overlap but cannot make routing decisions between workers.

## Key Differences from Generalized Architecture

| Aspect | Generalized | Optimized |
|--------|-------------|-----------|
| Frontend | Custom `frontend.py` with HTTP headers | Default `dynamo.frontend` with `nvext` |
| Hint Passing | HTTP headers (`x-prefix-*`) | `nvext.annotations` in request body |
| Tokenization | Custom (in frontend) | Handled by Dynamo pre-processor |
| Metrics | CSV files | Prometheus (`/metrics` endpoint) |
| Model Mapping | Custom `FRONTEND_MODEL_MAPPING` | Dynamo `--model-name`/`--model-path` |
| **Processor Registration** | `dynamo.processor.process` | **`dynamo.backend.generate`** (intercepts frontend) |
| **Worker Registration** | `dynamo.backend.generate` | **`dynamo.worker.generate`** (processor forwards to) |

### Why "Processor-as-Backend"?

The default Dynamo frontend has a built-in router (`DYN_ROUTER_MODE=round-robin|random|kv`) that routes directly to `dynamo.backend.generate`. To inject our custom Thompson Sampling routing:

1. **Processor claims `backend.generate`** - Frontend thinks it's talking to the backend
2. **Processor queries custom router** - Thompson Sampling selects best worker
3. **Processor forwards to `worker.generate`** - Actual SGLang workers
4. **The built-in frontend router is irrelevant** - We've intercepted the request pipeline

## `nvext` Annotations

The client passes routing hints via the `nvext.annotations` field in the request:

```json
{
  "model": "llama-3.3-70b",
  "messages": [{"role": "user", "content": "Hello!"}],
  "nvext": {
    "annotations": [
      "prefix_id:session-12345",
      "total_requests:10",
      "osl:MEDIUM",
      "iat:LOW"
    ]
  }
}
```

### Annotation Keys

| Key | Type | Description | Values |
|-----|------|-------------|--------|
| `prefix_id` | `string` | Unique identifier for request prefix and session | Any string |
| `total_requests` | `int` | Total expected requests for this prefix | Positive integer |
| `osl` | `enum` | Output Sequence Length expectation | `LOW`, `MEDIUM`, `HIGH` |
| `iat` | `enum` | Inter-Arrival Time (request frequency) | `LOW`, `MEDIUM`, `HIGH` |

## Quick Start

```bash
# Required: Set path to your model
export DYNAMO_MODEL_DIR="/path/to/Llama-3.3-70B-Instruct"

# Optional: Configure GPU devices (default: 0,1,2,3)
export DYNAMO_GPU_DEVICES="0,1,2,3"
export DYNAMO_TP_SIZE=4

# Optional: Set model name (default: llama-3.3-70b)
export DYNAMO_MODEL_NAME="llama-3.3-70b"

# Start the system
bash start_dynamo_optimized_thompson_hints.sh
```

## Component Startup Order

1. **etcd** - Service discovery and metadata
2. **NATS** - Message queue for KV events (if using KV-aware router mode)
3. **Backend Worker** - SGLang GPU worker → registers at `dynamo.worker.generate`
4. **Router** - Thompson Sampling router → registers at `dynamo.router.{find_worker,feedback}`
5. **Processor** - Request orchestrator → **registers at `dynamo.backend.generate`** (intercepts frontend!)
6. **Frontend** - HTTP API server → routes to `dynamo.backend.generate` (our processor)

> **Important**: The Processor must register as `backend.generate` before the Frontend starts,
> otherwise the Frontend might discover the SGLang worker directly (if it registered as `backend.generate`).

## Prometheus Metrics

All components expose metrics on port 8081 by default (`DYN_SYSTEM_PORT`):

### Router Metrics
```text
thompson_router_decisions_total{worker_id="0"} 1234
thompson_router_kv_overlap{worker_id="0"} 0.75
thompson_router_feedback_latency_seconds_bucket{le="0.1"} 100
thompson_router_reward{worker_id="0"} 0.65
thompson_router_pending_decisions 5
thompson_router_timeout_penalties_total 2
```

### Processor Metrics
```text
thompson_processor_requests_total 5000
thompson_processor_request_latency_seconds_bucket{le="1.0"} 4500
thompson_processor_tokens_in_total 128000
thompson_processor_tokens_out_total 64000
thompson_processor_routing_decisions_total{worker_id="0"} 1234
```

## Environment Variables

### GPU and Worker Configuration

These variables control how the backend worker uses GPUs. **Modify these to scale your deployment.**

| Variable | Default | Description |
|----------|---------|-------------|
| `DYNAMO_GPU_DEVICES` | `0,1,2,3` | Comma-separated list of GPU device IDs to use |
| `DYNAMO_TP_SIZE` | `4` | Tensor parallelism degree (must match number of GPUs) |
| `DYNAMO_MODEL_DIR` | (required) | Path to the model directory on the host |
| `DYNAMO_MODEL_NAME` | `llama-3.3-70b` | Model name exposed to clients |
| `DYNAMO_SHM_SIZE` | `16g` | Shared memory size for the container |
| `DYNAMO_WORKER_INIT_TIMEOUT_S` | `600` | Timeout (seconds) for worker initialization |

### Example GPU Configurations

```bash
# Default: Single worker using GPUs 0-3 with TP=4
export DYNAMO_GPU_DEVICES="0,1,2,3"
export DYNAMO_TP_SIZE=4

# 8-GPU system: Single worker using all 8 GPUs with TP=8
export DYNAMO_GPU_DEVICES="0,1,2,3,4,5,6,7"
export DYNAMO_TP_SIZE=8

# 8-GPU system: Use only GPUs 4-7 with TP=4
export DYNAMO_GPU_DEVICES="4,5,6,7"
export DYNAMO_TP_SIZE=4

# 2-GPU system: Use GPUs 0-1 with TP=2
export DYNAMO_GPU_DEVICES="0,1"
export DYNAMO_TP_SIZE=2
```

### Network and Metrics Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DYNAMO_HTTP_PORT` | `8000` | Frontend HTTP API port |
| `DYNAMO_METRICS_PORT` | `8081` | Prometheus metrics port |
| `DYN_HTTP_PORT` | `8000` | Dynamo frontend HTTP port (same as above) |
| `DYN_SYSTEM_PORT` | `8081` | Dynamo system and metrics port |
| `DYNAMO_ROUTER_WAIT_FOR_WORKERS_TIMEOUT_S` | `600` | Worker discovery timeout |

### Backend-Specific Configuration (REQUIRED)

| Variable | Values | Description |
|----------|--------|-------------|
| `DYNAMO_WORKER_COMPONENT` | `worker` or `backend` | **REQUIRED.** Component name where workers register. SGLang uses `worker` (via `--endpoint workers.worker.generate`). vLLM uses `backend` (hardcoded in `dynamo.vllm`). |

> **Important**: `DYNAMO_WORKER_COMPONENT` must be set for the router and processor to find
> the backend workers. Without this variable, startup will fail with an error.
>
> **Note on `DYN_ROUTER_MODE`**: The startup script passes `--router-mode round-robin` to the
> default frontend, but this is **irrelevant** in our architecture. The built-in router of the
> frontend routes to `dynamo.backend.generate`, which is our Processor (not a real backend).
> Our Processor intercepts the request and uses our custom Thompson Sampling router instead.

## Sample Client Request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.3-70b",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 100,
    "stream": true,
    "nvext": {
      "annotations": [
        "prefix_id:math-session-001",
        "total_requests:5",
        "osl:LOW",
        "iat:HIGH"
      ]
    }
  }'
```

## Request Flow (Detailed)

1. **Client → Frontend**: HTTP POST with `nvext` annotations
2. **Frontend (Pre-processor)**: Tokenize messages, creates `PreprocessedRequest` with annotations
3. **Frontend (Built-in Router)**: Routes to `dynamo.backend.generate` (round-robin, but only one "backend" - our processor!)
4. **Processor (as backend.generate)**: Receives request, extracts hints from annotations
5. **Processor → Router**: Queries Thompson Sampling router for worker selection
6. **Router**: Computes Thompson Sampling scores, returns `worker_id`
7. **Processor → Worker**: Sends request to `dynamo.worker.generate` via `engine_client.direct(worker_id)`
7. **Backend → Processor**: Streams response tokens
8. **Processor → Router**: Sends latency feedback for bandit update
9. **Processor → Frontend**: Streams response
10. **Frontend → Client**: SSE stream or JSON response

## Files

- `processor.py` - Custom processor with `nvext` annotation extraction
- `router.py` - Thompson Sampling router with Prometheus metrics
- `ARCHITECTURE.md` - This document
