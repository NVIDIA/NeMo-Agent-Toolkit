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

This directory contains the optimized implementation of the Thompson Sampling router for Dynamo, using the "Processor-as-Backend" pattern with **Dynamic Discovery** to intercept requests from the default Dynamo frontend.

## Architecture Overview (Dynamic Discovery Mode)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Client Request (with nvext.annotations)                                │
│       ↓                                                                 │
│  Default Dynamo Frontend (port 8000)                                    │
│       ↓ tokenization + nvext parsing                                    │
│       ↓ discovers backends via ETCD ModelWatcher                        │
│       ↓ finds Processor's model card!                                   │
│                                                                         │
│  Custom Processor (dynamo.backend.generate-{instance_id})               │
│       ↓ extracts: prefix_id, total_requests, osl, iat                   │
│       ↓ queries Thompson Sampling router                                │
│                                                                         │
│  Custom Router (dynamo.router.find_worker)                              │
│       ↓ KV overlap + workload-aware selection                           │
│       ↓ returns worker_id                                               │
│                                                                         │
│  Processor forwards to dynamo.worker.generate (with worker_id)          │
│       ↓                                                                 │
│  SGLang Worker (actual inference)                                       │
│       ↓                                                                 │
│  Response + Feedback to Router                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Components

| Component | File | Endpoint | Purpose |
|-----------|------|----------|---------|
| Processor | `processor.py` | `dynamo.backend.generate` + etcd model card | Intercepts frontend requests, extracts hints, coordinates routing |
| Router | `router.py` | `dynamo.router.find_worker` | Thompson Sampling + KV overlap worker selection |
| config | `config.yaml` | - | Router configuration parameters |

## Dynamic Discovery Pattern (Forward-Compatible)

Instead of using the deprecated `--static-endpoint` flag on the frontend, this processor uses **dynamic discovery** via etcd:

1. **Processor** registers as `dynamo.backend.generate` (dynamic mode with instance ID)
2. **Processor** calls `register_llm()` to advertise a model card in etcd
3. **Frontend ModelWatcher** discovers the processor's model card
4. **Frontend** routes requests to the discovered processor endpoint
5. **SGLang Worker** registers as `dynamo.worker.generate` (also dynamic)

### Why Dynamic Discovery?

The `--static-endpoint` flag is **deprecated** and will be removed in future Dynamo versions. Dynamic discovery provides:

- Forward compatibility with future Dynamo releases
- Support for multiple processor instances (load balancing)
- Standard Dynamo discovery patterns
- Dynamic scaling capabilities

## Processor Registration

The processor uses `register_llm()` to advertise itself in etcd:

```python
@dynamo_worker(static=False)  # Dynamic mode for ETCD discovery
async def worker(runtime: DistributedRuntime):
    component = runtime.namespace("dynamo").component("backend")
    # NOTE: create_service() was removed in Dynamo 0.8.x - endpoint creation handles registration
    endpoint = component.endpoint("generate")
    
    # Register model card so frontend can discover us
    await register_llm(
        model_input=ModelInput.Tokens,
        model_type=ModelType.Chat | ModelType.Completions,
        endpoint=endpoint,
        model_path=args.model_path,
        model_name=args.model_name,
    )
    
    handler = ProcessorRequestHandler(runtime, ...)
    await endpoint.serve_endpoint(handler.generate)
```

### Required Arguments

The processor now requires:
- `--model-path`: Path to the model directory (for tokenizer and model card)
- `--model-name`: Served model name (must match the model expected by the frontend)

## Usage

### Starting the System

```bash
# Set required environment variable
export DYNAMO_MODEL_DIR="/path/to/Llama-3.3-70B-Instruct"

# Start all components
bash ../start_dynamo_optimized_thompson_hints_sglang.sh

# or

bash ../start_dynamo_optimized_thompson_hints_vllm.sh
```

### Making Requests

```bash
# Basic request (no routing hints)
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llama-3.3-70b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'

# Request with nvext annotations (routing hints)
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llama-3.3-70b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50,
    "nvext": {
      "annotations": [
        "prefix_id:my-session-001",
        "total_requests:10",
        "osl:MEDIUM",
        "iat:LOW"
      ]
    }
  }'
```

### Routing Hint Annotations

| Annotation | Format | Description |
|------------|--------|-------------|
| `prefix_id` | `prefix_id:<string>` | Unique identifier for prefix reuse across requests |
| `total_requests` | `total_requests:<int>` | Expected total requests in this prefix group |
| `osl` | `osl:LOW\|MEDIUM\|HIGH` | Expected output sequence length |
| `iat` | `iat:LOW\|MEDIUM\|HIGH` | Inter-arrival time hint |

---

## Troubleshooting

### Verifying Processor Interception

To confirm that requests are flowing through the processor (not directly to workers), run:

```bash
docker logs dynamo-sglang-components 2>&1 | grep -E "(Processor|processor|Processing request|Routing decision|dynamo.backend|backend.generate|find_worker)" | tail -50
```

### Expected Output (Nominal Operation)

When the system is working correctly, you should see output similar to:

```
Step 3: Starting Custom Processor (Registers as backend.generate)...
Processor PID: 3735
Registered at: dynamo.backend.generate (intercepts frontend requests)

INFO processor._init_prometheus_metrics: Prometheus metrics initialized for processor
INFO processor.initialize: Router clients created, waiting for instances...
INFO dynamo_runtime::component::client: wait_for_instances: Found 1 instance(s) for endpoint
INFO processor.initialize: Router clients initialized successfully
INFO processor.initialize: Engine client created, waiting for worker instances...
INFO processor.initialize: Processor initialized successfully (routing to dynamo.worker.generate)

INFO processor.generate: Processing request: prefix=auto-3f0519ac1cc442d2... total=1 osl=MEDIUM iat=MEDIUM tokens=37
INFO processor.generate: Routing decision: worker=7587892168930944779 decision=bcc5180740ed44c6... reuse_budget=0

INFO processor.generate: Processing request: prefix=auto-2593032a6cf843ce... total=1 osl=MEDIUM iat=MEDIUM tokens=37
INFO processor.generate: Routing decision: worker=7587892168930944779 decision=ba4440fd3a144822... reuse_budget=0
```

### Key Indicators of Success

| Log Message | Meaning |
|-------------|---------|
| `Registering model card: model_name=...` | Processor registering with etcd |
| `Model card registered successfully` | Frontend can now discover the processor |
| `Router clients initialized successfully` | Connected to Thompson Sampling router |
| `Processor initialized successfully` | Ready to process requests |
| `Processing request: prefix=... tokens=N` | Request received and being processed |
| `Routing decision: worker=... decision=...` | Router selected a worker |

### Common Issues

#### 1. Frontend Not Finding Processor

**Symptom:** Requests fail or go directly to workers, bypassing processor.

**Cause:** Model card not registered or model name mismatch.

**Verification:**
```bash
# Check if processor registered its model card
docker logs dynamo-sglang-components 2>&1 | grep -i "model card"

# Check ETCD for registered models
curl -s http://localhost:2379/v3/kv/range -X POST \
  -H "Content-Type: application/json" \
  -d '{"key":"ZHluYW1v","range_end":"ZHluYW1w"}' | jq .
```

**Solution:**
1. Ensure `--model-name` matches between processor and frontend
2. Ensure `--model-path` points to a valid model directory
3. Processor must start BEFORE frontend

#### 2. "missing field `token_ids`" Error

**Cause:** Processor couldn't reach workers.

**Solution:** Ensure workers are registered and running:
```bash
docker logs dynamo-sglang-components 2>&1 | grep "worker.generate"
```

#### 3. Requests Bypassing Processor

**Symptom:** No "Processing request" logs, but responses still work.

**Cause:** Frontend is routing directly to workers instead of through the processor.

**Verification:**
```bash
# Check if processor is receiving requests
docker logs dynamo-sglang-components 2>&1 | grep "Processing request"
```

**Solution:**
1. Ensure processor's `--model-name` matches the frontend `--model-name` parameter exactly
2. Processor must register BEFORE frontend starts
3. Check that processor's model card is in etcd

#### 4. Router Not Found

**Symptom:** `Router stream ended without worker_id; falling back to engine load balancing`

**Cause:** Router not started or not registered.

**Solution:** Check router logs:
```bash
docker logs dynamo-sglang-components 2>&1 | grep -i router
```

---

## Prometheus Metrics

| Metric | Description |
|--------|-------------|
| `thompson_processor_requests_total` | Total requests processed |
| `thompson_processor_request_latency_seconds` | Request latency histogram |
| `thompson_processor_tokens_in_total` | Total input tokens |
| `thompson_processor_tokens_out_total` | Total output tokens |
| `thompson_processor_routing_decisions_total` | Routing decisions by worker |
| `thompson_processor_router_errors_total` | Router communication errors |
| `thompson_processor_engine_errors_total` | Backend engine errors |
| `thompson_processor_active_requests` | Currently active requests |

Access metrics:
```bash
curl http://localhost:8081/metrics | grep thompson_processor
```

---

## Configuration

See `config.yaml` for router configuration options and `PARAMETERS.md` for detailed parameter documentation.

