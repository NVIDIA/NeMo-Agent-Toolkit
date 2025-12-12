# Dynamo Backend Setup Guide

> ⚠️ **EXPERIMENTAL**: The NeMo Agent Toolkit and Dynamo integration is experimental and under active development. APIs, configurations, and features may change without notice.

This guide covers setting up, running, and configuring the NVIDIA Dynamo backend for the React Benchmark Agent evaluations.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Starting Dynamo](#starting-dynamo)
4. [Stopping Dynamo](#stopping-dynamo)
5. [Testing the Integration](#testing-the-integration)
6. [Monitoring](#monitoring)
7. [Dynamic Prefix Headers](#dynamic-prefix-headers)
8. [Configuration Reference](#configuration-reference)
9. [Troubleshooting](#troubleshooting)

---

## Overview

Dynamo is NVIDIA's high-performance LLM serving platform with KV cache optimization. This project provides three deployment modes:

| Mode | Script | Description | Best For |
|------|--------|-------------|----------|
| **Unified** | `start_dynamo_unified.sh` | Single worker, all operations | Development, testing |
| **Unified + Thompson** | `start_dynamo_unified_thompson_hints.sh` | Unified with predictive KV-aware router | Production, KV optimization |
| **Disaggregated** | `start_dynamo_disagg.sh` | Separate prefill/decode workers | High-throughput production |

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     DYNAMO BACKEND ARCHITECTURE                              │
└──────────────────────────────────────────────────────────────────────────────┘


                           CLIENT REQUEST
                    (NAT eval, curl, Python)
                                │
                                │  POST /v1/chat/completions
                                │  Headers:
                                │    x-prefix-id: react-bench-a1b2c3d4
                                │    x-prefix-total-requests: 10
                                │    x-prefix-osl: MEDIUM
                                │    x-prefix-iat: MEDIUM
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          DYNAMO FRONTEND                                     │
│                          Port 8099                                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                     HTTP API (OpenAI Compatible)                       │  │
│  │  ───────────────────────────────────────────────────────────────────── │  │
│  │  • /v1/chat/completions    - Chat completion endpoint                  │  │
│  │  • /v1/models              - List available models                     │  │
│  │  • /health                 - Health check                              │  │
│  │  • Extract x-prefix-* headers for router hints                         │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                         PROCESSOR                                      │  │
│  │  ───────────────────────────────────────────────────────────────────── │  │
│  │  • Tokenize messages → token_ids                                       │  │
│  │  • Extract prefix hints from headers                                   │  │
│  │  • Format engine request                                               │  │
│  │  • Track prefix state (outstanding requests)                           │  │
│  │  • CSV metrics logging                                                 │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                          ROUTER                                        │  │
│  │  ───────────────────────────────────────────────────────────────────── │  │
│  │                                                                        │  │
│  │  ┌──────────────────────┐  ┌──────────────────────────────────────┐    │  │
│  │  │   Worker Selection   │  │    Thompson Sampling (Optional)      │    │  │
│  │  │   ────────────────   │  │    ────────────────────────────────  │    │  │
│  │  │   1. KV cache overlap│  │    • LinTS for continuous params     │    │  │
│  │  │   2. Worker affinity │  │    • Beta bandits for discrete       │    │  │
│  │  │   3. Load balancing  │  │    • Explores vs exploits workers    │    │  │
│  │  │   4. OSL/IAT hints   │  │    • Learns optimal routing          │    │  │
│  │  └──────────────────────┘  └──────────────────────────────────────┘    │  │
│  │                                                                        │  │
│  │  Routing Decision Factors:                                             │  │
│  │  • overlap_score: KV cache reuse potential                             │  │
│  │  • prefill_cost: Estimated prefill compute                             │  │
│  │  • decode_cost: Based on OSL hint (LOW=1.0, MEDIUM=2.0, HIGH=3.0)      │  │
│  │  • iat_factor: Stickiness based on IAT (LOW=1.5, MEDIUM=1.0, HIGH=2.0) │  │
│  │  • load_modifier: Current worker queue depth                           │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
└────────────────────────────────────┼─────────────────────────────────────────┘
                                     │
                                     │  Route to selected worker
                                     │
        ┌────────────────────────────┴────────────────────────────────┐
        │                                                             │
        ▼                                                             ▼
┌─────────────────────────────┐                      ┌─────────────────────────────┐
│    UNIFIED WORKER           │         OR           │    DISAGGREGATED WORKERS    │
│    (GPUs 4,5,6,7, TP=4)     │                      │                             │
│                             │                      │  ┌────────────────────────┐ │
│  ┌───────────────────────┐  │                      │  │   PREFILL WORKER       │ │
│  │  SGLang Engine        │  │                      │  │   (GPUs 4,5, TP=2)     │ │
│  │  ─────────────────    │  │                      │  │   • Initial KV compute │ │
│  │  • Model: Llama-3.3-70B  │                      │  │   • Sends KV via NIXL  │ │
│  │  • KV Cache Management│  │                      │  └───────────┬────────────┘ │
│  │  • Token Generation   │  │                      │              │              │
│  │  • Streaming Support  │  │                      │              │ NIXL KV      │
│  └───────────────────────┘  │                      │              │ Transfer     │
│                             │                      │              ▼              │
│  All operations in one      │                      │  ┌────────────────────────┐ │
│  worker                     │                      │  │   DECODE WORKER        │ │
│                             │                      │  │   (GPUs 6,7, TP=2)     │ │
│                             │                      │  │   • Token generation   │ │
│                             │                      │  │   • Streaming output   │ │
│                             │                      │  └────────────────────────┘ │
└─────────────────────────────┘                      └─────────────────────────────┘
        │                                                             │
        └─────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │  STREAMING RESPONSE  │
                           │  ────────────────────│
                           │  {"choices": [...],  │
                           │   "content": "..."}  │
                           └──────────────────────┘


┌──────────────────────────────────────────────────────────────────────────────┐
│                        INFRASTRUCTURE SERVICES                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────┐         ┌────────────────────────┐               │
│  │   ETCD                 │         │   NATS                 │               │
│  │   ──────────────────── │         │   ──────────────────── │               │
│  │   • Worker discovery   │         │   • Message queue      │               │
│  │   • Metadata storage   │         │   • Prefill requests   │               │
│  │   • Health tracking    │         │   • JetStream enabled  │               │
│  │   Port: 2379/2389      │         │   Port: 4222/4232      │               │
│  └────────────────────────┘         └────────────────────────┘               │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

1. **Docker** installed and running
2. **NVIDIA GPU(s)** with CUDA support
3. **nvidia-docker** or NVIDIA Container Toolkit
4. **Llama-3.3-70B-Instruct** model downloaded locally

### Download Model (if needed)

```bash
# Using Hugging Face CLI
huggingface-cli download meta-llama/Llama-3.3-70B-Instruct \
  --local-dir /raid/bbednarski/models/Llama-3.3-70B-Instruct
```

### Verify GPU Access

```bash
nvidia-smi
# Should show available GPUs
```

---

## Starting Dynamo

All startup scripts are located in this directory (`external/dynamo/`).

### Option 1: Unified Mode (Development)

Single worker handling all operations. Simpler setup, good for development and testing.

```bash
cd /path/to/NeMo-Agent-Toolkit/external/dynamo

# Start Dynamo (do NOT use 'source')
bash start_dynamo_unified.sh > startup_output.txt 2>&1

# Wait for startup (watch GPU memory)
watch -n 1 nvidia-smi

# Verify Dynamo is running
curl -sv http://localhost:8099/health
# Expected: HTTP/1.1 200 OK

# when testing is complete, shut down the containers with:
bash stop_dynamo.sh
```

**Components started:**
- ETCD container (`etcd-dynamo`) on port 2389
- NATS container (`nats-dynamo`) on port 4232
- Dynamo container (`dynamo-sglang`) with unified worker on GPUs 4,5,6,7 (TP=4)

**Startup time**: ~5 minutes seconds for 70B model

### Option 2: Unified + Thompson Sampling Router (Production)

Unified worker with custom predictive KV-aware router using Thompson Sampling for optimal request routing.

```bash
cd /path/to/NeMo-Agent-Toolkit/external/dynamo

# Start Dynamo with Thompson Sampling router
bash start_dynamo_unified_thompson_hints.sh > startup_output.txt 2>&1

# Wait for startup
watch -n 1 nvidia-smi

# Verify
curl -sv http://localhost:8099/health

# when testing is complete, shut down the containers with:
bash stop_dynamo.sh
```

**Additional features:**
- Custom frontend with prefix hint header support
- Thompson Sampling router (LinTS + Beta bandits)
- KV cache overlap optimization
- Workload-aware routing based on OSL/IAT hints

**Custom components location:** `generalized/`
- `frontend.py` - Accepts x-prefix-* headers
- `processor.py` - Forwards hints to router, CSV metrics logging
- `router.py` - Thompson Sampling, KV overlap calculations

### Option 3: Disaggregated Mode (High-Throughput)

Separate prefill and decode workers for maximum throughput. More complex setup.

```bash
cd /path/to/NeMo-Agent-Toolkit/external/dynamo

# Start Dynamo disaggregated
bash start_dynamo_disagg.sh > startup_output.txt 2>&1

# Wait for startup (both workers need to initialize)
watch -n 1 nvidia-smi

# Verify
curl -sv http://localhost:8099/health

# when testing is complete, shut down the containers with:
bash stop_dynamo.sh
```

**Components started:**
- ETCD container on port 2379
- NATS container on port 4222
- Prefill Worker on GPUs 4,5 (TP=2)
- Decode Worker on GPUs 6,7 (TP=2)
- Dynamo Frontend on port 8099

**Startup time**: ~5 minutes (both workers must initialize)

**Note**: Disaggregated mode uses NIXL for KV cache transfer between workers.

### Verifying the Integration

After starting Dynamo with any of the above options, verify the integration is working.

#### Quick Validation with NAT

Run simple workflows to test basic connectivity and prefix header support:

```bash
cd /path/to/NeMo-Agent-Toolkit
source "${HOME}/.venvs/nat_dynamo_eval/bin/activate"

# Test basic Dynamo connectivity
nat run --config_file examples/dynamo_integration/react_benchmark_agent/configs/config_dynamo_e2e_test.yml \
  --input "What time is it?"

# Test Dynamo with dynamic prefix headers (for Predictive KV-Aware Cache router)
nat run --config_file examples/dynamo_integration/react_benchmark_agent/configs/config_dynamo_prefix_e2e_test.yml \
  --input "What time is it?"
```

#### Full Integration Test Suite

For comprehensive validation, run the integration test script:

```bash
cd /path/to/NeMo-Agent-Toolkit/external/dynamo
bash test_dynamo_integration.sh
```

**Environment variables** (optional):
- `DYNAMO_BACKEND` - Backend type: `sglang` # vllm and tensorRT still need to be developed
- `DYNAMO_MODEL` - Model name (default: `llama-3.3-70b`)
- `DYNAMO_PORT` - Frontend port (default: `8099`)

**Tests performed:**
1. NAT environment is active
2. Configuration files exist
3. Dynamo frontend is responding on the configured port
4. Basic chat completion request works
5. NAT workflow with basic config runs successfully
6. NAT workflow with prefix hints runs successfully

**Expected output (all tests passing):**
```
==========================================
Testing react_benchmark_agent with Dynamo
==========================================
Backend: sglang
Model: llama-3.3-70b
Port: 8099
==========================================

0. Checking if NAT environment is active...
✓ NAT command found

1. Checking if configuration files exist...
✓ Configuration files found

2. Checking if Dynamo frontend is running on port 8099...
✓ Dynamo frontend is running

3. Testing basic Dynamo endpoint...
✓ Dynamo endpoint is working

4. Testing NAT workflow with Dynamo (basic config)...
✓ Basic config test completed successfully

5. Testing NAT workflow with Dynamo (with prefix hints)...
✓ Prefix hints test completed successfully

==========================================
Test Summary
==========================================
Total tests: 6
Passed: 6
Failed: 0

✓ All tests passed!
```

If any tests fail, the script provides guidance on how to fix the issue.

---

## Stopping Dynamo

A single script stops all Dynamo components regardless of which mode was started:

```bash
cd /path/to/NeMo-Agent-Toolkit/external/dynamo
bash stop_dynamo.sh
```

**What it stops:**
- Dynamo container (`dynamo-sglang` or `dynamo-sglang-thompson`)
- ETCD container (`etcd-dynamo`)
- NATS container (`nats-dynamo`)

**Output:**
```
=========================================================
Stopping Dynamo SGLang FULL STACK
=========================================================

Stopping Dynamo container (standard)...
✓ Dynamo container stopped and removed

Stopping ETCD container...
✓ ETCD container stopped and removed

Stopping NATS container...
✓ NATS container stopped and removed

=========================================================
✓ All components stopped!
=========================================================
```

---

## Testing the Integration

An integration test script validates your Dynamo setup with NAT:

```bash
cd /path/to/NeMo-Agent-Toolkit/external/dynamo

# Activate NAT environment first
source "${HOME}/.venvs/nat_dynamo_eval/bin/activate"

# Run tests (do NOT use 'source')
./test_dynamo_integration.sh

# Show help
./test_dynamo_integration.sh --help
```

**What the test validates:**
1. NAT environment is activated
2. Configuration files exist
3. Dynamo frontend is running on port 8099
4. Dynamo endpoint responds correctly
5. NAT workflow executes with basic config
6. NAT workflow executes with prefix hints

**Expected output:**
```
==========================================
Test Summary
==========================================
Total tests: 6
Passed: 6
Failed: 0

✓ All tests passed!
==========================================
```

**Important**: Run the script directly with `./test_dynamo_integration.sh`, **NOT** with `source test_dynamo_integration.sh`. Using `source` will cause the script's `exit` commands to close your terminal.

### Quick Manual Tests

#### Using NAT (Recommended)

```bash
cd /path/to/NeMo-Agent-Toolkit
source "${HOME}/.venvs/nat_dynamo_eval/bin/activate"

# Basic Dynamo test
nat run --config_file examples/dynamo_integration/react_benchmark_agent/configs/config_dynamo_e2e_test.yml \
  --input "What time is it?"

# With prefix headers (for Thompson Sampling router)
nat run --config_file examples/dynamo_integration/react_benchmark_agent/configs/config_dynamo_prefix_e2e_test.yml \
  --input "What time is it?"
```

#### Using curl

```bash
# Basic chat completion
curl http://localhost:8099/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.3-70b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'

# Streaming test
curl http://localhost:8099/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.3-70b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50,
    "stream": true
  }'
```

---

## Monitoring

### Interactive Monitor

```bash
cd /path/to/NeMo-Agent-Toolkit/external/dynamo
./monitor_dynamo.sh
```

**Menu options:**
1. View Frontend logs
2. View Processor logs
3. View Router logs
4. View all component logs
5. View container logs
6. Test health endpoint
7. Test basic inference
8. Check GPU usage
9. Check process status

### Direct Commands

```bash
# View container logs
docker logs -f dynamo-sglang

# View ETCD logs
docker logs -f etcd-dynamo

# View NATS logs
docker logs -f nats-dynamo

# GPU utilization
watch -n 2 nvidia-smi

# Check running containers
docker ps --format "table {{.Names}}\t{{.Status}}"
```

---

## Dynamic Prefix Headers

When using the Thompson Sampling router (`start_dynamo_unified_thompson_hints.sh`), dynamic prefix headers enable optimal KV cache management and request routing.

### Overview

Prefix headers help the router:
- **Identify related requests** for KV cache reuse
- **Make routing decisions** based on workload characteristics
- **Track prefix state** for optimal worker selection
- **Improve throughput** through intelligent batching

### Configuration

Use the `dynamo` LLM type in your eval config. Prefix headers are sent by default:

```yaml
llms:
  dynamo_llm:
    _type: dynamo
    model_name: llama-3.3-70b
    base_url: http://localhost:8099/v1
    api_key: dummy
    
    # Prefix headers are enabled by default with template "nat-dynamo-{uuid}"
    # Optional: customize the template or routing hints
    # prefix_template: "react-benchmark-{uuid}"  # Custom template
    # prefix_template: null  # Set to null to disable prefix headers entirely
    prefix_total_requests: 10  # Expected requests per prefix
    prefix_osl: MEDIUM         # Output Sequence Length: LOW | MEDIUM | HIGH
    prefix_iat: MEDIUM         # Inter-Arrival Time: LOW | MEDIUM | HIGH
```

> **Note**: The `dynamo` LLM type automatically sends prefix headers using the default template `nat-dynamo-{uuid}`. To disable prefix headers entirely, set `prefix_template: null` in your config.

### Header Details

| Header | Description | Values |
|--------|-------------|--------|
| `x-prefix-id` | Unique identifier for request group | UUID-based string (null to disable all extra headers) |
| `x-prefix-total-requests` | Expected total requests for this prefix | Integer (1 for independent queries) |
| `x-prefix-osl` | Output Sequence Length hint | LOW (~50 tokens), MEDIUM (~200), HIGH (~500+) |
| `x-prefix-iat` | Inter-Arrival Time hint | LOW (rapid), MEDIUM (normal), HIGH (long delays) |

### Use Cases

#### Independent Queries (Evaluation)

Each question is independent, uses default prefix template:

```yaml
llms:
  eval_llm:
    _type: dynamo
    # prefix_template defaults to "nat-dynamo-{uuid}"
    prefix_total_requests: 1
    prefix_osl: MEDIUM
    prefix_iat: LOW  # Eval runs many queries quickly
```

#### Multi-Turn Conversations

Related requests should share a prefix:

```yaml
llms:
  chat_llm:
    _type: dynamo
    prefix_template: "chat-{uuid}"  # Optional: custom template
    prefix_total_requests: 8  # Average conversation length
    prefix_osl: MEDIUM
    prefix_iat: HIGH  # Users take time to type
```

#### Agent with Tool Calls

ReAct agents make multiple related calls:

```yaml
llms:
  agent_llm:
    _type: dynamo
    prefix_template: "agent-{uuid}"  # Optional: custom template
    prefix_total_requests: 5  # Typical tool call sequence
    prefix_osl: LOW   # Tool calls produce short responses
    prefix_iat: LOW   # Agent runs tool calls rapidly
```

### How It Works

1. **NAT Config** uses `_type: dynamo` (prefix headers enabled by default)
2. **Dynamo LLM Provider** generates unique UUID per request using the template
3. **Headers injected** into HTTP request:
   ```
   x-prefix-id: react-benchmark-a1b2c3d4e5f6g7h8
   x-prefix-total-requests: 1
   x-prefix-osl: MEDIUM
   x-prefix-iat: MEDIUM
   ```
4. **Dynamo Frontend** extracts headers
5. **Processor** tracks prefix state
6. **Router** makes routing decisions based on:
   - KV cache overlap with existing prefixes
   - Worker affinity for related requests
   - Load balancing across workers
   - Workload hints (OSL/IAT)

---

## Configuration Reference

### Script Variables

Each startup script has configurable variables at the top:

```bash
# start_dynamo_unified.sh
CONTAINER_NAME="dynamo-sglang"
WORKER_GPUS="4,5,6,7"
TP_SIZE=4
HTTP_PORT=8099
MODEL="/workspace/models/Llama-3.3-70B-Instruct"
SERVED_MODEL_NAME="llama-3.3-70b"
IMAGE="nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.6.1"
SHM_SIZE="16g"

# Infrastructure ports (non-default to avoid conflicts)
ETCD_CLIENT_PORT=2389
NATS_PORT=4232

# Local paths
LOCAL_MODEL_DIR="/raid/bbednarski/models/Llama-3.3-70B-Instruct"
```

### Customizing GPU Assignment

Edit the script to change GPU assignment:

```bash
# For different GPUs
WORKER_GPUS="0,1,2,3"

# In docker run command
--gpus '"device=0,1,2,3"'
```

### Customizing Model

```bash
# Different model
MODEL="/workspace/models/Llama-3.1-8B-Instruct"
SERVED_MODEL_NAME="llama-3.1-8b"
LOCAL_MODEL_DIR="/raid/your-username/models/Llama-3.1-8B-Instruct"
```

### Customizing Ports

```bash
# Different frontend port
HTTP_PORT=8080

# Different infrastructure ports
ETCD_CLIENT_PORT=2379
NATS_PORT=4222
```

---

## Troubleshooting

### Container Failed to Start

**Check logs:**
```bash
docker logs dynamo-sglang
```

**Common causes:**
- GPU not available
- Model path incorrect
- Port already in use

### Health Check Fails

```bash
# Check if container is running
docker ps --format '{{.Names}}'

# Check what's listening on port 8099
ss -tlnp | grep 8099
```

### ETCD Connection Issues

```bash
# Check ETCD health
curl http://localhost:2379/health 

# Check ETCD logs
docker logs etcd-dynamo
```

### NATS Connection Issues

```bash
# Check NATS is running
docker ps | grep nats-dynamo

# Check NATS logs
docker logs nats-dynamo
```

### Tokenizer Mismatch (Disaggregated Mode)

**Symptom**: `KeyError: 'token_ids'` or tokenizer errors

**Fix**: Clear ETCD data and restart
```bash
bash stop_dynamo.sh
# Wait a few seconds
bash start_dynamo_unified.sh
```

### Slow Model Loading

**Symptom**: Takes 3+ minutes to start

**Causes:**
- 70B model takes ~90-120 seconds normally
- Cold cache may require model download
- Insufficient GPU memory causes swapping

**Monitoring:**
```bash
# Watch GPU memory during startup
watch -n 1 nvidia-smi
```

### Streaming Not Working (Disaggregated Mode)

**Known Issue**: Disaggregated mode may have issues with streaming requests.

**Workaround**: Use unified mode for streaming, or use non-streaming requests:
```json
{"stream": false}
```

---

## File Structure

```
external/dynamo/                                # Dynamo backend
│
├── 📄 README.md                                # This file - Dynamo setup guide
├── 🔧 start_dynamo_unified.sh                  # Start Dynamo (unified mode)
├── 🔧 start_dynamo_unified_thompson_hints.sh   # Start with Thompson router
├── 🔧 start_dynamo_disagg.sh                   # Start Dynamo (disaggregated)
├── 🔧 stop_dynamo.sh                           # Stop all Dynamo services
├── 🔧 test_dynamo_integration.sh               # Integration tests
├── 🔧 monitor_dynamo.sh                        # Monitor running services
│
└── 📁 generalized/                             # Custom router components
    ├── frontend.py                             # Prefix header extraction
    ├── processor.py                            # Request processing + metrics
    └── router.py                               # Thompson Sampling router
```

---

## Quick Reference

### Commands

| Command | Description |
|---------|-------------|
| `bash start_dynamo_unified.sh` | Start unified mode |
| `bash start_dynamo_unified_thompson_hints.sh` | Start with Thompson router |
| `bash start_dynamo_disagg.sh` | Start disaggregated mode |
| `bash stop_dynamo.sh` | Stop all services |
| `./test_dynamo_integration.sh` | Run integration tests |
| `./monitor_dynamo.sh` | Interactive monitoring |
| `curl localhost:8099/health` | Health check |
| `docker logs -f dynamo-sglang` | View logs |
| `nat run --config_file examples/dynamo_integration/react_benchmark_agent/configs/config_dynamo_e2e_test.yml --input "..."` | Quick NAT validation |
| `nat run --config_file examples/dynamo_integration/react_benchmark_agent/configs/config_dynamo_prefix_e2e_test.yml --input "..."` | Test with prefix headers |

### Containers

| Container | Description |
|-----------|-------------|
| `dynamo-sglang` | Standard Dynamo worker |
| `etcd-dynamo` | Service discovery and metadata |
| `nats-dynamo` | Message queue for prefill requests |

### Related Documentation

- **[React Benchmark Agent](../../examples/dynamo_integration/react_benchmark_agent/README.md)** - Complete evaluation guide
- **[Architecture](../../examples/dynamo_integration/ARCHITECTURE.md)** - System diagrams
