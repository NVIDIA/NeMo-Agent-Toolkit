<!--
Copyright (c) 2025-2026 NVIDIA Corporation

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
<!-- path-check-skip-file -->

# Dynamo Backend Setup Guide

> [!NOTE]
> ⚠️ **EXPERIMENTAL**: This integration between NeMo Agent toolkit and Dynamo is experimental and under active development. APIs, configurations, and features may change without notice. We kindly ask that GitHub Issues are opened as bugs are issued quickly as features are subject to change.

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
| **Disaggregated** | `start_dynamo_disagg.sh` | Separate `prefill` and `decode` workers | High-throughput production |

### Architecture Overview

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                     DYNAMO BACKEND ARCHITECTURE                              │
└──────────────────────────────────────────────────────────────────────────────┘


                           CLIENT REQUEST
                        (eval, curl, Python)
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
│  │  │   4. OSL+IAT hints   │  │    • Learns optimal routing          │    │  │
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
│    (GPUs 0,1,2,3, TP=4)     │                      │                             │
│                             │                      │  ┌────────────────────────┐ │
│  ┌───────────────────────┐  │                      │  │   PREFILL WORKER       │ │
│  │  SGLang Engine        │  │                      │  │   (GPUs 0,1, TP=2)     │ │
│  │  ─────────────────    │  │                      │  │   • Initial KV compute │ │
│  │  • Model: Llama-3.3-70B  │                      │  │   • Sends KV via NIXL  │ │
│  │  • KV Cache Management│  │                      │  └───────────┬────────────┘ │
│  │  • Token Generation   │  │                      │              │              │
│  │  • Streaming Support  │  │                      │              │ NIXL KV      │
│  └───────────────────────┘  │                      │              │ Transfer     │
│                             │                      │              ▼              │
│  All operations in one      │                      │  ┌────────────────────────┐ │
│  worker                     │                      │  │   DECODE WORKER        │ │
│                             │                      │  │   (GPUs 2,3, TP=2)     │ │
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

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU Architecture** | NVIDIA Hopper (H100) or Blackwell (B200) | B200 for optimal performance |
| **GPU Count** | 2 GPUs for small models (2 workers) | 8 GPUs for optimal performance |
| **GPU Memory** | 80GB per GPU (H100) | 192GB per GPU (B200) |
| **System RAM** | 256GB | 512GB+ |

> **Note**: The [Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) model requires approximately 140GB of GPU memory when loaded with TP=4 (tensor parallelism across 4 GPUs). Ensure your GPU configuration has sufficient aggregate memory. If the Llama-3.3-70B-Instruct does not fit into your GPU memory, follow the same steps with The [Llama-3.1-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) for QA validation.

### Software Requirements

1. **Docker** installed and running (version 24.0+)
2. **NVIDIA Driver** with CUDA 12.0+ support
4. **Hugging Face CLI** for model downloads (optional, if model not already downloaded)
5. **Llama-3.3-70B-Instruct** model downloaded locally
6. **Python uv environment** python version 3.11-3.13


### Create a Python uv Environment

```bash
cd /path/to/NeMo-Agent-Toolkit
uv venv "${HOME}/.venvs/nat_dynamo_eval" --python 3.13
source "${HOME}/.venvs/nat_dynamo_eval/bin/activate"
```

### Download model weights (can skip if already done)

```bash
# Set your desired model directory
export DYNAMO_MODEL_DIR="${HOME}/models/Llama-3.3-70B-Instruct"

# Create the directory
# mkdir -p "$(dirname "$DYNAMO_MODEL_DIR")"
mkdir -p $DYNAMO_MODEL_DIR

# We will download the model weights directly from HuggingFace. Usage of
# llama models from requires approval from Meta. See `Access Notes` below.
# You will need to create a HuggingFace Access Token with read access in
# order to download the model. On the huggingface website visit:
# "Access Tokens" -> "+ Create access token" to generate a token starting
# with "hf_". Enter your token when prompted.
# Respond "n" when asked "Add token as git credential? (Y/n)"
uv pip install huggingface_hub
uv run huggingface-cli login  # Enter your HF token

uv run huggingface-cli download "meta-llama/Llama-3.3-70B-Instruct" \
  --local-dir "$DYNAMO_MODEL_DIR"
```

> **Access Note**: The Llama-3.3-70B-Instruct model requires approval from Meta. Request access at [huggingface.co/meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) before downloading.

### Environment Setup

Before running the Dynamo scripts, configure the following environment variables. See `.env.example` for a complete list of all available options.

```bash
cd external/dynamo/

# Copy and customize the example environment file
cp .env.example .env

# Edit with your settings
vi .env

# Source the environment before running scripts
source .env
```

Or set variables directly:

```bash
# Required: Set your model directory path
export DYNAMO_MODEL_DIR="/path/to/your/models/Llama-3.3-70B-Instruct" # or Llama-3.1-3B-Instruct for QA on H100 machines

# Optional: Set repository directory (for Thompson Sampling router)
export DYNAMO_REPO_DIR="/path/to/NeMo-Agent-Toolkit"

# Optional: Configure GPU devices (default: 0,1,2,3)
export DYNAMO_GPU_DEVICES="0,1,2,3"
```

### Verify GPU Access

```bash
# Check NVIDIA driver and GPU availability
nvidia-smi

# Expected output should show:
# - At least 4 GPUs (H100 or B200)
# - CUDA version 12.0+
# - Sufficient free memory per GPU
```

Example output for an 8x H100 system:

```text
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.65.06              Driver Version: 580.65.06      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA B200                    On  |   00000000:1B:00.0 Off |                    0 |
| N/A   31C    P0            187W / 1000W |  169082MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA B200                    On  |   00000000:43:00.0 Off |                    0 |
| N/A   31C    P0            187W / 1000W |  169178MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA B200                    On  |   00000000:52:00.0 Off |                    0 |
| N/A   36C    P0            193W / 1000W |  169230MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA B200                    On  |   00000000:61:00.0 Off |                    0 |
| N/A   36C    P0            195W / 1000W |  169230MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA B200                    On  |   00000000:9D:00.0 Off |                    0 |
| N/A   32C    P0            139W / 1000W |       4MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA B200                    On  |   00000000:C3:00.0 Off |                    0 |
| N/A   30C    P0            139W / 1000W |       4MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA B200                    On  |   00000000:D1:00.0 Off |                    0 |
| N/A   34C    P0            141W / 1000W |       4MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   7  NVIDIA B200                    On  |   00000000:DF:00.0 Off |                    0 |
| N/A   35C    P0            139W / 1000W |       4MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A         2700092      C   VLLM::Worker_TP0_EP0                  16901... |
|    1   N/A  N/A         2700093      C   VLLM::Worker_TP1_EP1                  16901... |
|    2   N/A  N/A         2700094      C   VLLM::Worker_TP2_EP2                  16901... |
|    3   N/A  N/A         2700095      C   VLLM::Worker_TP3_EP3                  16901... |
+-----------------------------------------------------------------------------------------+
```

### Verify Docker and NVIDIA Container Toolkit

```bash
# Verify Docker is running
docker info
```

---

## Starting Dynamo

Startup scripts can be found in the same directory (`NeMo-Agent-Toolkit/external/dynamo/`) at this `README.md`

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
# Expected: "HTTP/1.1 200 OK"

# when testing is complete, shut down the containers with:
bash stop_dynamo.sh
```

**Components started:**
- `etcd` container (`etcd-dynamo`) on port 2389
- `nats` container (`nats-dynamo`) on port 4232
- Dynamo container (`dynamo-sglang`) with unified worker on GPUs 0,1,2,3 (TP=4)

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
- Workload-aware routing based on OSL and IAT hints

**Custom components location:** `generalized/`
- `frontend.py` - Accepts x-prefix-* headers
- `processor.py` - Forwards hints to router, CSV metrics logging
- `router.py` - Thompson Sampling, KV overlap calculations

### Option 3: Disaggregated Mode (High-Throughput)

Separate `prefill` and `decode` workers for maximum throughput. More complex setup.

```bash
cd /path/to/NeMo-Agent-Toolkit/external/dynamo

export DYNAMO_PREFILL_GPUS="0,1"
export DYNAMO_DECODE_GPUS="2,3"

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
- `etcd` container on port 2379
- `nats` container on port 4222
- `prefill` Worker on GPUs 0,1 (TP=2)
- `decode` Worker on GPUs 2,3 (TP=2)
- Dynamo Frontend on port 8099

**Startup time**: ~5 minutes (both workers must initialize)

**Note**: Disaggregated mode uses NIXL for KV cache transfer between workers.

### Verifying the Integration

After starting Dynamo with any of the above options, verify the integration is working.

#### Quick Validation with NeMo Agent Toolkit

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
- `DYNAMO_BACKEND` - Backend type: `sglang` # `vllm` and tensorRT still need to be developed
- `DYNAMO_MODEL` - Model name (default: `llama-3.3-70b`)
- `DYNAMO_PORT` - Frontend port (default: `8099`)

**Tests performed:**
1. NeMo Agent toolkit environment is active
2. Configuration files exist
3. Dynamo frontend is responding on the configured port
4. Basic chat completion request works
5. Workflow with basic config runs successfully
6. Workflow with prefix hints runs successfully

**Expected output (all tests passing):**
```text
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
- `etcd` container (`etcd-dynamo`)
- `nats` container (`nats-dynamo`)

**Output:**
```text
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

An integration test script validates your Dynamo setup with NeMo Agent toolkit:

```bash
cd /path/to/NeMo-Agent-Toolkit/external/dynamo

# Activate the environment first
source "${HOME}/.venvs/nat_dynamo_eval/bin/activate"

# Run tests (do NOT use 'source')
./test_dynamo_integration.sh

# Show help
./test_dynamo_integration.sh --help
```

**What the test validates:**
1. The environment is activated
2. Configuration files exist
3. Dynamo frontend is running on port 8099
4. Dynamo endpoint responds correctly
5. Workflow executes with basic config
6. Workflow executes with prefix hints

**Expected output:**
```text
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

#### Using NeMo Agent Toolkit (Recommended)

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

# View `etcd` logs
docker logs -f etcd-dynamo

# View `nats` logs
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

### KV overlap routing: requirements and failure mode

Prefix headers do not include KV cache overlap. The router computes KV cache overlap scores by querying the backend through `dynamo.llm.KvIndexer`.

If overlap scores are unavailable, the router cannot account for KV cache match when routing and will behave like a non-KV-aware router for that signal.

This can happen in the following configuration:
- You are using a Dynamo image or build that does not include `dynamo.llm` KV routing classes. In this case, the router logs a warning that `dynamo.llm` is not available and overlap scores will be empty.

To confirm overlap scores are missing, check `router_metrics.csv` and verify that `overlap_chosen` is always `0.000000`.

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

1. **NeMo Agent Toolkit Configurations** uses `_type: dynamo` (prefix headers enabled by default)
2. **Dynamo LLM Provider** generates unique UUID per request using the template
3. **Headers injected** into HTTP request:
   ```text
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
   - Workload hints (OSL and IAT)

---

## Configuration Reference

### Environment Variables

The startup scripts support configuration through environment variables. Set these before running the scripts:

| Variable | Description | Default |
|----------|-------------|---------|
| `DYNAMO_MODEL_DIR` | Local path to the model directory | (required) |
| `DYNAMO_REPO_DIR` | Path to NeMo-Agent-Toolkit repository | Auto-detected |
| `DYNAMO_GPU_DEVICES` | Comma-separated GPU device IDs | `0,1,2,3` |
| `DYNAMO_HTTP_PORT` | Frontend HTTP port | `8099` |
| `DYNAMO_ETCD_PORT` | `etcd` client port | `2389` |
| `DYNAMO_NATS_PORT` | `nats` messaging port | `4232` |
| `DYNAMO_METRICS_URL` | Prometheus metrics endpoint URL for the router | `http://localhost:9090/metrics` |
| `ROUTER_METRICS_CSV` | Path to CSV file for router decision logging | `router_metrics.csv` |

Example configuration:

```bash
# Configure environment before running scripts
export DYNAMO_MODEL_DIR="/path/to/models/Llama-3.3-70B-Instruct"
export DYNAMO_GPU_DEVICES="0,1,2,3"
export DYNAMO_HTTP_PORT="8099"

# Then start Dynamo
bash start_dynamo_unified.sh
```

### Script Variables

Each startup script also has configurable variables at the top that can be edited directly:

```bash
# start_dynamo_unified.sh
CONTAINER_NAME="dynamo-sglang"
WORKER_GPUS="${DYNAMO_GPU_DEVICES:-0,1,2,3}"    # Override with env var or edit default
TP_SIZE=4
HTTP_PORT="${DYNAMO_HTTP_PORT:-8099}"
MODEL="/workspace/models/Llama-3.3-70B-Instruct"
SERVED_MODEL_NAME="llama-3.3-70b"
IMAGE="nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.6.1"
SHM_SIZE="16g"

# Infrastructure ports (non-default to avoid conflicts)
ETCD_CLIENT_PORT="${DYNAMO_ETCD_PORT:-2389}"
NATS_PORT="${DYNAMO_NATS_PORT:-4232}"

# Local paths - MUST be set via environment variable or edited here
LOCAL_MODEL_DIR="${DYNAMO_MODEL_DIR:?Error: DYNAMO_MODEL_DIR environment variable must be set}"
```

### Customizing GPU Assignment

Option 1: Use environment variable (recommended):

```bash
export DYNAMO_GPU_DEVICES="0,1,2,3"
bash start_dynamo_unified.sh
```

Option 2: Edit the script directly:

```bash
# In the script, change:
WORKER_GPUS="0,1,2,3"

# The docker run command will use:
--gpus '"device=0,1,2,3"'
```

### Customizing Model

For a different model, update both the model directory and served name:

```bash
# Set environment variable for model path
export DYNAMO_MODEL_DIR="${HOME}/models/Llama-3.1-8B-Instruct"

# Edit script variables for model metadata
MODEL="/workspace/models/Llama-3.1-8B-Instruct"
SERVED_MODEL_NAME="llama-3.1-8b"
TP_SIZE=2  # Smaller models may need fewer GPUs
```

### Customizing Ports

Option 1: Use environment variables:

```bash
export DYNAMO_HTTP_PORT="8080"
export DYNAMO_ETCD_PORT="2379"
export DYNAMO_NATS_PORT="4222"
bash start_dynamo_unified.sh
```

Option 2: Edit script directly:

```bash
HTTP_PORT=8080
ETCD_CLIENT_PORT=2379
NATS_PORT=4222
```

---

## Metrics CSV Files

The Thompson Sampling router (`start_dynamo_unified_thompson_hints.sh`) produces three CSV files for monitoring and analysis. These files are located in `/workspace/metrics/` inside the container.

### Accessing Metrics

```bash
# From the host
docker exec dynamo-sglang cat /workspace/metrics/router_metrics.csv
docker exec dynamo-sglang cat /workspace/metrics/processor_requests.csv
docker exec dynamo-sglang cat /workspace/metrics/frontend_throughput.csv

# From inside the container
docker exec -it dynamo-sglang bash
cat /workspace/metrics/router_metrics.csv
```

### router_metrics.csv

Logs every routing decision made by the Thompson Sampling router.

**Columns:**

| Column | Description |
|--------|-------------|
| `ts_epoch_ms` | Timestamp in milliseconds since epoch |
| `tokens_len` | Number of tokens in the request |
| `prefix_id` | Unique prefix identifier (auto-generated or from header) |
| `reuse_after` | Remaining reuse budget after this request |
| `chosen_worker` | Integer ID of the selected worker |
| `overlap_chosen` | KV cache overlap score (0.0-1.0) |
| `decode_cost` | Estimated `decode` cost |
| `prefill_cost` | Estimated `prefill` cost |
| `iat_level` | Inter-arrival time hint (LOW, MEDIUM, or HIGH) |
| `stickiness` | Worker affinity score |
| `load_mod` | Load modifier applied |

**Example output:**

```csv
ts_epoch_ms,tokens_len,prefix_id,reuse_after,chosen_worker,overlap_chosen,decode_cost,prefill_cost,iat_level,stickiness,load_mod
1767923263058,38,auto-9e05dbb0682f458a89b82f64bb328011,0,7587892060544177931,0.000000,2.000000,0.037109,MEDIUM,0.000,1.000000
```

### processor_requests.csv

Logs latency metrics for each processed request.

**Columns:**

| Column | Description |
|--------|-------------|
| `num_tokens` | Number of output tokens generated |
| `latency_ms` | Total request latency in milliseconds |
| `latency_ms_per_token` | Average latency per token |

**Example output:**

```csv
num_tokens,latency_ms,latency_ms_per_token
10,70152.021,7015.202100
```

### frontend_throughput.csv

Logs throughput metrics at regular intervals (default: every 5 seconds).

**Columns:**

| Column | Description |
|--------|-------------|
| `ts_epoch_ms` | Timestamp in milliseconds since epoch |
| `requests` | Number of requests completed in this interval |
| `interval_s` | Length of the measurement interval in seconds |
| `req_per_sec` | Computed requests per second |

**Example output:**

```csv
ts_epoch_ms,requests,interval_s,req_per_sec
1767923267849,0,5.000,0.000000
1767923272850,0,5.000,0.000000
1767923337856,1,5.000,0.200000
1767923342856,0,5.000,0.000000
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

### `etcd` Connection Issues

```bash
# Check `etcd` health
curl http://localhost:2389/health

# Check `etcd` logs
docker logs etcd-dynamo
```

### `nats` Connection Issues

```bash
# Check `nats` is running
docker ps | grep nats-dynamo

# Check `nats` logs
docker logs nats-dynamo
```

### Tokenizer Mismatch (Disaggregated Mode)

**Symptom**: `KeyError: 'token_ids'` or tokenizer errors

**Fix**: Clear `etcd` data and restart
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

```text
external/dynamo/                                # Dynamo backend
│
├── 📄 README.md                                # This file - Dynamo setup guide
├── 📄 .env.example                              # Example environment variables
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
| `nat run --config_file examples/dynamo_integration/react_benchmark_agent/configs/config_dynamo_e2e_test.yml --input "..."` | Quick NeMo Agent toolkit validation |
| `nat run --config_file examples/dynamo_integration/react_benchmark_agent/configs/config_dynamo_prefix_e2e_test.yml --input "..."` | Test with prefix headers |

### Containers

| Container | Description |
|-----------|-------------|
| `dynamo-sglang` | Standard Dynamo worker |
| `etcd-dynamo` | Service discovery and metadata |
| `nats-dynamo` | Message queue for `prefill` requests |

### Related Documentation

- **[React Benchmark Agent](../../examples/dynamo_integration/react_benchmark_agent/README.md)** - Complete evaluation guide
- **[Architecture](../../examples/dynamo_integration/ARCHITECTURE.md)** - System diagrams
