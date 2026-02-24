#!/bin/bash
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

# Dynamo vLLM with OPTIMIZED Thompson Sampling Router Architecture
# 
# Key difference from SGLang version:
#   - Uses vLLM backend instead of SGLang
#   - vLLM has native KVBM support for KV event publishing
#   - Different CLI flags (--block-size vs --page-size, etc.)
#   - Enables radix/prefix caching by default (no --disable-radix-cache)
#
# Architecture:
#   Client → Default Dynamo Frontend (tokenization + nvext parsing)
#         ↓ PreprocessedRequest with annotations
#   Custom Processor (extracts hints, queries router)
#         ↓ RouterRequest
#   Custom Router (Thompson Sampling + KV overlap)
#         ↓ worker_id
#   vLLM Backend Worker
#         ↓ response tokens
#   Processor sends feedback to Router
#
# Components:
#   - ETCD (metadata and worker discovery)
#   - NATS (message queue for KV events)
#   - Default Dynamo Frontend (HTTP API on port 8000)
#   - Custom Router (Thompson Sampling + KV overlap)
#   - Custom Processor (hint extraction + routing)
#   - vLLM Workers (unified mode, multiple workers with TP=2 each)
#
# Prometheus Metrics:
#   - Frontend: http://localhost:8000/metrics
#   - Backend/Router/Processor: http://localhost:8081/metrics
#
# To stop all components: bash stop_dynamo.sh

set -euo pipefail

# Load environment variables from .env file if present
# Supports: DYNAMO_FROM_SOURCE, DYNAMO_IMAGE, and all DYNAMO_* overrides
SCRIPT_DIR_EARLY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR_EARLY}/.env" ]; then
    set -a
    # Strip inline comments before sourcing (bash doesn't handle them natively)
    source <(grep -v '^\s*#' "${SCRIPT_DIR_EARLY}/.env" | sed 's/[[:space:]]*#.*$//')
    set +a
fi

# Configuration Variables (can be overridden via environment variables)
# See env.example for documentation on each variable
CONTAINER_NAME="dynamo-vllm"
WORKER_GPUS="${DYNAMO_GPU_DEVICES:-0,1,2,3,4,5,6,7}"
TP_SIZE="${DYNAMO_TP_SIZE:-4}"
HTTP_PORT="${DYNAMO_HTTP_PORT:-8000}"
# Metrics ports - each component gets its own port to avoid conflicts
# Using 18xxx range to avoid conflicts with common services
# Workers use sequential ports starting at WORKER_METRICS_PORT (18081, 18082, ...)
# Router and Processor are offset to allow for many workers
WORKER_METRICS_PORT="${DYNAMO_WORKER_METRICS_PORT:-18081}"
ROUTER_METRICS_PORT="${DYNAMO_ROUTER_METRICS_PORT:-18090}"
PROCESSOR_METRICS_PORT="${DYNAMO_PROCESSOR_METRICS_PORT:-18091}"
# SERVED_MODEL_NAME and MODEL are derived below after DYNAMO_MODEL_DIR is validated
SERVED_MODEL_NAME=""  # set after validation

# ============================================================================
# Image and MultiLRU Configuration Logic
# ============================================================================
# Three modes (controlled via .env or environment variables):
#
# 1. Source-built image (DYNAMO_FROM_SOURCE=true):
#      - Uses DYNAMO_IMAGE (e.g. "dynamo-vllm-source:main") built from the
#        dynamo main branch at DYNAMO_SOURCE_DIR.
#      - Forces use of multilru scripts (processor_multilru.py / router_multilru.py).
#      - Build the image first:
#          cd $DYNAMO_SOURCE_DIR
#          python container/render.py --framework=vllm --target=runtime --output-short-filename
#          docker build -t dynamo-vllm-source:main -f container/rendered.Dockerfile .
#
# 2. Custom MultiLRU image, NGC-based (DYNAMO_USE_MULTILRU=true):
#      - Uses DYNAMO_VLLM_IMAGE (default: "dynamo-multi-lru:latest").
#      - Uses multilru scripts.
#
# 3. Standard NGC image (default):
#      - Uses nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.9.0
#      - Uses standard router.py / processor.py scripts.
# ============================================================================

if [ "${DYNAMO_FROM_SOURCE:-false}" = "true" ]; then
    # Source-built image mode: use DYNAMO_IMAGE from .env
    if [ -z "${DYNAMO_IMAGE:-}" ]; then
        echo "ERROR: DYNAMO_FROM_SOURCE=true but DYNAMO_IMAGE is not set."
        echo "  Set DYNAMO_IMAGE in .env (e.g. DYNAMO_IMAGE=dynamo-vllm-source:main)"
        exit 1
    fi
    IMAGE="${DYNAMO_IMAGE}"
    DYNAMO_USE_MULTILRU="true"

    # Verify the image exists; offer build instructions if not
    if ! docker image inspect "${IMAGE}" > /dev/null 2>&1; then
        echo "✗ ERROR: Source image '${IMAGE}' not found."
        echo ""
        echo "Build it from the dynamo main branch:"
        if [ -n "${DYNAMO_SOURCE_DIR:-}" ]; then
            echo "  cd ${DYNAMO_SOURCE_DIR}"
        else
            echo "  cd /path/to/dynamo   # set DYNAMO_SOURCE_DIR in .env to customise"
        fi
        echo "  python container/render.py --framework=vllm --target=runtime --output-short-filename"
        echo "  docker build -t ${IMAGE} -f container/rendered.Dockerfile ."
        exit 1
    fi
    echo "✓ Using source-built image: ${IMAGE}"
elif [ "${DYNAMO_USE_MULTILRU:-}" = "true" ]; then
    # Explicitly enabled MultiLRU with NGC-based custom image
    DYNAMO_USE_MULTILRU="true"
    IMAGE="${DYNAMO_VLLM_IMAGE:-dynamo-multi-lru:latest}"
else
    # Default: standard NGC image, no MultiLRU
    DYNAMO_USE_MULTILRU="false"
    if [ "${DYNAMO_VLLM_IMAGE:-}" != "dynamo-multi-lru:latest" ] && [ -n "${DYNAMO_VLLM_IMAGE:-}" ]; then
        IMAGE="${DYNAMO_VLLM_IMAGE}"
    else
        IMAGE="nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.9.0"
    fi
fi

SHM_SIZE="${DYNAMO_SHM_SIZE:-16g}"
WORKER_INIT_TIMEOUT_S="${DYNAMO_WORKER_INIT_TIMEOUT_S:-1800}"

# KV Cache Configuration
# Block size in tokens - must match between vLLM (--block-size) and Frontend (--kv-cache-block-size)
KV_BLOCK_SIZE="${DYNAMO_KV_BLOCK_SIZE:-16}"
# Fraction of GPU memory for KV cache (0.0-1.0). Reduce to test cache pressure/degradation.
# NOTE: 0.85 is safer than 0.9+ to avoid OOM during vLLM warmup with large max_num_seqs
# Reads DYNAMO_MEM_FRACTION_STATIC first (shared with SGLang script), then DYNAMO_GPU_MEMORY_UTILIZATION
GPU_MEMORY_UTILIZATION="${DYNAMO_MEM_FRACTION_STATIC:-${DYNAMO_GPU_MEMORY_UTILIZATION:-0.85}}"
# Maximum concurrent sequences per worker. Lower values use less memory during warmup.
# vLLM default is 1024, but this can cause OOM on memory-constrained setups.
MAX_NUM_SEQS="${DYNAMO_MAX_NUM_SEQS:-256}"
# Override the number of GPU KV cache blocks (for experiments with limited cache).
# Set to a small number (e.g., 8-16) to force cache eviction behavior.
# Leave empty/unset to use automatic calculation based on GPU memory.
NUM_GPU_BLOCKS_OVERRIDE="${DYNAMO_NUM_GPU_BLOCKS_OVERRIDE:-}"

# Compute container-internal GPU indices (GPUs are renumbered 0,1,2,... inside the container)
NUM_GPUS=$(echo "$WORKER_GPUS" | tr ',' '\n' | wc -l)
CONTAINER_GPU_INDICES=$(seq -s, 0 $((NUM_GPUS - 1)))

# Calculate number of workers based on available GPUs and TP size
NUM_WORKERS=$((NUM_GPUS / TP_SIZE))

# Validate GPU/TP sizing
if [ "$TP_SIZE" -le 0 ] 2>/dev/null; then
    echo "ERROR: TP_SIZE must be a positive integer (got: '$TP_SIZE')" >&2
    echo "  WORKER_GPUS=$WORKER_GPUS  NUM_GPUS=$NUM_GPUS  TP_SIZE=$TP_SIZE" >&2
    exit 1
fi
if [ "$NUM_GPUS" -lt "$TP_SIZE" ]; then
    echo "ERROR: Not enough GPUs for the requested TP size (NUM_GPUS=$NUM_GPUS < TP_SIZE=$TP_SIZE)" >&2
    echo "  WORKER_GPUS=$WORKER_GPUS  NUM_GPUS=$NUM_GPUS  TP_SIZE=$TP_SIZE" >&2
    exit 1
fi
if [ $((NUM_GPUS % TP_SIZE)) -ne 0 ]; then
    echo "ERROR: NUM_GPUS ($NUM_GPUS) is not divisible by TP_SIZE ($TP_SIZE)" >&2
    echo "  WORKER_GPUS=$WORKER_GPUS  NUM_GPUS=$NUM_GPUS  TP_SIZE=$TP_SIZE  NUM_WORKERS would be $NUM_WORKERS" >&2
    exit 1
fi
if [ "$NUM_WORKERS" -le 0 ]; then
    echo "ERROR: NUM_WORKERS is 0 — no workers can be started with this GPU/TP configuration" >&2
    echo "  WORKER_GPUS=$WORKER_GPUS  NUM_GPUS=$NUM_GPUS  TP_SIZE=$TP_SIZE" >&2
    exit 1
fi

# vLLM-specific: Enable KVBM event publishing for radix tree observability
# Each worker needs a unique KV event port - configured via DYN_VLLM_KV_EVENT_PORT
# Port allocation: Worker 0 = 20080, Worker 1 = 20081, etc.
# This is set per-worker at startup time below
ENABLE_KV_EVENTS="${DYNAMO_ENABLE_KV_EVENTS:-true}"
KV_EVENT_BASE_PORT="${DYNAMO_KV_EVENT_BASE_PORT:-20080}"

# Local paths - DYNAMO_MODEL_DIR must be set or script will error
if [ -z "${DYNAMO_MODEL_DIR:-}" ]; then
    echo "ERROR: DYNAMO_MODEL_DIR environment variable must be set"
    echo ""
    echo "Example:"
    echo "  export DYNAMO_MODEL_DIR=\"/path/to/your/models/Llama-3.3-70B-Instruct\""
    echo ""
    echo "Then run this script again."
    exit 1
fi

# Validate model directory
if [ -d "${DYNAMO_MODEL_DIR}" ]; then
    if [ ! -f "${DYNAMO_MODEL_DIR}/config.json" ]; then
        echo "ERROR: ${DYNAMO_MODEL_DIR} exists but is not a valid model directory"
        echo ""
        echo "Missing: config.json"
        echo ""
        echo "Find it: find ~/.cache/huggingface/hub -name config.json -path '*Llama-3.3-70B*'"
        exit 1
    fi

    if ! grep -q '"model_type"' "${DYNAMO_MODEL_DIR}/config.json" 2>/dev/null; then
        echo "ERROR: ${DYNAMO_MODEL_DIR}/config.json is missing 'model_type' field"
        echo ""
        echo "This usually means incomplete/corrupted download. Try:"
        echo "  rm -rf ${DYNAMO_MODEL_DIR}"
        echo "  hf download meta-llama/Llama-3.3-70B-Instruct --local-dir ${DYNAMO_MODEL_DIR}"
        exit 1
    fi
fi
# Resolve LOCAL_MODEL_DIR to an absolute path (expands ~ and relative paths)
LOCAL_MODEL_DIR="$(eval echo "${DYNAMO_MODEL_DIR}")"
# Container-internal model path: always a clean /workspace/models/<name>
MODEL="/workspace/models/$(basename "$LOCAL_MODEL_DIR")"
SERVED_MODEL_NAME="${DYNAMO_MODEL_NAME:-$(basename "$LOCAL_MODEL_DIR")}"

# Repository directory - auto-detect from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUSTOM_DYNAMO_DIR="${SCRIPT_DIR}/components"

echo "========================================================="
echo "Dynamo vLLM with OPTIMIZED Thompson Sampling Router"
echo "========================================================="
if [ "${DYNAMO_FROM_SOURCE:-false}" = "true" ]; then
    echo "Configuration: Source-Built Image Mode (DYNAMO_FROM_SOURCE=true)"
    echo "  Image: $IMAGE (built from dynamo main branch)"
    echo "  Scripts: multilru (processor_multilru.py / router_multilru.py)"
elif [ "$DYNAMO_USE_MULTILRU" = "true" ]; then
    echo "Configuration: MultiLRU Mode (custom image: $IMAGE)"
else
    echo "Configuration: Standard Mode (image: $IMAGE)"
fi
echo "Model: $SERVED_MODEL_NAME (from $LOCAL_MODEL_DIR)"
echo "Container: $CONTAINER_NAME"
echo "HTTP Port: $HTTP_PORT (default Dynamo frontend)"
echo "Metrics Ports:"
echo "  - Worker:    $WORKER_METRICS_PORT (KV cache, internal)"
echo "  - Router:    $ROUTER_METRICS_PORT (Thompson routing)"
echo "  - Processor: $PROCESSOR_METRICS_PORT (KVE metrics)"
echo ""
echo "Architecture Differences (vs SGLang version):"
echo "  - vLLM backend (native KVBM support)"
echo "  - KV events enabled: $ENABLE_KV_EVENTS"
echo "  - Different CLI flags (--block-size, --gpu-memory-utilization)"
echo "  - Prefix caching enabled by default"
echo ""
echo "Components:"
echo "  - ETCD (metadata and discovery)"
echo "  - NATS (message queue for KV events)"
echo "  - Default Frontend (HTTP API on port $HTTP_PORT)"
echo "  - Custom Router (Thompson Sampling + KV overlap)"
echo "  - Custom Processor (hint extraction + routing)"
echo "  - vLLM Worker (unified mode)"
echo ""
echo "Backend Workers:"
echo "  Workers: $NUM_WORKERS (GPUs: $NUM_GPUS, TP=$TP_SIZE per worker)"
echo "  GPUs: $WORKER_GPUS"
echo "  Mode: UNIFIED (no prefill/decode disaggregation)"
echo ""
echo "KV Cache Configuration:"
echo "  Block Size: $KV_BLOCK_SIZE tokens (--block-size / --kv-cache-block-size)"
echo "  GPU Mem Utilization: $GPU_MEMORY_UTILIZATION (--gpu-memory-utilization, from DYNAMO_MEM_FRACTION_STATIC)"
echo "  Max Concurrent Seqs: $MAX_NUM_SEQS (--max-num-seqs, prevents OOM during warmup)"
echo "  KV Events: $ENABLE_KV_EVENTS (KVBM event publishing)"
if [ "$ENABLE_KV_EVENTS" = "true" ] && [ "$NUM_WORKERS" -gt 1 ]; then
    echo "    Per-worker ports: $KV_EVENT_BASE_PORT - $((KV_EVENT_BASE_PORT + NUM_WORKERS - 1))"
fi
if [ -n "$NUM_GPU_BLOCKS_OVERRIDE" ]; then
    echo "  ⚠️  GPU Blocks Override: $NUM_GPU_BLOCKS_OVERRIDE (EXPERIMENT MODE - limited cache!)"
fi
echo ""
echo "========================================================="

# Select router/processor scripts based on DYNAMO_USE_MULTILRU
if [ "$DYNAMO_USE_MULTILRU" = "true" ]; then
    ROUTER_SCRIPT="router_multilru.py"
    PROCESSOR_SCRIPT="processor_multilru.py"
else
    ROUTER_SCRIPT="router.py"
    PROCESSOR_SCRIPT="processor.py"
fi

# Verify selected components exist
if [ ! -f "$CUSTOM_DYNAMO_DIR/$ROUTER_SCRIPT" ]; then
    echo "✗ ERROR: Custom $ROUTER_SCRIPT not found at: $CUSTOM_DYNAMO_DIR/$ROUTER_SCRIPT"
    exit 1
fi
if [ ! -f "$CUSTOM_DYNAMO_DIR/$PROCESSOR_SCRIPT" ]; then
    echo "✗ ERROR: Custom $PROCESSOR_SCRIPT not found at: $CUSTOM_DYNAMO_DIR/$PROCESSOR_SCRIPT"
    exit 1
fi
echo "✓ Custom components found in: $CUSTOM_DYNAMO_DIR"
echo "  Router:    $ROUTER_SCRIPT"
echo "  Processor: $PROCESSOR_SCRIPT"
echo ""

# Start ETCD if not running
if docker ps -a --format '{{.Names}}' | grep -q "^etcd-dynamo$"; then
    echo "Removing existing ETCD container..."
    docker rm -f etcd-dynamo
fi

echo "Starting ETCD container..."
docker run -d \
  --name etcd-dynamo \
  --network host \
  -e ALLOW_NONE_AUTHENTICATION=yes \
  -e ETCD_LISTEN_CLIENT_URLS=http://0.0.0.0:2379 \
  -e ETCD_ADVERTISE_CLIENT_URLS=http://localhost:2379 \
  bitnamilegacy/etcd:3.6.1

# Wait for ETCD to be ready
echo "Waiting for ETCD to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:2379/health > /dev/null 2>&1; then
        echo "✓ ETCD is ready"
        sleep 2
        break
    fi
    if [ $i -eq 30 ]; then
        echo "✗ ERROR: ETCD failed to start within 30 seconds"
        docker logs etcd-dynamo
        exit 1
    fi
    sleep 1
done

# Start NATS if not running
if docker ps -a --format '{{.Names}}' | grep -q "^nats-dynamo$"; then
    echo "Removing existing NATS container..."
    docker rm -f nats-dynamo
fi

echo "Starting NATS container..."
docker run -d \
  --name nats-dynamo \
  --network host \
  nats:2.11.4 \
  -js

# Wait for NATS to be ready
echo "Waiting for NATS to be ready..."
for i in {1..30}; do
    if timeout 2 bash -c 'cat < /dev/null > /dev/tcp/localhost/4222' 2>/dev/null; then
        echo "✓ NATS is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "✗ ERROR: NATS failed to start within 30 seconds"
        docker logs nats-dynamo
        exit 1
    fi
    sleep 1
done
echo ""

# Start monitoring stack (Prometheus + Grafana) if not running
MONITORING_DIR="${SCRIPT_DIR}/monitoring"
if [ -f "$MONITORING_DIR/docker-compose.yml" ]; then
    PROMETHEUS_RUNNING=$(docker ps --format '{{.Names}}' | grep -q "^dynamo-prometheus$" && echo "true" || echo "false")
    GRAFANA_RUNNING=$(docker ps --format '{{.Names}}' | grep -q "^dynamo-grafana$" && echo "true" || echo "false")
    
    if [ "$PROMETHEUS_RUNNING" = "false" ] || [ "$GRAFANA_RUNNING" = "false" ]; then
        echo "Starting monitoring stack (Prometheus + Grafana)..."
        cd "$MONITORING_DIR"
        docker compose up -d
        cd "$SCRIPT_DIR"
        
        # Wait for Prometheus to be ready
        echo "Waiting for Prometheus to be ready..."
        for i in {1..30}; do
            if curl -s http://localhost:9090/-/ready > /dev/null 2>&1; then
                echo "✓ Prometheus is ready (http://localhost:9090)"
                break
            fi
            if [ $i -eq 30 ]; then
                echo "⚠ WARNING: Prometheus may not be fully ready yet"
            fi
            sleep 1
        done
        
        # Wait for Grafana to be ready
        echo "Waiting for Grafana to be ready..."
        for i in {1..30}; do
            if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
                echo "✓ Grafana is ready (http://localhost:3000)"
                break
            fi
            if [ $i -eq 30 ]; then
                echo "⚠ WARNING: Grafana may not be fully ready yet"
            fi
            sleep 1
        done
        echo ""
    else
        echo "✓ Monitoring stack already running"
        echo "  Prometheus: http://localhost:9090"
        echo "  Grafana:    http://localhost:3000"
        echo ""
    fi
else
    echo "⚠ Monitoring docker-compose.yml not found at: $MONITORING_DIR"
    echo "  Skipping monitoring stack startup"
    echo ""
fi

# Clean up existing Dynamo container if it exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Removing existing Dynamo container: $CONTAINER_NAME"
    docker rm -f $CONTAINER_NAME
fi

# Verify HF_TOKEN is set
if [ -z "${HF_TOKEN:-}" ]; then
    echo ""
    echo "⚠ HF_TOKEN environment variable is not set."
    echo ""
    if [ -d "$LOCAL_MODEL_DIR" ]; then
        echo "✓ Local model found - proceeding without HF_TOKEN"
        HF_TOKEN="dummy"
    else
        echo "✗ Local model NOT found and no HF_TOKEN to download it"
        echo ""
        printf "Please enter your HuggingFace token (or press Enter to skip): "
        read -s -r HF_TOKEN
        echo ""
        if [ -z "$HF_TOKEN" ]; then
            echo "WARNING: Proceeding without HF_TOKEN."
            HF_TOKEN="dummy"
        else
            echo "✓ HuggingFace token received"
        fi
    fi
else
    echo "✓ HuggingFace token is set"
fi
echo ""

# Verify model exists locally
if [ ! -d "$LOCAL_MODEL_DIR" ]; then
    echo "WARNING: Model directory not found at: $LOCAL_MODEL_DIR"
    echo ""
    echo "To download the model, run:"
    echo "  hf download meta-llama/Llama-3.3-70B-Instruct --local-dir $LOCAL_MODEL_DIR"
    echo ""
    read -p "Continue anyway (model will be downloaded from HuggingFace)? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# KV events are configured per-worker via --kv-events-config JSON inside the container
# Each worker gets a unique endpoint port: tcp://*:$KV_EVENT_PORT

# Start container with optimized Thompson Sampling components
echo ""
echo "Starting Dynamo container with OPTIMIZED Thompson Sampling components (vLLM)..."
docker run -d \
  --name $CONTAINER_NAME \
  --gpus "\"device=${WORKER_GPUS}\"" \
  --network host \
  --ipc=host \
  --shm-size=$SHM_SIZE \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $LOCAL_MODEL_DIR:$MODEL:ro \
  -v $CUSTOM_DYNAMO_DIR:/workspace/custom_dynamo:ro \
  -v ${SCRIPT_DIR}/monitoring/scripts:/workspace/monitoring/scripts:ro \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
  -e RUST_BACKTRACE=1 \
  -e PYTHONUNBUFFERED=1 \
  -e DYN_HTTP_PORT=$HTTP_PORT \
  -e DYN_ROUTER_MODE=round-robin \
  -e WORKER_METRICS_PORT=$WORKER_METRICS_PORT \
  -e ROUTER_METRICS_PORT=$ROUTER_METRICS_PORT \
  -e PROCESSOR_METRICS_PORT=$PROCESSOR_METRICS_PORT \
  -e KV_BLOCK_SIZE=$KV_BLOCK_SIZE \
  -e GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION \
  -e MAX_NUM_SEQS=$MAX_NUM_SEQS \
  -e ENABLE_KV_EVENTS=$ENABLE_KV_EVENTS \
  -e KV_EVENT_BASE_PORT=$KV_EVENT_BASE_PORT \
  -e DYNAMO_USE_MULTILRU=$DYNAMO_USE_MULTILRU \
  -e DYNAMO_WORKER_COMPONENT=backend \
  $IMAGE \
  bash -c "
    set -e

    echo '========================================================='
    echo 'Verifying external infrastructure services...'
    echo '========================================================='

    # Verify ETCD is accessible
    if curl -s http://localhost:2379/health > /dev/null 2>&1; then
        echo '✓ ETCD accessible at localhost:2379'
    else
        echo '✗ ERROR: ETCD not accessible at localhost:2379'
        exit 1
    fi

    # Verify NATS is accessible
    if timeout 2 bash -c '</dev/tcp/localhost/4222' 2>/dev/null; then
        echo '✓ NATS accessible at localhost:4222'
    else
        echo '✗ ERROR: NATS not accessible at localhost:4222'
        exit 1
    fi

    echo ''

    # Function to wait for worker initialization via ETCD registration
    wait_for_worker() {
        local worker_type=\$1
        local pid=\$2
        # Use WORKER_INIT_TIMEOUT_S (defaults to 1800s / 30 min)
        local max_wait=$WORKER_INIT_TIMEOUT_S
        local elapsed=0
        local poll_interval=5

        echo \"Waiting for \$worker_type worker (PID \$pid) to initialize...\"
        echo \"  Detection: ETCD worker registration\"
        echo \"  Timeout: \${max_wait}s\"

        while [ \$elapsed -lt \$max_wait ]; do
            if ! kill -0 \$pid 2>/dev/null; then
                echo \"ERROR: \$worker_type worker process died!\"
                return 1
            fi

            local etcd_response=\$(curl -s --max-time 2 http://localhost:2379/v3/kv/range \
                -X POST \
                -H \"Content-Type: application/json\" \
                -d '{\"key\":\"AA==\",\"range_end\":\"AA==\",\"keys_only\":true}' 2>&1)

            if [ \$((elapsed % 30)) -eq 0 ] && [ \$elapsed -gt 0 ]; then
                echo \"  [DEBUG] ETCD count: \$(echo \"\$etcd_response\" | grep -o '\"count\":\"[^\"]*\"')\"
            fi

            if echo \"\$etcd_response\" | grep -q '\"count\"' && \
               ! echo \"\$etcd_response\" | grep -q '\"count\":\"0\"'; then
                echo \"✓ \$worker_type worker is ready (registered with ETCD at \${elapsed}s)\"
                return 0
            fi

            sleep \$poll_interval
            elapsed=\$((elapsed + poll_interval))
            if [ \$((elapsed % 30)) -eq 0 ]; then
                echo \"  ... \${elapsed}s / \${max_wait}s (waiting for ETCD registration)\"
            fi
        done

        echo \"ERROR: \$worker_type worker failed to register with ETCD within \${max_wait}s\"
        return 1
    }

    # Function to wait for ALL workers to register with ETCD
    # Counts workers registered at workers.backend.generate endpoint
    wait_for_all_workers() {
        local expected_count=\$1
        local max_wait=$WORKER_INIT_TIMEOUT_S
        local elapsed=0
        local poll_interval=10

        echo \"\"
        echo \"Waiting for ALL \$expected_count vLLM workers to register with ETCD...\"
        echo \"  Detection: Count workers at workers.backend.generate endpoint\"
        echo \"  Timeout: \${max_wait}s\"
        echo \"\"

        while [ \$elapsed -lt \$max_wait ]; do
            # Check all worker PIDs are still alive
            for wpid in \"\${WORKER_PIDS[@]}\"; do
                if ! kill -0 \$wpid 2>/dev/null; then
                    echo \"ERROR: Worker process \$wpid died during initialization!\"
                    return 1
                fi
            done

            # Count worker registrations in ETCD
            # Workers register with keys like: v1/instances/workers/backend/generate/<instance_id>
            local worker_count=\$(curl -s --max-time 2 http://localhost:2379/v3/kv/range \
                -X POST \
                -H \"Content-Type: application/json\" \
                -d '{
                    \"key\": \"'\"djEvaW5zdGFuY2VzL3dvcmtlcnMvYmFja2VuZC9nZW5lcmF0ZS8=\"'\",
                    \"range_end\": \"'\"djEvaW5zdGFuY2VzL3dvcmtlcnMvYmFja2VuZC9nZW5lcmF0ZTA=\"'\",
                    \"count_only\": true
                }' 2>/dev/null | grep -o '\"count\":\"[^\"]*\"' | grep -o '[0-9]*' || echo \"0\")

            if [ \"\$worker_count\" -ge \"\$expected_count\" ]; then
                echo \"✓ All \$expected_count vLLM workers registered with ETCD (took \${elapsed}s)\"
                return 0
            fi

            if [ \$((elapsed % 30)) -eq 0 ]; then
                echo \"  [\${elapsed}s] Workers registered: \$worker_count / \$expected_count\"
            fi

            sleep \$poll_interval
            elapsed=\$((elapsed + poll_interval))
        done

        echo \"ERROR: Only \$worker_count / \$expected_count workers registered within \${max_wait}s\"
        echo \"  Some workers may still be initializing torch.compile (can take 10+ min first time)\"
        return 1
    }

    # =========================================================================
    # STARTUP ORDER WITH MODEL NAME ISOLATION
    # =========================================================================
    # Using different model names to force ALL traffic through the processor.
    # Workers register with internal model name (${SERVED_MODEL_NAME}-internal),
    # while processor registers with public model name (${SERVED_MODEL_NAME}).
    # Frontend only routes to backends matching the requested model name.
    #
    # Order:
    #   1. Workers (model=${SERVED_MODEL_NAME}-internal, not discovered for public model)
    #   2. Router (needs workers to be present)
    #   3. Processor (model=${SERVED_MODEL_NAME}, frontend discovers this)
    #   4. Frontend (routes ${SERVED_MODEL_NAME} requests to processor ONLY)
    # =========================================================================

    echo '========================================================='
    echo 'Step 1: Starting $NUM_WORKERS vLLM Unified Worker(s) (Host GPUs $WORKER_GPUS -> Container GPUs $CONTAINER_GPU_INDICES)...'
    echo '========================================================='
    # Workers register at workers.worker.generate (in 'workers' namespace)
    # They start first so the router can discover them during initialization
    # DYN_SYSTEM_PORT sets the Prometheus metrics port for this component

    # KV events configuration
    # NOTE: KV events are configured via --kv-events-config JSON, not --enable-kv-events flag
    # Each worker gets a unique endpoint port via the config
    # --enable-prefix-caching is a separate vLLM feature (always enabled by default in unified mode)
    if [ \"\$ENABLE_KV_EVENTS\" = \"true\" ]; then
        echo \"KV Events: ENABLED (per-worker ports starting at \$KV_EVENT_BASE_PORT)\"
    else
        echo \"KV Events: DISABLED (set DYNAMO_ENABLE_KV_EVENTS=true to enable)\"
    fi
    
    # Build optional --num-gpu-blocks-override flag (for cache size experiments)
    GPU_BLOCKS_OVERRIDE_OPT=\"\"
    if [ -n \"$NUM_GPU_BLOCKS_OVERRIDE\" ]; then
        GPU_BLOCKS_OVERRIDE_OPT=\"--num-gpu-blocks-override $NUM_GPU_BLOCKS_OVERRIDE\"
        echo \"GPU Blocks Override: $NUM_GPU_BLOCKS_OVERRIDE (experiment mode - limited cache!)\"
    fi

    # Start multiple workers, each using TP_SIZE GPUs
    WORKER_PIDS=()
    for i in \$(seq 0 \$(($NUM_WORKERS - 1))); do
        # Calculate GPU range for this worker (e.g., worker 0: 0,1; worker 1: 2,3; etc.)
        START_GPU=\$((i * $TP_SIZE))
        END_GPU=\$(((i + 1) * $TP_SIZE - 1))
        WORKER_GPU_LIST=\$(seq -s, \$START_GPU \$END_GPU)

        # Calculate port offsets for this worker to avoid ZMQ port conflicts
        # 
        # 1. NIXL Side Channel Ports (for KV transfer handshake)
        #    Each worker's NIXL connector uses TP_SIZE consecutive ports
        #    Port spacing = TP_SIZE (minimum needed to avoid overlap)
        #    Examples:
        #      TP=1, 8 GPUs → 8 workers: 5557, 5558, 5559, 5560, 5561, 5562, 5563, 5564
        #      TP=2, 8 GPUs → 4 workers: 5557-5558, 5559-5560, 5561-5562, 5563-5564
        #      TP=4, 8 GPUs → 2 workers: 5557-5560, 5561-5564
        #      TP=8, 8 GPUs → 1 worker:  5557-5564
        NIXL_BASE_PORT=\$((5557 + i * $TP_SIZE))
        
        # 2. KV Event Publisher Port (for publishing KV cache events to subscriber)
        #    Each worker needs a unique port for its ZMQ publisher
        #    Set via DYN_VLLM_KV_EVENT_PORT environment variable
        #    Default base: 20080, Worker 0: 20080, Worker 1: 20081, etc.
        KV_EVENT_PORT=\$(($KV_EVENT_BASE_PORT + i))
        
        echo \"Starting vLLM Worker \$i: GPUs \$WORKER_GPU_LIST (internal model name)\"
        echo \"  KV Block Size: $KV_BLOCK_SIZE tokens, GPU Mem Util: $GPU_MEMORY_UTILIZATION, Max Seqs: $MAX_NUM_SEQS\"
        echo \"  NIXL Port Range: \$NIXL_BASE_PORT - \$((NIXL_BASE_PORT + $TP_SIZE - 1)) (TP=$TP_SIZE)\"
        echo \"  KV Event Port: \$KV_EVENT_PORT (KV Events: $ENABLE_KV_EVENTS)\"
        # NOTE: dynamo.vllm does NOT accept --host/--port/--endpoint like dynamo.sglang
        # Endpoint is set via DYN_ENDPOINT env var, namespace via DYN_NAMESPACE
        # VLLM_NIXL_SIDE_CHANNEL_PORT sets the base port for NIXL handshake listener
        # DYN_VLLM_KV_EVENT_PORT sets the port for KV event publishing (unique per worker)
        # KV events are configured via --kv-events-config JSON with unique endpoint per worker
        
        # Build KV events config JSON for this worker (unique endpoint per worker)
        KV_EVENTS_JSON=\"{\\\"enable_kv_cache_events\\\":true,\\\"publisher\\\":\\\"zmq\\\",\\\"endpoint\\\":\\\"tcp://*:\$KV_EVENT_PORT\\\"}\"
        
        # Build scheduler class option - use DynamoScheduler for MultiLruBackend if available
        # Set DYNAMO_USE_MULTILRU=false to disable
        SCHEDULER_OPT=\"\"
        if [ \"\${DYNAMO_USE_MULTILRU:-false}\" = \"true\" ]; then
            SCHEDULER_OPT=\"--scheduler-cls kvbm.v2.vllm.schedulers.dynamo.DynamoScheduler\"
            echo \"  Scheduler: DynamoScheduler with MultiLruBackend (frequency-based eviction)\"
        else
            echo \"  Scheduler: Default vLLM scheduler\"
        fi
        
        if [ \"\$ENABLE_KV_EVENTS\" = \"true\" ]; then
            CUDA_VISIBLE_DEVICES=\$WORKER_GPU_LIST \
            DYN_SYSTEM_PORT=\$((WORKER_METRICS_PORT + i)) \
            DYN_NAMESPACE=workers \
            DYN_ENDPOINT=workers.worker.generate \
            VLLM_NIXL_SIDE_CHANNEL_PORT=\$NIXL_BASE_PORT \
            DYN_VLLM_KV_EVENT_PORT=\$KV_EVENT_PORT \
            python3 -m dynamo.vllm \
              --model $MODEL \
              --served-model-name ${SERVED_MODEL_NAME}-internal \
              --tensor-parallel-size $TP_SIZE \
              --trust-remote-code \
              --block-size $KV_BLOCK_SIZE \
              --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
              --max-num-seqs $MAX_NUM_SEQS \
              \$SCHEDULER_OPT \
              \$GPU_BLOCKS_OVERRIDE_OPT \
              --kv-events-config \"\$KV_EVENTS_JSON\" &
        else
            CUDA_VISIBLE_DEVICES=\$WORKER_GPU_LIST \
            DYN_SYSTEM_PORT=\$((WORKER_METRICS_PORT + i)) \
            DYN_NAMESPACE=workers \
            DYN_ENDPOINT=workers.worker.generate \
            VLLM_NIXL_SIDE_CHANNEL_PORT=\$NIXL_BASE_PORT \
            DYN_VLLM_KV_EVENT_PORT=\$KV_EVENT_PORT \
            python3 -m dynamo.vllm \
              --model $MODEL \
              --served-model-name ${SERVED_MODEL_NAME}-internal \
              --tensor-parallel-size $TP_SIZE \
              --trust-remote-code \
              --block-size $KV_BLOCK_SIZE \
              --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
              --max-num-seqs $MAX_NUM_SEQS \
              \$SCHEDULER_OPT \
              \$GPU_BLOCKS_OVERRIDE_OPT &
        fi
        WORKER_PIDS+=(\$!)
        echo \"  Worker \$i PID: \${WORKER_PIDS[\$i]}\"
    done
    echo \"\"
    echo \"Total workers started: \${#WORKER_PIDS[@]}\"
    echo \"Worker PIDs: \${WORKER_PIDS[*]}\"
    echo \"Registered at: workers.worker.generate (model: ${SERVED_MODEL_NAME}-internal)\"
    echo \"NOTE: Workers use internal model name so frontend only discovers processor\"
    echo \"\"

    # Wait for first worker to initialize (checks ETCD registration)
    wait_for_worker \"vLLM Unified\" \${WORKER_PIDS[0]} || exit 1

    # Wait for ALL workers to register with ETCD
    # vLLM workers can take a long time to initialize due to torch.compile
    if [ \${#WORKER_PIDS[@]} -gt 1 ]; then
        wait_for_all_workers \${#WORKER_PIDS[@]} || {
            echo \"WARNING: Not all workers initialized. Continuing with available workers.\"
            echo \"         Dashboard metrics may be incomplete.\"
        }
    fi

    echo ''
    echo '========================================================='
    echo 'Step 2: Starting Custom Router (Thompson Sampling + Prometheus)...'
    echo '========================================================='
    # Router uses config.yaml for all parameters
    # It needs workers to be present (started in Step 1)
    # DYN_SYSTEM_PORT sets the Prometheus metrics port for this component
    DYN_SYSTEM_PORT=\$ROUTER_METRICS_PORT \
    python3 /workspace/custom_dynamo/$ROUTER_SCRIPT \
      --config /workspace/custom_dynamo/config.yaml &
    ROUTER_PID=\$!
    echo \"Router PID: \$ROUTER_PID\"
    echo \"Metrics at: http://localhost:\$ROUTER_METRICS_PORT/metrics\"
    sleep 15
    echo \"\"

    echo ''
    echo '========================================================='
    echo 'Step 3: Starting Custom Processor (Static Mode)...'
    echo '========================================================='
    # STATIC MODE: Processor uses @dynamo_worker(static=True) so it registers
    # at dynamo.backend.generate WITHOUT an instance ID. This is required for
    # --static-endpoint on the frontend to find it.
    # DYN_SYSTEM_PORT sets the Prometheus metrics port for this component
    DYN_SYSTEM_PORT=\$PROCESSOR_METRICS_PORT \
    python3 /workspace/custom_dynamo/$PROCESSOR_SCRIPT \
      --enable-router \
      --model-path $MODEL \
      --model-name $SERVED_MODEL_NAME &
    PROCESSOR_PID=\$!
    echo \"Processor PID: \$PROCESSOR_PID\"
    echo \"Model: $SERVED_MODEL_NAME (from $MODEL)\"
    echo \"Registered at: dynamo.backend.generate (namespace=dynamo)\"
    echo \"Forwards to: workers.worker.generate (actual vLLM workers)\"
    echo \"Metrics at: http://localhost:\$PROCESSOR_METRICS_PORT/metrics\"
    sleep 15
    echo \"\"

    echo ''
    echo '========================================================='
    echo 'Step 4: Starting Default Dynamo Frontend (Namespace-Scoped Discovery)...'
    echo '========================================================='
    # NAMESPACE-SCOPED DISCOVERY: Frontend discovers backends via ETCD ModelWatcher,
    # but only from the 'dynamo' namespace. Workers are in the 'workers' namespace,
    # so the frontend will ONLY discover the processor (in 'dynamo' namespace).
    # This ensures ALL requests go through the Thompson Sampling router.
    echo \"Frontend KV Block Size: $KV_BLOCK_SIZE tokens (must match worker --block-size)\"
    python3 -m dynamo.frontend \
      --http-port $HTTP_PORT \
      --model-name $SERVED_MODEL_NAME \
      --model-path $MODEL \
      --kv-cache-block-size $KV_BLOCK_SIZE \
      --namespace dynamo &
    FRONTEND_PID=\$!
    echo \"Frontend PID: \$FRONTEND_PID\"
    echo \"Discovery: ETCD ModelWatcher (namespace=dynamo, discovers processor ONLY)\"
    sleep 15
    echo \"\"

    echo ''
    echo '========================================================='
    echo '✓ All components started successfully!'
    echo '========================================================='
    echo \"Infrastructure Services (External):\"
    echo \"  ETCD: localhost:2379\"
    echo \"  NATS: localhost:4222\"
    echo \"\"
    echo \"Dynamo Components (This Container):\"
    echo \"  vLLM Unified Workers: \${#WORKER_PIDS[@]} workers (GPUs $WORKER_GPUS, TP=$TP_SIZE each)\"
    for i in \$(seq 0 \$((\${#WORKER_PIDS[@]} - 1))); do
        START_GPU=\$((i * $TP_SIZE))
        END_GPU=\$(((i + 1) * $TP_SIZE - 1))
        echo \"    Worker \$i: PID \${WORKER_PIDS[\$i]}, GPUs \$START_GPU-\$END_GPU\"
    done
    echo \"    → Registered at: workers.worker.generate (hidden from frontend)\"
    echo \"  Router: PID \$ROUTER_PID  (Thompson Sampling + Prometheus)\"
    echo \"    → Registered at: dynamo.router.{find_worker,feedback}\"
    echo \"    → Metrics: http://localhost:\$ROUTER_METRICS_PORT/metrics\"
    echo \"  Processor: PID \$PROCESSOR_PID  (NVExt annotation extraction)\"
    echo \"    → Registered at: dynamo.backend.generate (STATIC mode)\"
    echo \"    → Metrics: http://localhost:\$PROCESSOR_METRICS_PORT/metrics\"
    echo \"  Frontend: PID \$FRONTEND_PID  (Default Dynamo HTTP API on port $HTTP_PORT)\"
    echo \"    → Discovery: ETCD ModelWatcher\"
    echo \"    → Metrics: http://localhost:$HTTP_PORT/metrics\"
    echo ''
    echo 'Request Flow (Dynamic Discovery - Thompson Sampling when routed to processor):'
    echo '  Client → Default Frontend API (port $HTTP_PORT)'
    echo '         ↓ (tokenization + nvext parsing)'
    echo '  Frontend routes via ETCD ModelWatcher (processor OR workers)'
    echo '         ↓'
    echo '  IF routed to Processor (dynamo.backend.generate):'
    echo '         ↓ (extract hints from annotations)'
    echo '         ↓ (query Thompson Sampling router)'
    echo '  Custom Router → worker_id'
    echo '         ↓ (KV overlap + workload-aware selection)'
    echo '  Processor routes to → workers.worker.generate (with worker_id)'
    echo '         ↓'
    echo '  vLLM Unified Worker (workers.worker.generate)'
    echo '         ↓'
    echo '  Response + Feedback to Router'
    echo ''
    echo 'Prometheus Metrics Endpoints:'
    echo '  - Frontend:  http://localhost:$HTTP_PORT/metrics (latency, throughput)'
    echo '  - Workers:   http://localhost:\$WORKER_METRICS_PORT/metrics - \$((WORKER_METRICS_PORT + \${#WORKER_PIDS[@]} - 1))/metrics (KV cache)'
    echo '  - Router:    http://localhost:\$ROUTER_METRICS_PORT/metrics (thompson_router_*)'
    echo '  - Processor: http://localhost:\$PROCESSOR_METRICS_PORT/metrics (thompson_* KVE)'
    echo '========================================================='

    # Monitor all processes
    while true; do
        if ! kill -0 \$FRONTEND_PID 2>/dev/null; then
            echo \"ERROR: Frontend died!\"
            exit 1
        fi
        if ! kill -0 \$PROCESSOR_PID 2>/dev/null; then
            echo \"ERROR: Processor died!\"
            exit 1
        fi
        if ! kill -0 \$ROUTER_PID 2>/dev/null; then
            echo \"ERROR: Router died!\"
            exit 1
        fi
        for i in \$(seq 0 \$((\${#WORKER_PIDS[@]} - 1))); do
            if ! kill -0 \${WORKER_PIDS[\$i]} 2>/dev/null; then
                echo \"ERROR: Worker \$i (PID \${WORKER_PIDS[\$i]}) died!\"
                exit 1
            fi
        done
        sleep 10
    done
  "

# Wait for container to start
echo ""
echo "Waiting for container to start..."
sleep 15

# Check if container started successfully
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo ""
    echo "========================================================="
    echo "✓ Dynamo with OPTIMIZED Thompson Sampling Router Started! (vLLM)"
    echo "========================================================="
    echo ""
    echo "Architecture (Model Name Isolation - Thompson Sampling):"
    echo ""
    echo "  Model Name Isolation Mode:"
    echo "    - Workers register with internal model name (${SERVED_MODEL_NAME}-internal)"
    echo "    - Processor registers with public model name (${SERVED_MODEL_NAME})"
    echo "    - Frontend routes ${SERVED_MODEL_NAME} requests to processor ONLY"
    echo "    - ALL requests go through Thompson Sampling router"
    echo ""
    echo "  Startup Order:"
    echo "    1. Workers     → model=${SERVED_MODEL_NAME}-internal (not matched by frontend)"
    echo "    2. Router      → dynamo.router.{find_worker,feedback}"
    echo "    3. Processor   → model=${SERVED_MODEL_NAME} (matched by frontend)"
    echo "    4. Frontend    → routes to processor for ${SERVED_MODEL_NAME} requests"
    echo ""
    echo "  Request Flow (ALL requests go through processor):"
    echo "    Client Request (with nvext.annotations)"
    echo "      ↓"
    echo "    Default Dynamo Frontend (port $HTTP_PORT)"
    echo "      ↓ ETCD ModelWatcher (namespace=dynamo) routes to processor"
    echo "    Custom Processor (dynamo.backend.generate)"
    echo "      ↓ extracts: prefix_id, total_requests, osl, iat"
    echo "      ↓ queries Thompson Sampling router"
    echo "    Custom Router → worker_id"
    echo "      ↓ KV overlap + workload-aware selection"
    echo "    Processor forwards to workers.worker.generate"
    echo "      ↓"
    echo "    vLLM Unified Workers ($NUM_WORKERS x TP=$TP_SIZE = $NUM_GPUS GPUs total)"
    echo "      ↓"
    echo "    Response + Feedback Loop"
    echo ""
    echo "Infrastructure Services (Managed):"
    echo "  ETCD: etcd-dynamo container, localhost:2379"
    echo "  NATS: nats-dynamo container, localhost:4222"
    echo ""
    echo "Prometheus Metrics Endpoints:"
    echo "  Frontend:  http://localhost:$HTTP_PORT/metrics (latency, throughput)"
    echo "  Workers:   http://localhost:$WORKER_METRICS_PORT/metrics - $((WORKER_METRICS_PORT + NUM_WORKERS - 1))/metrics (KV cache)"
    echo "  Router:    http://localhost:$ROUTER_METRICS_PORT/metrics (routing)"
    echo "  Processor: http://localhost:$PROCESSOR_METRICS_PORT/metrics (KVE)"
    echo ""
    echo "Dynamo Components:"
    echo "  Frontend: HTTP API on port $HTTP_PORT"
    echo "  vLLM Unified Workers: $NUM_WORKERS workers (TP=$TP_SIZE each)"
    echo ""
    echo "KV Cache Settings:"
    echo "  Block Size: $KV_BLOCK_SIZE tokens (DYNAMO_KV_BLOCK_SIZE)"
    echo "  GPU Mem Utilization: $GPU_MEMORY_UTILIZATION (DYNAMO_MEM_FRACTION_STATIC or DYNAMO_GPU_MEMORY_UTILIZATION)"
    echo "  Max Concurrent Seqs: $MAX_NUM_SEQS (DYNAMO_MAX_NUM_SEQS)"
    echo "  KV Events: $ENABLE_KV_EVENTS (DYNAMO_ENABLE_KV_EVENTS)"
    if [ "${DYNAMO_USE_MULTILRU:-false}" = "true" ]; then
        echo "  Scheduler: DynamoScheduler with MultiLruBackend (DYNAMO_USE_MULTILRU=true)"
        echo "    → 4-pool system: Cold→Warm→Hot→VeryHot"
        echo "    → Promotion thresholds: [2, 6, 15] accesses"
    else
        echo "  Scheduler: Default vLLM scheduler (DYNAMO_USE_MULTILRU=false)"
    fi
    echo ""
    echo "API Endpoint: http://localhost:$HTTP_PORT/v1/chat/completions"
    echo "Health Check: http://localhost:$HTTP_PORT/health"
    echo ""
    echo "NVExt Annotations (in request body):"
    echo "  \"nvext\": {"
    echo "    \"annotations\": ["
    echo "      \"prefix_id:<unique_id>\","
    echo "      \"total_requests:<number>\","
    echo "      \"osl:LOW|MEDIUM|HIGH\","
    echo "      \"iat:LOW|MEDIUM|HIGH\""
    echo "    ]"
    echo "  }"
    echo ""
    echo "Monitoring Dashboards:"
    echo "  Grafana:    http://localhost:3000 (no login required)"
    echo "  Prometheus: http://localhost:9090"
    echo ""
    echo "Useful Commands:"
    echo "  Interactive shell:    docker exec -it $CONTAINER_NAME bash"
    echo "  View Dynamo logs:     docker logs -f $CONTAINER_NAME"
    echo "  View ETCD logs:       docker logs -f etcd-dynamo"
    echo "  View NATS logs:       docker logs -f nats-dynamo"
    echo "  GPU usage:            watch -n 2 nvidia-smi"
    echo "  Stop all:             bash stop_dynamo.sh"
    echo "  Stop all + metrics:   bash stop_dynamo.sh --kill-metrics"
    echo ""
    echo "Query Metrics (vLLM uses 'vllm:' prefix):"
    echo "  curl http://localhost:$HTTP_PORT/metrics | grep dynamo_frontend"
    echo "  curl http://localhost:$WORKER_METRICS_PORT/metrics | grep vllm:"
    echo "  curl http://localhost:$ROUTER_METRICS_PORT/metrics | grep thompson_router"
    echo "  curl http://localhost:$PROCESSOR_METRICS_PORT/metrics | grep thompson_kve"
    echo ""
    echo "========================================================="
    echo "Test Request (with nvext annotations):"
    echo "========================================================="
    echo ""
    echo "# Basic test (no hints)"
    echo "curl http://localhost:$HTTP_PORT/v1/chat/completions \\"
    echo "  -H 'Content-Type: application/json' \\"
    echo "  -d '{"
    echo "    \"model\": \"$SERVED_MODEL_NAME\","
    echo "    \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}],"
    echo "    \"max_tokens\": 50"
    echo "  }'"
    echo ""
    echo "# Test with nvext annotations (routing hints)"
    echo "curl http://localhost:$HTTP_PORT/v1/chat/completions \\"
    echo "  -H 'Content-Type: application/json' \\"
    echo "  -d '{"
    echo "    \"model\": \"$SERVED_MODEL_NAME\","
    echo "    \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}],"
    echo "    \"max_tokens\": 50,"
    echo "    \"nvext\": {"
    echo "      \"annotations\": ["
    echo "        \"prefix_id:test-session-001\","
    echo "        \"total_requests:5\","
    echo "        \"osl:MEDIUM\","
    echo "        \"iat:LOW\""
    echo "      ]"
    echo "    }"
    echo "  }'"
    echo ""
    echo "# Streaming test with hints"
    echo "curl http://localhost:$HTTP_PORT/v1/chat/completions \\"
    echo "  -H 'Content-Type: application/json' \\"
    echo "  -d '{"
    echo "    \"model\": \"$SERVED_MODEL_NAME\","
    echo "    \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}],"
    echo "    \"max_tokens\": 50,"
    echo "    \"stream\": true,"
    echo "    \"nvext\": {"
    echo "      \"annotations\": [\"prefix_id:stream-test\", \"total_requests:1\"]"
    echo "    }"
    echo "  }'"
    echo ""
    echo "========================================================="
    echo ""
    echo "Waiting for vLLM to initialize (this may take 5-10 minutes for a 70B model)..."
    echo "Monitoring logs (Ctrl+C to exit, container continues)..."
    echo ""

    # Wait for server to be ready
    echo "Checking for API availability (timeout=${WORKER_INIT_TIMEOUT_S}s)..."
    max_attempts=$WORKER_INIT_TIMEOUT_S
    attempt=0

    while [ $attempt -lt $max_attempts ]; do
        # Use || true to prevent curl connection failures from exiting due to set -e
        # curl returns "000" for connection refused, so we just need to prevent the exit
        health_response=$(curl -s --max-time 5 -o /dev/null -w "%{http_code}" http://localhost:$HTTP_PORT/health 2>/dev/null) || true
        if [ "$health_response" = "200" ]; then
            echo "✓ Dynamo API is ready! (health check passed)"
            break
        fi
        attempt=$((attempt + 1))
        if [ $((attempt % 15)) -eq 0 ]; then
            echo "  ... still waiting ($attempt/$max_attempts) - health response: $health_response"
        fi
        sleep 1
    done

    if [ $attempt -ge $max_attempts ]; then
        echo ""
        echo "⚠ Timeout waiting for API. Check logs with: docker logs $CONTAINER_NAME"
        echo ""
    else
        echo ""
        echo "Quick test (polling every 15s for up to 5 minutes):"
        echo ""
        
        quick_test_max_attempts=20  # 20 * 15s = 5 minutes
        quick_test_attempt=0
        quick_test_success=false
        
        while [ $quick_test_attempt -lt $quick_test_max_attempts ]; do
            quick_test_attempt=$((quick_test_attempt + 1))
            echo "  Attempt $quick_test_attempt/$quick_test_max_attempts..."
            
            quick_test_response=$(curl -s --max-time 60 http://localhost:$HTTP_PORT/v1/chat/completions \
              -H "Content-Type: application/json" \
              -d '{
                "model": "'$SERVED_MODEL_NAME'",
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 20
              }' 2>&1) || true
            
            # Check if response is empty/null
            if [ -z "$quick_test_response" ]; then
                echo "    Empty response, retrying in 15s..."
                sleep 15
                continue
            fi
            
            # Check if response contains an error
            error_message=$(echo "$quick_test_response" | jq -r '.error.message // .error // empty' 2>/dev/null)
            if [ -n "$error_message" ]; then
                echo ""
                echo "========================================================="
                echo "✗ Quick test failed with error:"
                echo "  $error_message"
                echo "========================================================="
                echo ""
                echo "Full response:"
                echo "$quick_test_response" | jq . 2>/dev/null || echo "$quick_test_response"
                echo ""
                echo "Check logs with: docker logs $CONTAINER_NAME"
                exit 1
            fi
            
            # Check if response has valid choices (success)
            choices_content=$(echo "$quick_test_response" | jq -r '.choices[0].message.content // empty' 2>/dev/null)
            if [ -n "$choices_content" ]; then
                echo ""
                echo "========================================================="
                echo "✓ Quick test successful!"
                echo "========================================================="
                echo ""
                echo "$quick_test_response" | jq '.choices[0].message.content, .usage'
                echo ""
                echo "========================================================="
                echo "Container is running. View logs with:"
                echo "  docker logs -f $CONTAINER_NAME"
                echo "========================================================="
                quick_test_success=true
                break
            fi
            
            # Response exists but no choices - might still be loading
            echo "    Response received but no valid choices, retrying in 15s..."
            echo "    Response: $(echo "$quick_test_response" | head -c 200)..."
            sleep 15
        done
        
        if [ "$quick_test_success" = false ]; then
            echo ""
            echo "========================================================="
            echo "⚠ Quick test timed out after 5 minutes"
            echo "========================================================="
            echo ""
            echo "Container is running but may not be fully ready."
            echo "Try manually: curl http://localhost:$HTTP_PORT/v1/chat/completions ..."
            echo "Check logs with: docker logs $CONTAINER_NAME"
        fi
    fi
else
    echo ""
    echo "========================================================="
    echo "✗ Container failed to start!"
    echo "========================================================="
    echo ""
    echo "Check logs with: docker logs $CONTAINER_NAME"
    exit 1
fi

