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

# Dynamo SGLang FULL STACK with Thompson Sampling Router & Prefix Hints
# Architecture: ETCD + NATS + Custom Frontend/Router/Processor → SGLang Backend Worker (Unified)
#
# This script manages ALL required components:
#   - ETCD (metadata and worker discovery)
#   - NATS (message queue for requests)
#   - Custom Dynamo Frontend with prefix hints support
#   - Custom Router with Thompson Sampling (LinTS + Beta bandits)
#   - Custom Processor with workload-aware routing
#   - Unified Worker (GPUs 0,1,2,3, TP=4, no disaggregation)
#
# Frontend: Port 8099 (HTTP API with prefix hint headers)
# ETCD: localhost:2379 (container: etcd-dynamo)
# NATS: localhost:4222 (container: nats-dynamo)
# Worker runs in container: dynamo-sglang
#
# Custom Components Location: external/dynamo/generalized/
#   - frontend.py: Accepts x-prefix-* headers, tool-call parsing
#   - processor.py: Forwards hints to router, CSV metrics logging
#   - router.py: Thompson Sampling, KV overlap, workload-aware routing
#
# To stop all components: bash stop_dynamo.sh

# Configuration Variables (can be overridden via environment variables)
CONTAINER_NAME="dynamo-sglang"
WORKER_GPUS="${DYNAMO_GPU_DEVICES:-0,1,2,3}"
TP_SIZE="${DYNAMO_TP_SIZE:-4}"
HTTP_PORT="${DYNAMO_HTTP_PORT:-8099}"
MODEL="/workspace/models/Llama-3.3-70B-Instruct"
SERVED_MODEL_NAME="${DYNAMO_MODEL_NAME:-llama-3.3-70b}"
IMAGE="nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.7.1"
SHM_SIZE="${DYNAMO_SHM_SIZE:-16g}"
WORKER_INIT_TIMEOUT_S="${DYNAMO_WORKER_INIT_TIMEOUT_S:-600}"

# Local paths - DYNAMO_MODEL_DIR must be set or script will error
if [ -z "${DYNAMO_MODEL_DIR}" ]; then
    echo "ERROR: DYNAMO_MODEL_DIR environment variable must be set"
    echo ""
    echo "Example:"
    echo "  export DYNAMO_MODEL_DIR=\"/path/to/your/models/Llama-3.3-70B-Instruct\""
    echo ""
    echo "Then run this script again."
    exit 1
fi
# If directory exists, validate it's a proper model directory (NVBug 5756833)
# If it doesn't exist, the download workflow later will handle it
if [ -d "${DYNAMO_MODEL_DIR}" ]; then
    if [ ! -f "${DYNAMO_MODEL_DIR}/config.json" ]; then
        echo "ERROR: ${DYNAMO_MODEL_DIR} exists but is not a valid model directory"
        echo ""
        echo "Missing: config.json"
        echo ""
        echo "Common mistake - pointing to cache root instead of model snapshot:"
        echo "   Wrong: ~/.cache/huggingface/"
        echo "   Right: ~/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/<hash>"
        echo ""
        echo "Find it: find ~/.cache/huggingface/hub -name config.json -path '*Llama-3.3-70B*'"
        exit 1
    fi

    # Verify config.json has model_type field (exact error from NVBug 5756833)
    if ! grep -q '"model_type"' "${DYNAMO_MODEL_DIR}/config.json" 2>/dev/null; then
        echo "ERROR: ${DYNAMO_MODEL_DIR}/config.json is missing 'model_type' field"
        echo ""
        echo "This usually means incomplete/corrupted download. Try:"
        echo "  rm -rf ${DYNAMO_MODEL_DIR}"
        echo "  huggingface-cli download meta-llama/Llama-3.3-70B-Instruct --local-dir ${DYNAMO_MODEL_DIR}"
        exit 1
    fi
fi
LOCAL_MODEL_DIR="${DYNAMO_MODEL_DIR}"

# Repository directory - auto-detect from script location or use env var
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# LOCAL_REPO_DIR="${DYNAMO_REPO_DIR:-$SCRIPT_DIR}"
# Custom dynamo components are always relative to the script location (external/dynamo/generalized)
CUSTOM_DYNAMO_DIR="${SCRIPT_DIR}/generalized"

echo "========================================================="
echo "Dynamo SGLang with Thompson Sampling Router (UNIFIED)"
echo "========================================================="
echo "Model: Llama-3.3-70B-Instruct"
echo "Container: $CONTAINER_NAME"
echo "HTTP Port: $HTTP_PORT"
echo ""
echo "Custom Components:"
echo "  - Frontend: Prefix hints (x-prefix-id, x-prefix-total-requests, etc.)"
echo "  - Router: Thompson Sampling (LinTS + Beta bandits)"
echo "  - Processor: Workload-aware routing with OSL/IAT hints"
echo ""
echo "Components:"
echo "  - ETCD (metadata and discovery)"
echo "  - NATS (message queue for requests)"
echo "  - Custom Frontend (HTTP API on port $HTTP_PORT)"
echo "  - Custom Router (KV overlap + Thompson Sampling)"
echo "  - Custom Processor (hint forwarding + metrics)"
echo "  - SGLang Worker (unified mode)"
echo ""
echo "Backend Worker:"
echo "  Unified: GPUs $WORKER_GPUS (TP=$TP_SIZE)"
echo "  Mode: UNIFIED (no prefill/decode disaggregation)"
echo ""
echo "========================================================="

# Verify custom components exist
if [ ! -f "$CUSTOM_DYNAMO_DIR/frontend.py" ]; then
    echo "✗ ERROR: Custom frontend.py not found at: $CUSTOM_DYNAMO_DIR/frontend.py"
    exit 1
fi
if [ ! -f "$CUSTOM_DYNAMO_DIR/router.py" ]; then
    echo "✗ ERROR: Custom router.py not found at: $CUSTOM_DYNAMO_DIR/router.py"
    exit 1
fi
if [ ! -f "$CUSTOM_DYNAMO_DIR/processor.py" ]; then
    echo "✗ ERROR: Custom processor.py not found at: $CUSTOM_DYNAMO_DIR/processor.py"
    exit 1
fi
echo "✓ Custom components found in: $CUSTOM_DYNAMO_DIR"
echo ""

# Start ETCD if not running
if docker ps -a --format '{{.Names}}' | grep -q "^etcd-dynamo$"; then
    # Container exists (running or stopped), remove it first
    echo ""
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
        sleep 2  # Extra settling time
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
    # Container exists (running or stopped), remove it first
    echo ""
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

# Clean up existing Dynamo container if it exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Removing existing Dynamo container: $CONTAINER_NAME"
    docker rm -f $CONTAINER_NAME
fi

# Verify HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "⚠ HF_TOKEN environment variable is not set."
    echo ""
    echo "The model is cached locally at: $LOCAL_MODEL_DIR"
    if [ -d "$LOCAL_MODEL_DIR" ]; then
        echo "✓ Local model found - proceeding without HF_TOKEN"
        echo "  Note: Set HF_TOKEN if you need to download models from HuggingFace"
        HF_TOKEN="dummy"  # Set dummy token since model is cached
    else
        echo "✗ Local model NOT found and no HF_TOKEN to download it"
        echo ""
        read -p "Please enter your HuggingFace token (or press Enter to skip): " HF_TOKEN

        if [ -z "$HF_TOKEN" ]; then
            echo ""
            echo "WARNING: Proceeding without HF_TOKEN. This may fail if the model needs to be downloaded."
            echo "To set HF_TOKEN: export HF_TOKEN='your_token_here'"
            HF_TOKEN="dummy"
        else
            echo ""
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
    echo "  huggingface-cli download meta-llama/Llama-3.3-70B-Instruct --local-dir $LOCAL_MODEL_DIR"
    echo ""
    read -p "Continue anyway (model will be downloaded from HuggingFace)? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start container with unified SGLang worker + custom Dynamo components
echo ""
echo "Starting Dynamo container with custom Thompson Sampling components..."
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
  -e HF_TOKEN="$HF_TOKEN" \
  -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
  -e RUST_BACKTRACE=1 \
  -e PYTHONUNBUFFERED=1 \
  -e FRONTEND_MODEL_MAPPING="{\"$SERVED_MODEL_NAME\": \"$MODEL\"}" \
  -e FRONTEND_TPS_INTERVAL=5 \
  -e FRONTEND_TPS_CSV=/workspace/metrics/frontend_throughput.csv \
  -e PROCESSOR_METRICS_CSV=/workspace/metrics/processor_requests.csv \
  -e PROCESSOR_METRICS_MAX_ROWS=2048 \
  -e ROUTER_METRICS_CSV=/workspace/metrics/router_metrics.csv \
  $IMAGE \
  bash -c "
    set -e  # Exit on any error

    # Create metrics directory
    mkdir -p /workspace/metrics

    echo '========================================================='
    echo 'Verifying external infrastructure services...'
    echo '========================================================='

    # Verify ETCD is accessible (basic HTTP check)
    if curl -s http://localhost:2379/health > /dev/null 2>&1; then
        echo '✓ ETCD accessible at localhost:2379'
    else
        echo '✗ ERROR: ETCD not accessible at localhost:2379'
        echo '  Make sure ETCD container is running with --network host'
        exit 1
    fi

    # Verify NATS is accessible (basic TCP check)
    if timeout 2 bash -c '</dev/tcp/localhost/4222' 2>/dev/null; then
        echo '✓ NATS accessible at localhost:4222'
    else
        echo '✗ ERROR: NATS not accessible at localhost:4222'
        echo '  Make sure NATS container is running with --network host'
        exit 1
    fi

    echo ''

    # Function to wait for worker initialization by checking ETCD registration
    # Dynamo workers register with ETCD, they don't expose HTTP health endpoints
    wait_for_worker() {
        local worker_type=\$1
        local pid=\$2
        # local max_wait=300
        local max_wait=${WORKER_INIT_TIMEOUT_S:-600}
        local elapsed=0
        local poll_interval=5

        echo \"Waiting for \$worker_type worker (PID \$pid) to initialize...\"
        echo \"  Detection: ETCD worker registration\"
        echo \"  Timeout: \${max_wait}s\"

        while [ \$elapsed -lt \$max_wait ]; do
            # Check if process is still running
            if ! kill -0 \$pid 2>/dev/null; then
                echo \"ERROR: \$worker_type worker process died!\"
                return 1
            fi

            # Check ETCD for registered workers using v3 API
            # Query ALL keys to find where Dynamo registers (empty key "" with range_end "\0" = all keys)
            # Base64: "" -> AA==, "\0" -> AA==  (we use keys_only to reduce response size)
            local etcd_response=\$(curl -s --max-time 2 http://localhost:2379/v3/kv/range \
                -X POST \
                -H \"Content-Type: application/json\" \
                -d '{\"key\":\"AA==\",\"range_end\":\"AA==\",\"keys_only\":true}' 2>&1)

            # Debug: Print ETCD response every 30s (truncated)
            if [ \$((elapsed % 30)) -eq 0 ] && [ \$elapsed -gt 0 ]; then
                echo \"  [DEBUG] ETCD keys found: \$(echo \"\$etcd_response\" | grep -o '\"key\":\"[^\"]*\"' | head -5)\"
                echo \"  [DEBUG] ETCD count: \$(echo \"\$etcd_response\" | grep -o '\"count\":\"[^\"]*\"')\"
            fi

            # Check if we got any keys back (count > 0 means workers registered)
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
        echo \"  Image: $IMAGE\"
        echo \"  The model may require more time to load, or there may be a startup error.\"
        echo \"  Check worker logs for details.\"
        return 1
    }

    echo '========================================================='
    echo 'Step 1: Starting Unified Worker (GPUs 0,1,2,3 = Host GPUs $WORKER_GPUS)...'
    echo '========================================================='
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python3 -m dynamo.sglang \
      --model-path $MODEL \
      --served-model-name $SERVED_MODEL_NAME \
      --host 0.0.0.0 \
      --port 30000 \
      --tp $TP_SIZE \
      --trust-remote-code \
      --enable-metrics \
      --mem-fraction-static 0.8 &
    WORKER_PID=\$!
    echo \"Unified Worker PID: \$WORKER_PID\"
    echo \"\"

    # Wait for unified worker to initialize (checks ETCD registration)
    wait_for_worker \"Unified\" \$WORKER_PID || exit 1

    echo ''
    echo '========================================================='
    echo 'Step 2: Starting Custom Router (Thompson Sampling + KV Overlap)...'
    echo '========================================================='
    python3 /workspace/custom_dynamo/router.py \
      --block-size 64 \
      --router-type kv \
      --affinity-base 0.30 \
      --affinity-reuse-weight 0.15 \
      --affinity-iat-weight 0.20 \
      --base-ts-weight 0.10 \
      --sticky-load-floor 0.70 \
      --temp-base 1.0 \
      --temp-min 0.15 \
      --temp-max 2.0 \
      --switch-cost-base 0.20 \
      --switch-cost-reuse 0.08 \
      --switch-cost-iat 0.05 \
      --queue-penalty-weight 0.50 \
      --gpu-penalty-weight 1.00 \
      --outstanding-work-weight 0.45 \
      --job-gpu-coupling-weight 0.40 \
      --job-queue-coupling-weight 0.20 \
      --prefill-token-scale 1024.0 \
      --prefill-weight 1.0 \
      --lints-lambda 1.0 \
      --lints-v 0.25 \
      --lints-forget 0.995 \
      --feedback-timeout-seconds 120.0 \
      --pending-sweep-interval-seconds 5.0 \
      --timeout-reward 0.0 \
      --latency-ema-alpha 0.2 &
    ROUTER_PID=\$!
    echo \"Router PID: \$ROUTER_PID\"
    echo \"Router will wait for backend workers internally via wait_for_instances()...\"
    sleep 15  # Brief pause to let router start
    echo \"\"

    echo ''
    echo '========================================================='
    echo 'Step 3: Starting Custom Processor (Workload-Aware)...'
    echo '========================================================='
    python3 /workspace/custom_dynamo/processor.py \
      --model $MODEL \
      --enable-router &
    PROCESSOR_PID=\$!
    echo \"Processor PID: \$PROCESSOR_PID\"
    echo \"Processor will wait for router and backend internally via wait_for_instances()...\"
    sleep 15  # Brief pause to let processor start
    echo \"\"

    echo ''
    echo '========================================================='
    echo 'Step 4: Starting Custom Frontend (Prefix Hints Support)...'
    echo '========================================================='
    python3 /workspace/custom_dynamo/frontend.py &
    FRONTEND_PID=\$!
    echo \"Frontend PID: \$FRONTEND_PID\"
    echo \"Frontend will wait for processor internally via wait_for_instances()...\"
    sleep 15  # Brief pause to let frontend start
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
    echo \"  Unified Worker: PID \$WORKER_PID  (GPUs $WORKER_GPUS, TP=$TP_SIZE, internal port 30000)\"
    echo \"  Router: PID \$ROUTER_PID  (Thompson Sampling + KV overlap)\"
    echo \"  Processor: PID \$PROCESSOR_PID  (Workload-aware routing)\"
    echo \"  Frontend: PID \$FRONTEND_PID  (HTTP API on port $HTTP_PORT)\"
    echo ''
    echo 'Request Flow:'
    echo '  Client → Frontend API (port $HTTP_PORT, accepts x-prefix-* headers)'
    echo '         ↓'
    echo '  Frontend parses prefix hints (ID, total requests, OSL, IAT)'
    echo '         ↓'
    echo '  Processor forwards hints to Router'
    echo '         ↓'
    echo '  Router (Thompson Sampling) selects worker based on:'
    echo '    - KV cache overlap'
    echo '    - Prefix reuse budget'
    echo '    - OSL/IAT hints'
    echo '    - Worker load (GPU, queue)'
    echo '    - LinTS contextual features'
    echo '         ↓'
    echo '  Unified Worker executes request'
    echo '         ↓'
    echo '  Processor sends feedback (latency) to Router'
    echo '         ↓'
    echo '  Router updates bandits (LinTS + Beta)'
    echo '         ↓'
    echo '  Response'
    echo ''
    echo 'Metrics CSV Files:'
    echo '  - /workspace/metrics/frontend_throughput.csv'
    echo '  - /workspace/metrics/processor_requests.csv'
    echo '  - /workspace/metrics/router_metrics.csv'
    echo '========================================================='

    # Monitor all processes
    while true; do
        # Check if any critical process died
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
        if ! kill -0 \$WORKER_PID 2>/dev/null; then
            echo \"ERROR: Unified worker died!\"
            exit 1
        fi
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
    echo "✓ Dynamo with Thompson Sampling Router Started!"
    echo "========================================================="
    echo ""
    echo "Architecture:"
    echo "  Client Request (with optional x-prefix-* headers)"
    echo "    ↓"
    echo "  Custom Frontend (port $HTTP_PORT)"
    echo "    ↓ (prefix hints: ID, total, OSL, IAT)"
    echo "  Custom Processor"
    echo "    ↓"
    echo "  Custom Router (Thompson Sampling)"
    echo "    ↓ (KV overlap + workload-aware selection)"
    echo "  Unified Worker (GPUs $WORKER_GPUS, TP=$TP_SIZE)"
    echo "    ↓"
    echo "  Response + Feedback Loop"
    echo ""
    echo "Infrastructure Services (Managed):"
    echo "  ETCD: etcd-dynamo container, localhost:2379"
    echo "  NATS: nats-dynamo container, localhost:4222"
    echo ""
    echo "Dynamo Components (This Container):"
    echo "  Frontend: HTTP API on port $HTTP_PORT"
    echo "  Router: Thompson Sampling (LinTS + Beta bandits)"
    echo "  Processor: Workload-aware routing"
    echo "  Unified Worker: GPUs $WORKER_GPUS (TP=$TP_SIZE, internal port 30000)"
    echo ""
    echo "API Endpoint: http://localhost:$HTTP_PORT/v1/chat/completions"
    echo "Health Check: http://localhost:$HTTP_PORT/health"
    echo ""
    echo "Prefix Hint Headers (optional):"
    echo "  x-prefix-id: <unique_prefix_id>"
    echo "  x-prefix-total-requests: <number>"
    echo "  x-prefix-osl: LOW|MEDIUM|HIGH"
    echo "  x-prefix-iat: LOW|MEDIUM|HIGH"
    echo ""
    echo "Useful Commands:"
    echo "  Interactive shell:    docker exec -it $CONTAINER_NAME bash"
    echo "  View Dynamo logs:     docker logs -f $CONTAINER_NAME"
    echo "  View ETCD logs:       docker logs -f etcd-dynamo"
    echo "  View NATS logs:       docker logs -f nats-dynamo"
    echo "  GPU usage:            watch -n 2 nvidia-smi"
    echo "  Stop all:             bash stop_dynamo.sh"
    echo ""
    echo "Metrics Access (from host):"
    echo "  docker exec $CONTAINER_NAME cat /workspace/metrics/frontend_throughput.csv"
    echo "  docker exec $CONTAINER_NAME cat /workspace/metrics/processor_requests.csv"
    echo "  docker exec $CONTAINER_NAME cat /workspace/metrics/router_metrics.csv"
    echo ""
    echo "========================================================="
    echo "Test Request:"
    echo "========================================================="
    echo ""
    echo "# Basic test (no prefix hints)"
    echo "curl http://localhost:$HTTP_PORT/v1/chat/completions \\"
    echo "  -H 'Content-Type: application/json' \\"
    echo "  -d '{"
    echo "    \"model\": \"$SERVED_MODEL_NAME\","
    echo "    \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}],"
    echo "    \"max_tokens\": 50"
    echo "  }'"
    echo ""
    echo "# Test with prefix hints"
    echo "curl http://localhost:$HTTP_PORT/v1/chat/completions \\"
    echo "  -H 'Content-Type: application/json' \\"
    echo "  -H 'x-prefix-id: test-prefix-001' \\"
    echo "  -H 'x-prefix-total-requests: 5' \\"
    echo "  -H 'x-prefix-osl: MEDIUM' \\"
    echo "  -H 'x-prefix-iat: LOW' \\"
    echo "  -d '{"
    echo "    \"model\": \"$SERVED_MODEL_NAME\","
    echo "    \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}],"
    echo "    \"max_tokens\": 50"
    echo "  }'"
    echo ""
    echo "# Streaming test"
    echo "curl http://localhost:$HTTP_PORT/v1/chat/completions \\"
    echo "  -H 'Content-Type: application/json' \\"
    echo "  -d '{"
    echo "    \"model\": \"$SERVED_MODEL_NAME\","
    echo "    \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}],"
    echo "    \"max_tokens\": 50,"
    echo "    \"stream\": true"
    echo "  }'"
    echo ""
    echo "========================================================="
    echo ""
    echo "Waiting for SGLang to initialize (this will likely take 5-10 minutes for a 70B model)..."
    echo "Monitoring logs (Ctrl+C to exit, container continues)..."
    echo ""

    # Wait for server to be ready (check /health for custom frontend)
    # Note: Custom frontend doesn't implement /v1/models, so we use /health
    # Worker registration is already confirmed via ETCD check above
    echo "Checking for API availability (timeout=15 minutes)..."
    max_attempts=900
    attempt=0

    while [ $attempt -lt $max_attempts ]; do
        # Check /health - custom frontend health endpoint
        health_response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$HTTP_PORT/health 2>/dev/null)
        if [ "$health_response" = "200" ]; then
            echo "✓ Dynamo API is ready! (frontend health check passed)"
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
        echo "Quick test:"
        echo ""
        curl -s http://localhost:$HTTP_PORT/v1/chat/completions \
          -H "Content-Type: application/json" \
          -d '{
            "model": "'$SERVED_MODEL_NAME'",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 20
          }' | jq '.choices[0].message.content, .usage'

        echo ""
        echo "========================================================="
        echo "Container is running. View logs with:"
        echo "  docker logs -f $CONTAINER_NAME"
        echo "========================================================="
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
