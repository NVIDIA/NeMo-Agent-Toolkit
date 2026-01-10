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

# Dynamo SGLang FULL STACK with Unified Worker
# Architecture: ETCD + NATS + Dynamo Frontend (API) → SGLang Backend Worker (Unified)
#
# This script manages ALL required components:
#   - ETCD (metadata and worker discovery)
#   - NATS (message queue for requests)
#   - Dynamo Frontend (HTTP API with built-in processor + router)
#   - Unified Worker (GPUs 0,1,2,3, TP=4, no disaggregation)
#
# Frontend: Port 8099 (HTTP API)
# ETCD: localhost:2379 (container: etcd-dynamo) - default port, override with DYNAMO_ETCD_PORT
# NATS: localhost:4222 (container: nats-dynamo) - default port, override with DYNAMO_NATS_PORT
# Worker runs in container: dynamo-sglang
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

# Infrastructure ports (can be overridden via environment variables)
ETCD_CLIENT_PORT="${DYNAMO_ETCD_PORT:-2379}"
ETCD_PEER_PORT="${DYNAMO_ETCD_PEER_PORT:-2390}"
NATS_PORT="${DYNAMO_NATS_PORT:-4222}"
WORKER_INIT_TIMEOUT_S="${DYNAMO_WORKER_INIT_TIMEOUT_S:-600}"

# Compute container-internal GPU indices (GPUs are renumbered 0,1,2,... inside the container)
NUM_GPUS=$(echo "$WORKER_GPUS" | tr ',' '\n' | wc -l)
CONTAINER_GPU_INDICES=$(seq -s, 0 $((NUM_GPUS - 1)))

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

echo "========================================================="
echo "Dynamo SGLang FULL STACK (UNIFIED MODE)"
echo "========================================================="
echo "Model: Llama-3.3-70B-Instruct"
echo "Container: $CONTAINER_NAME"
echo "HTTP Port: $HTTP_PORT"
echo ""
echo "Components:"
echo "  - ETCD (metadata and discovery)"
echo "  - NATS (message queue for requests)"
echo "  - Dynamo Frontend (HTTP API on port $HTTP_PORT)"
echo "  - SGLang Worker (unified mode)"
echo ""
echo "Backend Worker:"
echo "  Unified: GPUs $WORKER_GPUS (TP=$TP_SIZE)"
echo "  Mode: UNIFIED (no prefill/decode disaggregation)"
echo ""
echo "========================================================="

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
  -e ETCD_LISTEN_CLIENT_URLS=http://0.0.0.0:$ETCD_CLIENT_PORT \
  -e ETCD_ADVERTISE_CLIENT_URLS=http://localhost:$ETCD_CLIENT_PORT \
  -e ETCD_LISTEN_PEER_URLS=http://0.0.0.0:$ETCD_PEER_PORT \
  -e ETCD_INITIAL_ADVERTISE_PEER_URLS=http://localhost:$ETCD_PEER_PORT \
  -e ETCD_INITIAL_CLUSTER=default=http://localhost:$ETCD_PEER_PORT \
  bitnamilegacy/etcd:3.6.1

# Wait for ETCD to be ready
echo "Waiting for ETCD to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:$ETCD_CLIENT_PORT/health > /dev/null 2>&1; then
        echo "✓ ETCD is ready on port $ETCD_CLIENT_PORT"
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
  -js -p $NATS_PORT

# Wait for NATS to be ready
echo "Waiting for NATS to be ready..."
for i in {1..30}; do
    if timeout 2 bash -c "cat < /dev/null > /dev/tcp/localhost/$NATS_PORT" 2>/dev/null; then
        echo "✓ NATS is ready on port $NATS_PORT"
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

# Start container with unified SGLang worker + Dynamo frontend
echo ""
echo "Starting Dynamo container with unified SGLang worker + frontend..."
docker run -d \
  --name $CONTAINER_NAME \
  --gpus "\"device=${WORKER_GPUS}\"" \
  --network host \
  --ipc=host \
  --shm-size=$SHM_SIZE \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $LOCAL_MODEL_DIR:$MODEL:ro \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
  -e RUST_BACKTRACE=1 \
  -e PYTHONUNBUFFERED=1 \
  -e ETCD_ENDPOINTS=http://localhost:$ETCD_CLIENT_PORT \
  -e NATS_SERVER=nats://localhost:$NATS_PORT \
  $IMAGE \
  bash -c "
    set -e  # Exit on any error

    echo '========================================================='
    echo 'Verifying external infrastructure services...'
    echo '========================================================='

    # Verify ETCD is accessible
    if curl -s http://localhost:$ETCD_CLIENT_PORT/health > /dev/null 2>&1; then
        echo \"✓ ETCD accessible at localhost:$ETCD_CLIENT_PORT\"
    else
        echo \"✗ ERROR: ETCD not accessible at localhost:$ETCD_CLIENT_PORT\"
        echo '  Make sure ETCD container is running with --network host'
        exit 1
    fi

    # Verify NATS is accessible (basic TCP check)
    if timeout 2 bash -c '</dev/tcp/localhost/$NATS_PORT' 2>/dev/null; then
        echo \"✓ NATS accessible at localhost:$NATS_PORT\"
    else
        echo \"✗ ERROR: NATS not accessible at localhost:$NATS_PORT\"
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
        local max_wait=${DYNAMO_WORKER_INIT_TIMEOUT_S:-600}
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
            local etcd_response=\$(curl -s --max-time 2 http://localhost:$ETCD_CLIENT_PORT/v3/kv/range \
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
    echo 'Step 1: Starting Unified Worker (Host GPUs $WORKER_GPUS -> Container GPUs $CONTAINER_GPU_INDICES)...'
    echo '========================================================='
    CUDA_VISIBLE_DEVICES=$CONTAINER_GPU_INDICES \
    python3 -m dynamo.sglang \
      --model-path $MODEL \
      --served-model-name $SERVED_MODEL_NAME \
      --host 0.0.0.0 \
      --port 30000 \
      --tp $TP_SIZE \
      --trust-remote-code \
      --mem-fraction-static 0.8 &
    WORKER_PID=\$!
    echo \"Unified Worker PID: \$WORKER_PID\"
    echo \"\"

    # Wait for unified worker to initialize (checks ETCD registration)
    wait_for_worker \"Unified\" \$WORKER_PID || exit 1

    echo ''
    echo '========================================================='
    echo 'Step 2: Starting Dynamo Frontend (HTTP API on port $HTTP_PORT)...'
    echo '========================================================='
    python3 -m dynamo.frontend \
      --http-port=$HTTP_PORT \
      --model-name $SERVED_MODEL_NAME \
      --model-path $MODEL &
    FRONTEND_PID=\$!
    echo \"Frontend PID: \$FRONTEND_PID\"
    echo \"Waiting 15s for frontend to discover workers...\"
    sleep 15
    echo \"\"

    echo ''
    echo '========================================================='
    echo '✓ All components started successfully!'
    echo '========================================================='
    echo \"Infrastructure Services (External):\"
    echo \"  ETCD: localhost:$ETCD_CLIENT_PORT\"
    echo \"  NATS: localhost:$NATS_PORT\"
    echo \"\"
    echo \"Dynamo Components (This Container):\"
    echo \"  Unified Worker: PID \$WORKER_PID  (GPUs $WORKER_GPUS, TP=$TP_SIZE, internal port 30000)\"
    echo \"  Frontend: PID \$FRONTEND_PID  (HTTP API on port $HTTP_PORT)\"
    echo ''
    echo 'Request Flow:'
    echo '  Client → Frontend API (port $HTTP_PORT)'
    echo '         ↓'
    echo '  Frontend discovers workers via ETCD'
    echo '         ↓'
    echo '  Frontend routes to Unified Worker'
    echo '         ↓'
    echo '  Response'
    echo '========================================================='

    # Monitor all processes
    while true; do
        # Check if any critical process died
        if ! kill -0 \$FRONTEND_PID 2>/dev/null; then
            echo \"ERROR: Frontend died!\"
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
    echo "✓ Dynamo SGLang FULL STACK Started (UNIFIED MODE)!"
    echo "========================================================="
    echo ""
    echo "Architecture:"
    echo "  Client Request"
    echo "    ↓"
    echo "  Dynamo Frontend (port $HTTP_PORT)"
    echo "    ↓"
    echo "  Frontend discovers workers via ETCD"
    echo "    ↓"
    echo "  Frontend routes to Unified Worker"
    echo "    ↓              (localhost:$ETCD_CLIENT_PORT - worker discovery)"
    echo "  Unified Worker (GPUs $WORKER_GPUS, TP=$TP_SIZE)"
    echo "    ↓"
    echo "  Response"
    echo ""
    echo "Infrastructure Services (Managed):"
    echo "  ETCD: etcd-dynamo container, localhost:$ETCD_CLIENT_PORT"
    echo "  NATS: nats-dynamo container, localhost:$NATS_PORT"
    echo ""
    echo "Dynamo Components (This Container):"
    echo "  Frontend: HTTP API on port $HTTP_PORT"
    echo "  Unified Worker: GPUs $WORKER_GPUS (TP=$TP_SIZE, internal port 30000)"
    echo ""
    echo "API Endpoint: http://localhost:$HTTP_PORT/v1/chat/completions"
    echo "Health Check: http://localhost:$HTTP_PORT/health"
    echo "Models Endpoint: http://localhost:$HTTP_PORT/v1/models"
    echo ""
    echo "Useful Commands:"
    echo "  Interactive shell:    docker exec -it $CONTAINER_NAME bash"
    echo "  View Dynamo logs:     docker logs -f $CONTAINER_NAME"
    echo "  View ETCD logs:       docker logs -f etcd-dynamo"
    echo "  View NATS logs:       docker logs -f nats-dynamo"
    echo "  GPU usage:            watch -n 2 nvidia-smi"
    echo "  Stop all:             bash stop_dynamo.sh"
    echo ""
    echo "========================================================="
    echo "Test Request:"
    echo "========================================================="
    echo ""
    echo "# Basic test"
    echo "curl http://localhost:$HTTP_PORT/v1/chat/completions \\"
    echo "  -H 'Content-Type: application/json' \\"
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
    echo "NAT Integration Test:"
    echo "========================================================="
    echo ""
    echo "cd /path/to/NeMo-Agent-Toolkit"
    echo "source .venv/bin/activate"
    echo ""
    echo "nat run \\"
    echo "  --config_file examples/dynamo_integration/react_benchmark_agent/configs/config_dynamo_e2e_test.yml \\"
    echo "  --input 'Hello'"
    echo ""
    echo "========================================================="
    echo ""
    echo "Waiting for SGLang to initialize (this will likely take 5-10 minutes for a 70B model)..."
    echo "Monitoring logs (Ctrl+C to exit, container continues)..."
    echo ""

    # Wait for server to be ready (check /v1/models which only works when workers are discovered)
    echo "Checking for API availability (timeout=15 minutes)..."
    max_attempts=900
    attempt=0

    while [ $attempt -lt $max_attempts ]; do
        # Check /v1/models - only returns data when workers are registered
        models_response=$(curl -s http://localhost:$HTTP_PORT/v1/models 2>/dev/null)
        if echo "$models_response" | grep -q '"id"'; then
            echo "✓ SGLang API is ready! (models discovered)"
            break
        fi
        attempt=$((attempt + 1))
        if [ $((attempt % 15)) -eq 0 ]; then
            echo "  ... still waiting ($attempt/$max_attempts)"
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
