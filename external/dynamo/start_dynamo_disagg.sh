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

# Dynamo SGLang FULL STACK with Disaggregation
# Architecture: ETCD + NATS + Dynamo Frontend (API) → SGLang Backend Workers (Disaggregated)
#
# This script manages ALL required components:
#   - ETCD (metadata and worker discovery)
#   - NATS (message queue for prefill requests)
#   - Dynamo Frontend (HTTP API with built-in processor + router)
#   - Prefill Worker (GPUs 0,1, TP=2)
#   - Decode Worker (GPUs 2,3, TP=2)
#
# Frontend: Port 8099 (HTTP API)
# ETCD: localhost:2379 (container: etcd-dynamo)
# NATS: localhost:4222 (container: nats-dynamo)
# Workers run in container: dynamo-sglang
#
# To stop all components: bash stop_dynamo_disagg.sh

# Configuration Variables (can be overridden via environment variables)
CONTAINER_NAME="dynamo-sglang"
PREFILL_GPUS="${DYNAMO_PREFILL_GPUS:-0,1}"
DECODE_GPUS="${DYNAMO_DECODE_GPUS:-2,3}"
TP_SIZE="${DYNAMO_TP_SIZE:-2}"
HTTP_PORT="${DYNAMO_HTTP_PORT:-8099}"
MODEL="/workspace/models/Llama-3.3-70B-Instruct"
SERVED_MODEL_NAME="${DYNAMO_MODEL_NAME:-llama-3.3-70b}"
IMAGE="nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.7.1"
SHM_SIZE="${DYNAMO_SHM_SIZE:-16g}"
WORKER_INIT_TIMEOUT_S="${DYNAMO_WORKER_INIT_TIMEOUT_S:-600}"

# Disaggregation configuration
DISAGG_BOOTSTRAP_PORT="${DYNAMO_DISAGG_BOOTSTRAP_PORT:-12345}"
DISAGG_TRANSFER_BACKEND="${DYNAMO_DISAGG_TRANSFER_BACKEND:-nixl}"  # Options: nixl, nccl, gloo

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
echo "Dynamo SGLang FULL STACK (DISAGGREGATED MODE)"
echo "========================================================="
echo "Model: Llama-3.3-70B-Instruct"
echo "Container: $CONTAINER_NAME"
echo "HTTP Port: $HTTP_PORT"
echo ""
echo "Components:"
echo "  - ETCD (metadata and discovery)"
echo "  - NATS (message queue for prefill requests)"
echo "  - Dynamo Frontend (HTTP API on port $HTTP_PORT)"
echo "  - SGLang Workers (disaggregated prefill/decode)"
echo ""
echo "Backend Workers:"
echo "  Prefill: GPUs $PREFILL_GPUS (TP=$TP_SIZE)"
echo "  Decode: GPUs $DECODE_GPUS (TP=$TP_SIZE)"
echo "  Transfer: $DISAGG_TRANSFER_BACKEND"
echo "  Mode: DISAGGREGATED (prefill/decode separation)"
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

# Start container with disaggregated SGLang server
echo ""
echo "Starting Dynamo container with disaggregated SGLang server..."
docker run -d \
  --name $CONTAINER_NAME \
  --gpus "\"device=${PREFILL_GPUS},${DECODE_GPUS}\"" \
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
  $IMAGE \
  bash -c "
    set -e  # Exit on any error

    echo '========================================================='
    echo 'Verifying external infrastructure services...'
    echo '========================================================='

    # Verify ETCD is accessible
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
    # For disaggregated mode, we track expected worker count
    # wait_for_worker() {
    #     local worker_type=\$1
    #     local pid=\$2
    #     local expected_count=\${3:-1}  # Expected number of registered workers
    #     local max_wait=300
    #     local elapsed=0
    #     local poll_interval=5

    #     echo \"Waiting for \$worker_type worker (PID \$pid) to initialize...\"
    #     echo \"  Detection: ETCD worker registration (expecting \$expected_count worker(s))\"
    #     echo \"  Timeout: \${max_wait}s\"

    #     while [ \$elapsed -lt \$max_wait ]; do
    #         # Check if process is still running
    #         if ! kill -0 \$pid 2>/dev/null; then
    #             echo \"ERROR: \$worker_type worker process died!\"
    #             return 1
    #         fi

    #         # Check ETCD for registered workers using v3 API
    #         # Query ALL keys to find where Dynamo registers (empty key "" with range_end "\0" = all keys)
    #         # Base64: "" -> AA==, "\0" -> AA==  (we use keys_only to reduce response size)
    #         local etcd_response=\$(curl -s --max-time 2 st \
    #             -X POST \
    #             -H \"Content-Type: application/json\" \
    #             -d '{\"key\":\"AA==\",\"range_end\":\"AA==\",\"keys_only\":true}' 2>&1)

    #         # Extract count from response and check if we have enough workers
    #         local current_count=\$(echo \"\$etcd_response\" | grep -o '\"count\":\"[0-9]*\"' | grep -o '[0-9]*' || echo \"0\")

    #         # Debug: Print ETCD response every 30s (truncated)
    #         if [ \$((elapsed % 30)) -eq 0 ] && [ \$elapsed -gt 0 ]; then
    #             echo \"  [DEBUG] ETCD keys found: \$(echo \"\$etcd_response\" | grep -o '\"key\":\"[^\"]*\"' | head -5)\"
    #             echo \"  [DEBUG] ETCD count: \$(echo \"\$etcd_response\" | grep -o '\"count\":\"[^\"]*\"')\"
    #         fi

    #         if [ \"\$current_count\" -ge \"\$expected_count\" ] 2>/dev/null; then
    #             echo \"✓ \$worker_type worker is ready (registered with ETCD at \${elapsed}s, count=\$current_count)\"
    #             return 0
    #         fi

    #         sleep \$poll_interval
    #         elapsed=\$((elapsed + poll_interval))
    #         if [ \$((elapsed % 30)) -eq 0 ]; then
    #             echo \"  ... \${elapsed}s / \${max_wait}s (waiting for ETCD registration, current=\$current_count)\"
    #         fi
    #     done

    #     echo \"ERROR: \$worker_type worker failed to register with ETCD within \${max_wait}s\"
    #     echo \"  Image: $IMAGE\"
    #     echo \"  The model may require more time to load, or there may be a startup error.\"
    #     echo \"  Check worker logs for details.\"
    #     return 1
    # }

    # echo '========================================================='
    # echo 'Step 1: Starting Prefill Worker (GPUs 0,1 = Host GPUs $PREFILL_GPUS)...'
    # echo '========================================================='
    # CUDA_VISIBLE_DEVICES=0,1 \
    # python3 -m dynamo.sglang \
    #   --model-path $MODEL \
    #   --served-model-name $SERVED_MODEL_NAME \
    #   --host 0.0.0.0 \
    #   --port 30000 \
    #   --tp $TP_SIZE \
    #   --trust-remote-code \
    #   --disaggregation-mode prefill \
    #   --disaggregation-bootstrap-port $DISAGG_BOOTSTRAP_PORT \
    #   --disaggregation-transfer-backend $DISAGG_TRANSFER_BACKEND \
    #   --mem-fraction-static 0.8 &
    # PREFILL_PID=\$!
    # echo \"Prefill Worker PID: \$PREFILL_PID\"
    # echo \"\"

    # # Wait for prefill worker to initialize (expects 1 worker in ETCD)
    # wait_for_worker \"Prefill\" \$PREFILL_PID 1 || exit 1

    # echo ''
    # echo '========================================================='
    # echo 'Step 2: Starting Decode Worker (GPUs 2,3 = Host GPUs $DECODE_GPUS)...'
    # echo '========================================================='
    # CUDA_VISIBLE_DEVICES=2,3 \
    # python3 -m dynamo.sglang \
    #   --model-path $MODEL \
    #   --served-model-name $SERVED_MODEL_NAME \
    #   --host 0.0.0.0 \
    #   --tp $TP_SIZE \
    #   --trust-remote-code \
    #   --disaggregation-mode decode \
    #   --disaggregation-bootstrap-port $DISAGG_BOOTSTRAP_PORT \
    #   --disaggregation-transfer-backend $DISAGG_TRANSFER_BACKEND \
    #   --mem-fraction-static 0.8 &
    # DECODE_PID=\$!
    # echo \"Decode Worker PID: \$DECODE_PID\"
    # echo \"\"

    # # Wait for decode worker to initialize (expects 2 workers in ETCD - prefill + decode)
    # wait_for_worker \"Decode\" \$DECODE_PID 2 || exit 1

    echo '========================================================='
    echo 'Steps 1 & 2: Starting Prefill & Decode Workers in PARALLEL...'
    echo '========================================================='

    # Start Prefill Worker (background)
    echo \"Starting Prefill Worker (GPUs 0,1 = Host GPUs $PREFILL_GPUS)...\"
    CUDA_VISIBLE_DEVICES=0,1 \
    python3 -m dynamo.sglang \
      --model-path $MODEL \
      --served-model-name $SERVED_MODEL_NAME \
      --host 0.0.0.0 \
      --port 30000 \
      --tp $TP_SIZE \
      --trust-remote-code \
      --disaggregation-mode prefill \
      --disaggregation-bootstrap-port $DISAGG_BOOTSTRAP_PORT \
      --disaggregation-transfer-backend $DISAGG_TRANSFER_BACKEND \
      --mem-fraction-static 0.8 &
    PREFILL_PID=\$!
    echo \"Prefill Worker PID: \$PREFILL_PID\"

    # Start Decode Worker (background) - immediately, no waiting for prefill
    echo \"Starting Decode Worker (GPUs 2,3 = Host GPUs $DECODE_GPUS)...\"
    CUDA_VISIBLE_DEVICES=2,3 \
    python3 -m dynamo.sglang \
      --model-path $MODEL \
      --served-model-name $SERVED_MODEL_NAME \
      --host 0.0.0.0 \
      --tp $TP_SIZE \
      --trust-remote-code \
      --disaggregation-mode decode \
      --disaggregation-bootstrap-port $DISAGG_BOOTSTRAP_PORT \
      --disaggregation-transfer-backend $DISAGG_TRANSFER_BACKEND \
      --mem-fraction-static 0.8 &
    DECODE_PID=\$!
    echo \"Decode Worker PID: \$DECODE_PID\"
    echo \"\"

    # Wait for BOTH workers to register (expects 2 workers in ETCD)
    echo \"Waiting for both workers to initialize in parallel...\"
    wait_for_workers_parallel() {
        # local max_wait=300
        local max_wait=${WORKER_INIT_TIMEOUT_S:-600}
        local elapsed=0
        local poll_interval=5

        echo \"  Detection: ETCD worker registration (expecting 2 workers)\"
        echo \"  Timeout: \${max_wait}s\"

        while [ \$elapsed -lt \$max_wait ]; do
            # Check if both processes are still running
            if ! kill -0 \$PREFILL_PID 2>/dev/null; then
                echo \"ERROR: Prefill worker process died!\"
                return 1
            fi
            if ! kill -0 \$DECODE_PID 2>/dev/null; then
                echo \"ERROR: Decode worker process died!\"
                return 1
            fi

            # Check ETCD for registered workers
            local etcd_response=\$(curl -s --max-time 2 \
                -X POST http://localhost:2379/v3/kv/range \
                -H \"Content-Type: application/json\" \
                -d '{\"key\":\"AA==\",\"range_end\":\"AA==\",\"keys_only\":true}' 2>&1)

            local current_count=\$(echo \"\$etcd_response\" | grep -o '\"count\":\"[0-9]*\"' | grep -o '[0-9]*' || echo \"0\")

            if [ \"\$current_count\" -ge 2 ] 2>/dev/null; then
                echo \"✓ Both workers registered in ETCD (\$current_count workers)\"
                return 0
            fi

            echo \"  [\${elapsed}s] Waiting... (ETCD workers: \${current_count:-0}/2)\"
            sleep \$poll_interval
            elapsed=\$((elapsed + poll_interval))
        done

        echo \"ERROR: Timeout waiting for workers to register\"
        return 1
    }

    wait_for_workers_parallel || exit 1

    echo ''
    echo '========================================================='
    echo 'Step 3: Starting Dynamo Frontend (HTTP API on port $HTTP_PORT)...'
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
    echo \"  ETCD: localhost:2379\"
    echo \"  NATS: localhost:4222\"
    echo \"\"
    echo \"Dynamo Components (This Container):\"
    echo \"  Prefill Worker: PID \$PREFILL_PID  (GPUs $PREFILL_GPUS, TP=$TP_SIZE, internal port 30000)\"
    echo \"  Decode Worker:  PID \$DECODE_PID  (GPUs $DECODE_GPUS, TP=$TP_SIZE, registers with runtime)\"
    echo \"  Frontend: PID \$FRONTEND_PID  (HTTP API on port $HTTP_PORT)\"
    echo ''
    echo 'Request Flow:'
    echo '  Client → Frontend API (port $HTTP_PORT)'
    echo '         ↓'
    echo '  Frontend discovers workers via ETCD'
    echo '         ↓'
    echo '  Frontend routes to Decode Worker'
    echo '         ↓'
    echo '  Decode Worker ← NATS → Prefill Worker (KV transfer via NIXL)'
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
        if ! kill -0 \$PREFILL_PID 2>/dev/null; then
            echo \"ERROR: Prefill worker died!\"
            exit 1
        fi
        if ! kill -0 \$DECODE_PID 2>/dev/null; then
            echo \"ERROR: Decode worker died!\"
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
    echo "✓ Dynamo SGLang FULL STACK Started (DISAGGREGATED MODE)!"
    echo "========================================================="
    echo ""
    echo "Architecture:"
    echo "  Client Request"
    echo "    ↓"
    echo "  Dynamo Frontend (port $HTTP_PORT)"
    echo "    ↓"
    echo "  Frontend discovers workers via ETCD"
    echo "    ↓"
    echo "  Frontend routes to Decode Worker"
    echo "    ↓"
    echo "  Decode Worker ← NATS → Prefill Worker (KV transfer via NIXL)"
    echo "    ↓              (localhost:4222)"
    echo "  Prefill Worker → NIXL Transfer → Decode Worker"
    echo "    ↓              (ETCD metadata at localhost:2379)"
    echo "  Response"
    echo ""
    echo "Infrastructure Services (Managed):"
    echo "  ETCD: etcd-dynamo container, localhost:2379"
    echo "  NATS: nats-dynamo container, localhost:4222"
    echo ""
    echo "Dynamo Components (This Container):"
    echo "  Frontend: HTTP API on port $HTTP_PORT"
    echo "  Prefill Worker: GPUs $PREFILL_GPUS (TP=$TP_SIZE, internal)"
    echo "  Decode Worker:  GPUs $DECODE_GPUS (TP=$TP_SIZE, internal)"
    echo "  Transfer Backend: $DISAGG_TRANSFER_BACKEND"
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
    echo "  Stop all:             bash stop_dynamo_disagg.sh"
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
    echo "source /path/to/your/venv/bin/activate"
    echo "export HF_HOME=~/.cache/huggingface"
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
