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

# Dynamo SGLang with OPTIMIZED Thompson Sampling Router Architecture
# 
# Key difference from generalized architecture:
#   - Uses DEFAULT Dynamo frontend (python -m dynamo.frontend)
#   - Custom Processor + Router components
#   - Routing hints passed via nvext.annotations instead of HTTP headers
#   - Prometheus metrics instead of CSV files
#
# Architecture:
#   Client → Default Dynamo Frontend (tokenization + nvext parsing)
#         ↓ PreprocessedRequest with annotations
#   Custom Processor (extracts hints, queries router)
#         ↓ RouterRequest
#   Custom Router (Thompson Sampling + KV overlap)
#         ↓ worker_id
#   SGLang Backend Worker
#         ↓ response tokens
#   Processor sends feedback to Router
#
# Components:
#   - ETCD (metadata and worker discovery)
#   - NATS (message queue for KV events)
#   - Default Dynamo Frontend (HTTP API on port 8000)
#   - Custom Router (Thompson Sampling + KV overlap)
#   - Custom Processor (hint extraction + routing)
#   - SGLang Worker (unified mode, GPUs 0-3, TP=4)
#
# Prometheus Metrics:
#   - Frontend: http://localhost:8000/metrics
#   - Backend/Router/Processor: http://localhost:8081/metrics
#
# To stop all components: bash stop_dynamo.sh

set -euo pipefail

# Configuration Variables (can be overridden via environment variables)
CONTAINER_NAME="dynamo-sglang-optimized"
WORKER_GPUS="${DYNAMO_GPU_DEVICES:-0,1,2,3}"
TP_SIZE="${DYNAMO_TP_SIZE:-4}"
HTTP_PORT="${DYNAMO_HTTP_PORT:-8000}"
METRICS_PORT="${DYNAMO_METRICS_PORT:-8081}"
MODEL="/workspace/models/Llama-3.3-70B-Instruct"
SERVED_MODEL_NAME="${DYNAMO_MODEL_NAME:-llama-3.3-70b}"
IMAGE="nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.7.1"
SHM_SIZE="${DYNAMO_SHM_SIZE:-16g}"
WORKER_INIT_TIMEOUT_S="${DYNAMO_WORKER_INIT_TIMEOUT_S:-600}"

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
        echo "  huggingface-cli download meta-llama/Llama-3.3-70B-Instruct --local-dir ${DYNAMO_MODEL_DIR}"
        exit 1
    fi
fi
LOCAL_MODEL_DIR="${DYNAMO_MODEL_DIR}"

# Repository directory - auto-detect from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUSTOM_DYNAMO_DIR="${SCRIPT_DIR}/optimized"

echo "========================================================="
echo "Dynamo SGLang with OPTIMIZED Thompson Sampling Router"
echo "========================================================="
echo "Model: Llama-3.3-70B-Instruct"
echo "Container: $CONTAINER_NAME"
echo "HTTP Port: $HTTP_PORT (default Dynamo frontend)"
echo "Metrics Port: $METRICS_PORT (Prometheus)"
echo ""
echo "Architecture Differences (vs generalized):"
echo "  - Default Dynamo frontend (not custom frontend.py)"
echo "  - Hints via nvext.annotations (not HTTP headers)"
echo "  - Prometheus metrics (not CSV files)"
echo ""
echo "Components:"
echo "  - ETCD (metadata and discovery)"
echo "  - NATS (message queue for KV events)"
echo "  - Default Frontend (HTTP API on port $HTTP_PORT)"
echo "  - Custom Router (Thompson Sampling + KV overlap)"
echo "  - Custom Processor (hint extraction + routing)"
echo "  - SGLang Worker (unified mode)"
echo ""
echo "Backend Worker:"
echo "  Unified: GPUs $WORKER_GPUS (TP=$TP_SIZE)"
echo ""
echo "========================================================="

# Verify custom components exist
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
        read -p "Please enter your HuggingFace token (or press Enter to skip): " HF_TOKEN
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
    echo "  huggingface-cli download meta-llama/Llama-3.3-70B-Instruct --local-dir $LOCAL_MODEL_DIR"
    echo ""
    read -p "Continue anyway (model will be downloaded from HuggingFace)? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start container with optimized Thompson Sampling components
echo ""
echo "Starting Dynamo container with OPTIMIZED Thompson Sampling components..."
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
  -e DYN_HTTP_PORT=$HTTP_PORT \
  -e DYN_SYSTEM_PORT=$METRICS_PORT \
  -e DYN_ROUTER_MODE=round-robin \
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
        local max_wait=${WORKER_INIT_TIMEOUT_S:-600}
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

    echo '========================================================='
    echo 'Step 1: Starting Unified Worker (GPUs 0,1,2,3 = Host GPUs $WORKER_GPUS)...'
    echo '========================================================='
    # CRITICAL: Register worker at dynamo.worker.generate (not default backend.generate)
    # This allows the custom Processor to register as backend.generate and intercept
    # frontend requests, then forward to these workers after Thompson Sampling routing.
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python3 -m dynamo.sglang \
      --model-path $MODEL \
      --served-model-name $SERVED_MODEL_NAME \
      --host 0.0.0.0 \
      --port 30000 \
      --tp $TP_SIZE \
      --trust-remote-code \
      --enable-metrics \
      --mem-fraction-static 0.8 \
      --endpoint dynamo.worker.generate &
    WORKER_PID=\$!
    echo \"Unified Worker PID: \$WORKER_PID\"
    echo \"Registered at: dynamo.worker.generate\"
    echo \"\"

    # Wait for unified worker to initialize
    wait_for_worker \"Unified\" \$WORKER_PID || exit 1

    echo ''
    echo '========================================================='
    echo 'Step 2: Starting Custom Router (Thompson Sampling + Prometheus)...'
    echo '========================================================='
    # Router uses config.yaml for all parameters
    # Override specific values with --affinity-base, --temp-base, --lints-v, or --override
    python3 /workspace/custom_dynamo/router.py \
      --config /workspace/custom_dynamo/config.yaml &
    ROUTER_PID=\$!
    echo \"Router PID: \$ROUTER_PID\"
    sleep 15
    echo \"\"

    echo ''
    echo '========================================================='
    echo 'Step 3: Starting Custom Processor (Dynamic Discovery Mode)...'
    echo '========================================================='
    # DYNAMIC DISCOVERY MODE (forward-compatible, --static-endpoint deprecated):
    # Processor registers as dynamo.backend.generate AND calls register_llm()
    # to advertise a model card in ETCD. The frontend's ModelWatcher discovers
    # this and routes requests to us.
    python3 /workspace/custom_dynamo/processor.py \
      --enable-router \
      --model-path $MODEL \
      --model-name $SERVED_MODEL_NAME &
    PROCESSOR_PID=\$!
    echo \"Processor PID: \$PROCESSOR_PID\"
    echo \"Model: $SERVED_MODEL_NAME (from $MODEL)\"
    echo \"Registered at: dynamo.backend.generate (discovered via ETCD model card)\"
    echo \"Forwards to: dynamo.worker.generate (actual SGLang workers)\"
    sleep 15
    echo \"\"

    echo ''
    echo '========================================================='
    echo 'Step 4: Starting Default Dynamo Frontend (Dynamic Discovery)...'
    echo '========================================================='
    # DYNAMIC DISCOVERY MODE (forward-compatible):
    # No --static-endpoint needed! The frontend uses its ModelWatcher to
    # discover backends registered in ETCD. Our processor registered a
    # model card in Step 3, so the frontend will find and route to it.
    python3 -m dynamo.frontend \
      --http-port $HTTP_PORT \
      --model-name $SERVED_MODEL_NAME \
      --model-path $MODEL &
    FRONTEND_PID=\$!
    echo \"Frontend PID: \$FRONTEND_PID\"
    echo \"Discovery: ETCD ModelWatcher (no --static-endpoint)\"
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
    echo \"  Unified Worker: PID \$WORKER_PID  (GPUs $WORKER_GPUS, TP=$TP_SIZE)\"
    echo \"    → Registered at: dynamo.worker.generate\"
    echo \"  Router: PID \$ROUTER_PID  (Thompson Sampling + Prometheus)\"
    echo \"    → Registered at: dynamo.router.{find_worker,feedback}\"
    echo \"  Processor: PID \$PROCESSOR_PID  (NVExt annotation extraction)\"
    echo \"    → Registered at: dynamo.backend.generate (model card in ETCD)\"
    echo \"  Frontend: PID \$FRONTEND_PID  (Default Dynamo HTTP API on port $HTTP_PORT)\"
    echo \"    → Discovery: ETCD ModelWatcher (finds processor's model card)\"
    echo ''
    echo 'Request Flow (Dynamic Discovery Mode):'
    echo '  Client → Default Frontend API (port $HTTP_PORT)'
    echo '         ↓ (tokenization + nvext parsing)'
    echo '  Frontend discovers backends via ETCD ModelWatcher'
    echo '         ↓ (finds Processor model card!)'
    echo '  Custom Processor (dynamo.backend.generate-{id})'
    echo '         ↓ (extract hints from annotations)'
    echo '         ↓ (query Thompson Sampling router)'
    echo '  Custom Router → worker_id'
    echo '         ↓ (KV overlap + workload-aware selection)'
    echo '  Processor routes to → dynamo.worker.generate (with worker_id)'
    echo '         ↓'
    echo '  Unified Worker (dynamo.worker.generate)'
    echo '         ↓'
    echo '  Response + Feedback to Router'
    echo ''
    echo 'Prometheus Metrics:'
    echo '  - Frontend: http://localhost:$HTTP_PORT/metrics'
    echo '  - Backend: http://localhost:$METRICS_PORT/metrics'
    echo '  - Router: thompson_router_* metrics'
    echo '  - Processor: thompson_processor_* metrics'
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
    echo "✓ Dynamo with OPTIMIZED Thompson Sampling Router Started!"
    echo "========================================================="
    echo ""
    echo "Architecture (Dynamic Discovery - Forward Compatible):"
    echo ""
    echo "  Endpoint Registration:"
    echo "    • SGLang Worker:  dynamo.worker.generate  (actual inference)"
    echo "    • Processor:      dynamo.backend.generate + ETCD model card"
    echo "    • Router:         dynamo.router.{find_worker,feedback}"
    echo ""
    echo "  Discovery Mode:"
    echo "    • Frontend uses ETCD ModelWatcher (no --static-endpoint)"
    echo "    • Processor registers model card via register_llm()"
    echo "    • Frontend discovers processor as a 'backend' automatically"
    echo ""
    echo "  Request Flow:"
    echo "    Client Request (with nvext.annotations)"
    echo "      ↓"
    echo "    Default Dynamo Frontend (port $HTTP_PORT)"
    echo "      ↓ discovers backends via ETCD ModelWatcher"
    echo "    Custom Processor (discovered via model card)"
    echo "      ↓ extracts: prefix_id, total_requests, osl, iat"
    echo "      ↓ queries Thompson Sampling router"
    echo "    Custom Router → worker_id"
    echo "      ↓ KV overlap + workload-aware selection"
    echo "    Processor forwards to dynamo.worker.generate"
    echo "      ↓"
    echo "    Unified Worker (GPUs $WORKER_GPUS, TP=$TP_SIZE)"
    echo "      ↓"
    echo "    Response + Feedback Loop"
    echo ""
    echo "Infrastructure Services (Managed):"
    echo "  ETCD: etcd-dynamo container, localhost:2379"
    echo "  NATS: nats-dynamo container, localhost:4222"
    echo ""
    echo "Prometheus Metrics:"
    echo "  Frontend: http://localhost:$HTTP_PORT/metrics"
    echo "  Backend/Router/Processor: http://localhost:$METRICS_PORT/metrics"
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
    echo "Useful Commands:"
    echo "  Interactive shell:    docker exec -it $CONTAINER_NAME bash"
    echo "  View Dynamo logs:     docker logs -f $CONTAINER_NAME"
    echo "  View ETCD logs:       docker logs -f etcd-dynamo"
    echo "  View NATS logs:       docker logs -f nats-dynamo"
    echo "  GPU usage:            watch -n 2 nvidia-smi"
    echo "  Stop all:             bash stop_dynamo.sh"
    echo ""
    echo "Prometheus Metrics:"
    echo "  curl http://localhost:$HTTP_PORT/metrics | grep dynamo"
    echo "  curl http://localhost:$METRICS_PORT/metrics | grep thompson"
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
    echo "Waiting for SGLang to initialize (this may take 5-10 minutes for a 70B model)..."
    echo "Monitoring logs (Ctrl+C to exit, container continues)..."
    echo ""

    # Wait for server to be ready
    echo "Checking for API availability (timeout=15 minutes)..."
    max_attempts=900
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
