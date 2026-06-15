#!/usr/bin/env bash
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

# Nemotron 3 Nano on 2 GPUs with a Dynamo frontend and priority-aware SGLang worker.
# Edit the config below, then: bash dynamo_stack_sensitivity.sh
# Ctrl+C to stop. Logs in /tmp/dynamo-stack/
#
# Prerequisites (from the Dynamo source checkout; run once, stays up):
#   export DYNAMO_SOURCE_DIR="${HOME}/dynamo"
#   cd "$DYNAMO_SOURCE_DIR/dev"
#   docker compose -f docker-compose.yml up -d --remove-orphans
#   docker compose -f docker-observability.yml up -d --remove-orphans
#   cd "$DYNAMO_SOURCE_DIR"
set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
CONTEXT_LENGTH=262144
MEM_FRACTION=0.7
HTTP_PORT="${HTTP_PORT:-8099}"
TP_SIZE="${TP_SIZE:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-1800}"

# This demo runs a local two-GPU stack. Avoid loading host-specific NCCL network
# plugins unless the user explicitly opts into them for their environment.
NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

LOG_DIR="/tmp/dynamo-stack"

# ── Cleanup ──────────────────────────────────────────────────────────────────
PIDS=()

# shellcheck disable=SC2329 # Invoked by EXIT/INT/TERM traps.
cleanup() {
    echo ""
    echo "Shutting down..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    echo "Done. Logs in $LOG_DIR/"
}
trap cleanup EXIT INT TERM

mkdir -p "$LOG_DIR"

require_positive_integer() {
    local name="$1"
    local value="$2"

    if ! [[ "$value" =~ ^[0-9]+$ ]] || ((10#$value < 1)); then
        echo "$name must be a positive integer. Current value: $value"
        exit 1
    fi
}

validate_config() {
    require_positive_integer "TP_SIZE" "$TP_SIZE"
    require_positive_integer "HTTP_PORT" "$HTTP_PORT"
    require_positive_integer "STARTUP_TIMEOUT_SECONDS" "$STARTUP_TIMEOUT_SECONDS"

    local visible_devices="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
    if [[ -z "$visible_devices" ]]; then
        echo "CUDA_VISIBLE_DEVICES must list at least $TP_SIZE device(s)."
        exit 1
    fi
    if [[ "$visible_devices" == ,* || "$visible_devices" == *, || "$visible_devices" == *,,* ]]; then
        echo "CUDA_VISIBLE_DEVICES contains an empty device entry: $visible_devices"
        exit 1
    fi
    CUDA_VISIBLE_DEVICES="$visible_devices"

    local devices
    IFS=',' read -r -a devices <<<"$visible_devices"
    if ((${#devices[@]} < TP_SIZE)); then
        echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES exposes ${#devices[@]} device(s), but TP_SIZE=$TP_SIZE requires at least $TP_SIZE."
        exit 1
    fi
}

wait_for_model() {
    local frontend_pid="$1"
    local worker_pid="$2"
    local endpoint="http://localhost:${HTTP_PORT}/v1/models"
    local deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
    local response

    echo "Waiting up to ${STARTUP_TIMEOUT_SECONDS}s for $MODEL to register at $endpoint..."
    while true; do
        if response=$(curl -sf "$endpoint" 2>/dev/null) && grep -F "\"$MODEL\"" <<<"$response" >/dev/null; then
            echo "Model registered. Dynamo endpoint is ready."
            return 0
        fi

        if ! kill -0 "$frontend_pid" 2>/dev/null; then
            echo "Dynamo frontend exited before model registration. See $LOGFILE."
            exit 1
        fi

        if ! kill -0 "$worker_pid" 2>/dev/null; then
            echo "Dynamo worker exited before model registration. See $LOGFILE."
            exit 1
        fi

        if ((SECONDS >= deadline)); then
            echo "Timed out waiting for model registration at $endpoint."
            echo "Last /v1/models response:"
            curl -s "$endpoint" || true
            echo ""
            echo "See $LOGFILE."
            exit 1
        fi

        sleep 5
    done
}

monitor_processes() {
    while true; do
        for pid in "${PIDS[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "A Dynamo process exited. See $LOGFILE."
                exit 1
            fi
        done

        sleep 5
    done
}

# ── Preflight ────────────────────────────────────────────────────────────────
validate_config
curl -sf http://localhost:2379/health >/dev/null 2>&1 || { echo "etcd not running. See header comment."; exit 1; }
curl -sf http://localhost:8222/healthz >/dev/null 2>&1 || { echo "NATS not running. See header comment."; exit 1; }

# ── Priority scheduling flags (SGLang version-dependent) ───────────────────────
# The worker is launched via `python3 -m dynamo.sglang`, which validates CLI args
# at its OWN layer before delegating to SGLang's argparse. An arg that raw SGLang
# accepts can still be rejected by dynamo.sglang, making the worker exit during
# argument parsing before the model registers on /v1/models.
#
# --schedule-low-priority-values-first is one such arg: SGLang exposes it, but
# dynamo.sglang forbids it (components/src/dynamo/sglang/args.py) because Dynamo
# normalizes request priority so that HIGHER values are always higher priority at
# the API layer. We therefore never pass it. Priority routing still works through
# --enable-priority-scheduling alone; the priority *values* injected by the NAT
# Dynamo client must follow Dynamo's higher-is-higher convention (see dynamo_llm.py).
#
# Probe raw SGLang ServerArgs only to decide whether --enable-priority-scheduling
# is available; dynamo.sglang accepts that flag when SGLang supports it.
PRIORITY_ARGS=()
SGLANG_PRIORITY_PROBE="$(python3 - <<'PY' 2>/dev/null || true
try:
    from sglang.srt.server_args import ServerArgs
    fields = getattr(ServerArgs, "__dataclass_fields__", {})
    print("PROBE_OK")
    if "enable_priority_scheduling" in fields:
        print("enable_priority_scheduling")
except Exception:
    pass
PY
)"
if ! grep -q PROBE_OK <<<"$SGLANG_PRIORITY_PROBE"; then
    echo "WARNING: could not probe SGLang ServerArgs; starting worker without priority"
    echo "         scheduling flags. The priority-routing benefit this demo measures"
    echo "         will not be exercised."
elif grep -q enable_priority_scheduling <<<"$SGLANG_PRIORITY_PROBE"; then
    PRIORITY_ARGS+=(--enable-priority-scheduling)
else
    echo "WARNING: installed SGLang does not support --enable-priority-scheduling;"
    echo "         starting worker without priority scheduling. The priority-routing"
    echo "         benefit this demo measures will not be exercised."
fi

LOGFILE="$LOG_DIR/all.log"
: > "$LOGFILE"

echo "Launching Dynamo stack with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, TP_SIZE=$TP_SIZE"
echo "NCCL startup defaults: NCCL_NET_PLUGIN=$NCCL_NET_PLUGIN, NCCL_IB_DISABLE=$NCCL_IB_DISABLE"

# ── Frontend ─────────────────────────────────────────────────────────────────
OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend \
    --http-port "$HTTP_PORT" \
    > >(tee -a "$LOGFILE") 2>&1 &
FRONTEND_PID=$!
PIDS+=("$FRONTEND_PID")

# ── Workers ──────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
OTEL_SERVICE_NAME=dynamo-worker-0 \
DYN_SYSTEM_PORT=8081 \
NCCL_NET_PLUGIN="$NCCL_NET_PLUGIN" \
NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
NCCL_DEBUG="$NCCL_DEBUG" \
python3 -m dynamo.sglang \
    --model-path "$MODEL" \
    --served-model-name "$MODEL" \
    --tp "$TP_SIZE" \
    --mem-fraction-static $MEM_FRACTION \
    --context-length $CONTEXT_LENGTH \
    --trust-remote-code \
    --enable-metrics \
    ${PRIORITY_ARGS[@]+"${PRIORITY_ARGS[@]}"} \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080"}' \
    > >(tee -a "$LOGFILE") 2>&1 &
WORKER_PID=$!
PIDS+=("$WORKER_PID")

wait_for_model "$FRONTEND_PID" "$WORKER_PID"

echo "Ctrl+C to stop. Log: $LOGFILE"
monitor_processes
