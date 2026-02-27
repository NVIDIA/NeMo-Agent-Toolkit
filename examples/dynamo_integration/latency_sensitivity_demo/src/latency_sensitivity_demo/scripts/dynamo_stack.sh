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

# Nemotron 3 Nano on 2 GPUs with Dynamo frontend, HiCache, and PIN.
# Edit the config below, then: ./dynamo-stack.sh
# Ctrl+C to stop. Logs in /tmp/dynamo-stack/
#
# Prerequisites (run once, stays up):
#   cd dynamo/deploy
#   docker compose -f docker-compose.yml up -d --remove-orphans
#   docker compose -f docker-observability.yml up -d --remove-orphans
set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
PAGE_SIZE=64
HICACHE_RATIO=1.0
HICACHE_POLICY=write_through
CONTEXT_LENGTH=262144
MEM_FRACTION=0.7

LOG_DIR="/tmp/dynamo-stack"

# ── Cleanup ──────────────────────────────────────────────────────────────────
PIDS=()

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

# ── Preflight ────────────────────────────────────────────────────────────────
curl -sf http://localhost:2379/health >/dev/null 2>&1 || { echo "etcd not running. See header comment."; exit 1; }
curl -sf http://localhost:8222/healthz >/dev/null 2>&1 || { echo "NATS not running. See header comment."; exit 1; }

LOGFILE="$LOG_DIR/all.log"
> "$LOGFILE"

# ── Frontend ─────────────────────────────────────────────────────────────────
OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend \
    --http-port 8099 \
    2>&1 | tee -a "$LOGFILE" &
PIDS+=($!)

# ── Workers ──────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0,1 \
OTEL_SERVICE_NAME=dynamo-worker-0 \
DYN_SYSTEM_PORT=8081 \
python3 -m dynamo.sglang \
    --model-path "$MODEL" \
    --served-model-name "$MODEL" \
    --tp 2 \
    --mem-fraction-static $MEM_FRACTION \
    --context-length $CONTEXT_LENGTH \
    --trust-remote-code \
    --enable-metrics \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080"}' \
    2>&1 | tee -a "$LOGFILE" &
PIDS+=($!)

echo "Ctrl+C to stop. Log: $LOGFILE"
wait
