#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run all A365 smoke scenarios: telemetry-only, telemetry+tooling, A365 front-end.
# From examples/a365_example: ./scripts/smoke_all.sh
# Set A365_BEARER_TOKEN (and for scenario 3: A365_APP_ID, A365_APP_PASSWORD) as needed.

set -u
cd "$(dirname "$0")/.."
# Load .env if present (e.g. A365_BEARER_TOKEN)
if [ -f .env ]; then set -a; source .env; set +a; fi
# Real JWTs from get_a365_token.py are hundreds of chars; short values break telemetry/MCP auth.
# (Use a named temp for length: macOS bash 3.2 rejects ${#VAR:-}.)
if [ -n "${AZURE_TENANT_ID:-}" ] && [ -n "${AZURE_CLIENT_ID:-}" ] && [ -n "${AZURE_CLIENT_SECRET:-}" ]; then
  _a365_bt_len="${A365_BEARER_TOKEN-}"
  _a365_bt_len="${#_a365_bt_len}"
  if [ "${_a365_bt_len:-0}" -lt 120 ]; then
    echo "Refreshing A365_BEARER_TOKEN from client credentials (token was ${_a365_bt_len} chars)."
    export A365_BEARER_TOKEN="$(uv run python scripts/get_a365_token.py 2>/dev/null)" || {
      echo "WARN: get_a365_token.py failed; keep existing A365_BEARER_TOKEN if set."
    }
  fi
  unset _a365_bt_len
fi
CONFIG_DIR="${CONFIG_DIR:-configs}"
# NAT + workflow build + NIM can exceed 12s on cold start; curl 000 = connection refused.
STARTUP_WAIT="${STARTUP_WAIT:-22}"
HTTP_PORT="${HTTP_PORT:-8000}"

run_scenario() {
  local name="$1"
  local config="$2"
  shift 2
  echo "--- Scenario: $name (config=$config, nat $*) ---"
  if ! [ -f "$CONFIG_DIR/$config" ]; then
    echo "SKIP: $CONFIG_DIR/$config not found"
    return 0
  fi
  uv run nat "$@" --config_file "$CONFIG_DIR/$config" &
  local pid=$!
  sleep "$STARTUP_WAIT"
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "FAIL: server exited before request"
    return 1
  fi
  if [ "${1:-}" = "serve" ]; then
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://localhost:${HTTP_PORT}/generate" \
      -H "Content-Type: application/json" -d '{"input_message": "What time is it?"}' || true)
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    if [ "$code" = "200" ]; then
      echo "PASS: /generate returned 200"
    else
      echo "FAIL: /generate returned $code (expected 200)"
      return 1
    fi
  else
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    echo "PASS: server started and stopped (no HTTP check for nat $*)"
  fi
  return 0
}

FAILED=0

# 1. Telemetry-only (FastAPI serve + A365 tracing)
run_scenario "Telemetry only" "config_local_telemetry_only.yml" "serve" || FAILED=1

# 2. Telemetry + A365 MCP tooling (requires nvidia-nat-mcp; discovery may 401)
run_scenario "Telemetry + tooling" "config_telemetry_and_tooling.yml" "serve" || {
  echo "  (Tip: install nvidia-nat-mcp if A365 tooling is required; 401 on discovery is expected without scope)"
  FAILED=1
}

# 3. A365 front-end (Bot on 3978; requires A365_APP_ID, A365_APP_PASSWORD)
if [ -z "${A365_APP_ID:-}" ] || [ -z "${A365_APP_PASSWORD:-}" ]; then
  echo "--- Scenario: A365 front-end ---"
  echo "SKIP: set A365_APP_ID and A365_APP_PASSWORD to run"
else
  run_scenario "A365 front-end" "config_a365_front_end.yml" start a365 || {
    echo "  (Tip: run manually: uv run nat start a365 --config_file $CONFIG_DIR/config_a365_front_end.yml)"
    FAILED=1
  }
fi

[ $FAILED -eq 0 ] && echo "--- All scenarios passed or skipped ---" || echo "--- One or more scenarios failed ---"
exit $FAILED
