#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Roll out the Bot+MCP+telemetry image to Azure Container Apps and wire A365_BEARER_TOKEN.
#
# Prerequisites: az login; value for A365_BEARER_TOKEN (client-credentials JWT for A365 tooling+telemetry).
#
# Usage (use separate lines in zsh; avoid inline comments containing AZURE_*):
#   cd examples/a365_example && set -a && source .env && set +a
#   export A365_BEARER_TOKEN="$(uv run python scripts/get_a365_token.py)"
#   cd ../.. && ./examples/a365_example/deploy/aca_rollout_mcp.sh
#
# Or:
#   ./aca_rollout_mcp.sh '<paste-jwt>'
#
set -euo pipefail

RG="${ACA_RG:-a365-nat-dev-rg-1}"
APP="${ACA_APP:-nat-a365-bot}"
# Pin tag so ACA pulls a new image (avoids stale :latest digest on the node)
IMAGE="${ACA_IMAGE:-a365natdevacr1.azurecr.io/nat-a365-bot:20260327-mcp}"
SECRET_NAME="${ACA_BEARER_SECRET:-a365-bearer}"

TOKEN="${1:-${A365_BEARER_TOKEN:-}}"
if [[ -z "${TOKEN}" ]]; then
  echo "Set A365_BEARER_TOKEN or pass JWT as first argument." >&2
  exit 1
fi

echo "ACA_IMAGE=${IMAGE} (must match the tag you pushed from build_and_push.sh)"

echo "Setting ACA secret ${SECRET_NAME} (name max 20 chars; adjust ACA_BEARER_SECRET if needed)"
az containerapp secret set -g "${RG}" -n "${APP}" --secrets "${SECRET_NAME}=${TOKEN}"

# Optional: export AZURE_TENANT_ID before running (backup if config omits tooling_gateway_tenant_id).
# Optional: export A365_ALLOWED_AUDIENCES="aud1,aud2" if inbound Bot JWT aud differs from A365_APP_ID.
EXTRA_ENV=()
if [[ -n "${AZURE_TENANT_ID:-}" ]]; then
  EXTRA_ENV+=("AZURE_TENANT_ID=${AZURE_TENANT_ID}")
  echo "Also setting AZURE_TENANT_ID on ${APP} (from shell environment)."
fi
if [[ -n "${A365_ALLOWED_AUDIENCES:-}" ]]; then
  EXTRA_ENV+=("A365_ALLOWED_AUDIENCES=${A365_ALLOWED_AUDIENCES}")
  echo "Also setting A365_ALLOWED_AUDIENCES on ${APP} (from shell environment)."
fi

echo "Updating ${APP}: image ${IMAGE}, ENVIRONMENT=Production, A365_BEARER_TOKEN=secretref:${SECRET_NAME}"
# ENVIRONMENT=Production: A365 SDK loads MCP server URLs from the tooling gateway (not only local ToolingManifest).
# The image Dockerfile defaults to Development for local docker; ACA must override for real cloud MCP.
az containerapp update -g "${RG}" -n "${APP}" \
  --image "${IMAGE}" \
  --set-env-vars \
    "ENVIRONMENT=Production" \
    "A365_BEARER_TOKEN=secretref:${SECRET_NAME}" \
    "${EXTRA_ENV[@]}"

echo "Done. Watch revisions: az containerapp revision list -g ${RG} -n ${APP} -o table"
echo "Logs: az containerapp logs show -g ${RG} -n ${APP} --follow"
