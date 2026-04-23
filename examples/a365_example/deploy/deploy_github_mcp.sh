#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Deploy the GitHub MCP server (ghcr.io/github/github-mcp-server) to Azure Container Apps.
# No build step needed — uses the official prebuilt image directly.
#
# Usage:
#   export GITHUB_TOKEN="github_pat_..."
#   ./deploy_github_mcp.sh

set -euo pipefail

RG="${RG:-a365-nat-dev-rg-1}"
APP="${APP:-github-mcp}"
IMAGE="ghcr.io/github/github-mcp-server:latest"
ENVIRONMENT="${ACA_ENVIRONMENT:-$(az containerapp env list -g "${RG}" --query "[0].name" -o tsv)}"

GITHUB_TOKEN="${GITHUB_TOKEN:-}"
if [[ -z "${GITHUB_TOKEN}" ]]; then
    echo "ERROR: set GITHUB_TOKEN" >&2
    exit 1
fi

echo "==> Deploying GitHub MCP to ACA (app: ${APP}, env: ${ENVIRONMENT})"
if az containerapp show -g "${RG}" -n "${APP}" &>/dev/null; then
    echo "    Updating existing container app..."
    az containerapp secret set -g "${RG}" -n "${APP}" \
        --secrets "github-token=${GITHUB_TOKEN}"
    az containerapp update -g "${RG}" -n "${APP}" \
        --image "${IMAGE}"
else
    echo "    Creating new container app..."
    az containerapp create \
        -g "${RG}" \
        -n "${APP}" \
        --environment "${ENVIRONMENT}" \
        --image "${IMAGE}" \
        --target-port 8082 \
        --ingress external \
        --min-replicas 1 \
        --max-replicas 1 \
        --secrets "github-token=${GITHUB_TOKEN}" \
        --env-vars "GITHUB_PERSONAL_ACCESS_TOKEN=secretref:github-token" \
        --args "http" "--port" "8082" "--read-only" "--toolsets" "issues,pull_requests,repos,users"
fi

FQDN="$(az containerapp show -g "${RG}" -n "${APP}" --query "properties.configuration.ingress.fqdn" -o tsv)"
echo ""
echo "Done. GitHub MCP URL: https://${FQDN}/mcp"
