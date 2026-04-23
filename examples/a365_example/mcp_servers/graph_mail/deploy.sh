#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build, push, and deploy the Graph Mail MCP server to Azure Container Apps.
#
# Usage (from this directory):
#   export GRAPH_MAIL_TOKEN="$(az account get-access-token --resource https://graph.microsoft.com --query accessToken -o tsv)"
#   ./deploy.sh

set -euo pipefail

REGISTRY="${REGISTRY:-a365natdevacr1.azurecr.io}"
IMAGE_NAME="${IMAGE_NAME:-graph-mail-mcp}"
TAG="${TAG:-$(date +%Y%m%d%H%M)}"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"

RG="${RG:-a365-nat-dev-rg-1}"
APP="${APP:-graph-mail-mcp}"
ENVIRONMENT="${ACA_ENVIRONMENT:-$(az containerapp env list -g "${RG}" --query "[0].name" -o tsv)}"

TOKEN="${GRAPH_MAIL_TOKEN:-}"
if [[ -z "${TOKEN}" ]]; then
    echo "Fetching fresh Graph token via az CLI..."
    TOKEN="$(az account get-access-token --resource https://graph.microsoft.com --query accessToken -o tsv)"
fi

echo "==> Logging in to ACR"
az acr login --name "${REGISTRY%.azurecr.io}"

echo "==> Building and pushing ${FULL_IMAGE}"
docker buildx build \
    --platform linux/amd64 \
    --provenance=false \
    --sbom=false \
    -t "${FULL_IMAGE}" \
    --push \
    .

echo "==> Deploying to ACA (app: ${APP}, env: ${ENVIRONMENT})"
if az containerapp show -g "${RG}" -n "${APP}" &>/dev/null; then
    echo "    Updating existing container app..."
    az containerapp secret set -g "${RG}" -n "${APP}" \
        --secrets "graph-mail-token=${TOKEN}"
    az containerapp update -g "${RG}" -n "${APP}" \
        --image "${FULL_IMAGE}"
else
    echo "    Creating new container app..."
    az containerapp create \
        -g "${RG}" \
        -n "${APP}" \
        --environment "${ENVIRONMENT}" \
        --image "${FULL_IMAGE}" \
        --registry-server "${REGISTRY}" \
        --target-port 8100 \
        --ingress external \
        --min-replicas 1 \
        --max-replicas 1 \
        --secrets "graph-mail-token=${TOKEN}" \
        --env-vars "GRAPH_MAIL_TOKEN=secretref:graph-mail-token"
fi

FQDN="$(az containerapp show -g "${RG}" -n "${APP}" --query "properties.configuration.ingress.fqdn" -o tsv)"
echo ""
echo "Done. MCP server URL: https://${FQDN}/mcp"
echo "Add to ToolingManifest.json or NAT config:"
echo "  url: https://${FQDN}/mcp"
