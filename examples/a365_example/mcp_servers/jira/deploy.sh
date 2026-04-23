#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build, push, and deploy the JIRA MCP server to Azure Container Apps.
#
# Usage (from this directory):
#   export JIRA_EMAIL="..."
#   export JIRA_API_TOKEN="..."
#   export JIRA_SITE="alexander-fournier.atlassian.net"
#   ./deploy.sh

set -euo pipefail

REGISTRY="${REGISTRY:-a365natdevacr1.azurecr.io}"
IMAGE_NAME="${IMAGE_NAME:-jira-mcp}"
TAG="${TAG:-$(date +%Y%m%d%H%M)}"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"

RG="${RG:-a365-nat-dev-rg-1}"
APP="${APP:-jira-mcp}"
ENVIRONMENT="${ACA_ENVIRONMENT:-$(az containerapp env list -g "${RG}" --query "[0].name" -o tsv)}"

JIRA_EMAIL="${JIRA_EMAIL:-${ATLASSIAN_EMAIL:-}}"
JIRA_API_TOKEN="${JIRA_API_TOKEN:-${ATLASSIAN_API_TOKEN:-}}"
JIRA_SITE="${JIRA_SITE:-${ATLASSIAN_SITE:-}}"

if [[ -z "${JIRA_EMAIL}" || -z "${JIRA_API_TOKEN}" || -z "${JIRA_SITE}" ]]; then
    echo "ERROR: set JIRA_EMAIL, JIRA_API_TOKEN, and JIRA_SITE" >&2
    exit 1
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
        --secrets "jira-api-token=${JIRA_API_TOKEN}"
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
        --target-port 8101 \
        --ingress external \
        --min-replicas 1 \
        --max-replicas 1 \
        --secrets "jira-api-token=${JIRA_API_TOKEN}" \
        --env-vars \
            "JIRA_EMAIL=${JIRA_EMAIL}" \
            "JIRA_API_TOKEN=secretref:jira-api-token" \
            "JIRA_SITE=${JIRA_SITE}"
fi

FQDN="$(az containerapp show -g "${RG}" -n "${APP}" --query "properties.configuration.ingress.fqdn" -o tsv)"
echo ""
echo "Done. MCP server URL: https://${FQDN}/mcp/"
