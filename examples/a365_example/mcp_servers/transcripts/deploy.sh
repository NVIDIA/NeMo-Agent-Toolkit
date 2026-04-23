#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REGISTRY="${REGISTRY:-a365natdevacr1.azurecr.io}"
IMAGE_NAME="${IMAGE_NAME:-transcript-mcp}"
TAG="${TAG:-$(date +%Y%m%d%H%M)}"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"

RG="${RG:-a365-nat-dev-rg-1}"
APP="${APP:-transcript-mcp}"
ENVIRONMENT="${ACA_ENVIRONMENT:-$(az containerapp env list -g "${RG}" --query "[0].name" -o tsv)}"

if [[ -z "${TRANSCRIPT_BLOB_CONNECTION_STRING:-}" ]]; then
  echo "ERROR: missing TRANSCRIPT_BLOB_CONNECTION_STRING" >&2
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
  az containerapp secret set -g "${RG}" -n "${APP}" \
    --secrets "transcript-blob-connection-string=${TRANSCRIPT_BLOB_CONNECTION_STRING}"
  az containerapp update -g "${RG}" -n "${APP}" --image "${FULL_IMAGE}"
else
  az containerapp create \
    -g "${RG}" \
    -n "${APP}" \
    --environment "${ENVIRONMENT}" \
    --image "${FULL_IMAGE}" \
    --registry-server "${REGISTRY}" \
    --target-port 8111 \
    --ingress external \
    --min-replicas 1 \
    --max-replicas 1 \
    --secrets "transcript-blob-connection-string=${TRANSCRIPT_BLOB_CONNECTION_STRING}" \
    --env-vars \
      "TRANSCRIPT_BLOB_CONNECTION_STRING=secretref:transcript-blob-connection-string" \
      "TRANSCRIPT_BLOB_CONTAINER=${TRANSCRIPT_BLOB_CONTAINER:-call-transcripts}"
fi

FQDN="$(az containerapp show -g "${RG}" -n "${APP}" --query "properties.configuration.ingress.fqdn" -o tsv)"
echo "Done. Transcript MCP URL: https://${FQDN}/mcp"
