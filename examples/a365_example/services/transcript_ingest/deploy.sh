#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REGISTRY="${REGISTRY:-a365natdevacr1.azurecr.io}"
IMAGE_NAME="${IMAGE_NAME:-transcript-ingest}"
TAG="${TAG:-$(date +%Y%m%d%H%M)}"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"

RG="${RG:-a365-nat-dev-rg-1}"
APP="${APP:-transcript-ingest}"
ENVIRONMENT="${ACA_ENVIRONMENT:-$(az containerapp env list -g "${RG}" --query "[0].name" -o tsv)}"

required_env=(
  GRAPH_CLIENT_ID
  GRAPH_CLIENT_SECRET
  GRAPH_TENANT_ID
  GRAPH_TRANSCRIPT_NOTIFICATION_URL
  TRANSCRIPT_BLOB_CONNECTION_STRING
)

for var_name in "${required_env[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    echo "ERROR: missing required env ${var_name}" >&2
    exit 1
  fi
done

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
  az containerapp secret set -g "${RG}" -n "${APP}" --secrets \
    "graph-client-secret=${GRAPH_CLIENT_SECRET}" \
    "transcript-blob-connection-string=${TRANSCRIPT_BLOB_CONNECTION_STRING}"
  az containerapp update -g "${RG}" -n "${APP}" \
    --image "${FULL_IMAGE}" \
    --set-env-vars \
      "GRAPH_CLIENT_ID=${GRAPH_CLIENT_ID}" \
      "GRAPH_TENANT_ID=${GRAPH_TENANT_ID}" \
      "GRAPH_TRANSCRIPT_NOTIFICATION_URL=${GRAPH_TRANSCRIPT_NOTIFICATION_URL}" \
      "GRAPH_TRANSCRIPT_LIFECYCLE_URL=${GRAPH_TRANSCRIPT_LIFECYCLE_URL:-}" \
      "TRANSCRIPT_BLOB_CONTAINER=${TRANSCRIPT_BLOB_CONTAINER:-call-transcripts}"
else
  az containerapp create \
    -g "${RG}" \
    -n "${APP}" \
    --environment "${ENVIRONMENT}" \
    --image "${FULL_IMAGE}" \
    --registry-server "${REGISTRY}" \
    --target-port 8110 \
    --ingress external \
    --min-replicas 1 \
    --max-replicas 1 \
    --secrets \
      "graph-client-secret=${GRAPH_CLIENT_SECRET}" \
      "transcript-blob-connection-string=${TRANSCRIPT_BLOB_CONNECTION_STRING}" \
    --env-vars \
      "GRAPH_CLIENT_ID=${GRAPH_CLIENT_ID}" \
      "GRAPH_CLIENT_SECRET=secretref:graph-client-secret" \
      "GRAPH_TENANT_ID=${GRAPH_TENANT_ID}" \
      "GRAPH_TRANSCRIPT_NOTIFICATION_URL=${GRAPH_TRANSCRIPT_NOTIFICATION_URL}" \
      "GRAPH_TRANSCRIPT_LIFECYCLE_URL=${GRAPH_TRANSCRIPT_LIFECYCLE_URL:-}" \
      "TRANSCRIPT_BLOB_CONNECTION_STRING=secretref:transcript-blob-connection-string" \
      "TRANSCRIPT_BLOB_CONTAINER=${TRANSCRIPT_BLOB_CONTAINER:-call-transcripts}"
fi

FQDN="$(az containerapp show -g "${RG}" -n "${APP}" --query "properties.configuration.ingress.fqdn" -o tsv)"
echo "Done. Transcript ingest URL: https://${FQDN}"
