#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Deploy a lightweight Arize Phoenix server for NAT trace visualization.
# The HTTP trace endpoint is https://<fqdn>/v1/traces and the UI is https://<fqdn>.
#
# Defaults match the A365 demo resource group. Override with:
#   ACA_RG, ACA_ENVIRONMENT, PHOENIX_ACA_APP, PHOENIX_IMAGE, PHOENIX_CPU, PHOENIX_MEMORY

set -euo pipefail

RG="${ACA_RG:-a365-nat-dev-rg-1}"
APP="${PHOENIX_ACA_APP:-phoenix-observability}"
IMAGE="${PHOENIX_IMAGE:-arizephoenix/phoenix:13.22}"
ENVIRONMENT="${ACA_ENVIRONMENT:-$(az containerapp env list -g "${RG}" --query "[0].name" -o tsv)}"
CPU="${PHOENIX_CPU:-1.0}"
MEMORY="${PHOENIX_MEMORY:-2Gi}"

if [[ -z "${ENVIRONMENT}" ]]; then
  echo "Could not determine Container Apps environment. Set ACA_ENVIRONMENT." >&2
  exit 1
fi

if az containerapp show -g "${RG}" -n "${APP}" &>/dev/null; then
  echo "Updating ${APP} to ${IMAGE}"
  az containerapp update \
    -g "${RG}" \
    -n "${APP}" \
    --image "${IMAGE}" \
    --cpu "${CPU}" \
    --memory "${MEMORY}" \
    --min-replicas 1
else
  echo "Creating ${APP} in ${RG}/${ENVIRONMENT} from ${IMAGE}"
  az containerapp create \
    -g "${RG}" \
    -n "${APP}" \
    --environment "${ENVIRONMENT}" \
    --image "${IMAGE}" \
    --ingress external \
    --target-port 6006 \
    --transport auto \
    --cpu "${CPU}" \
    --memory "${MEMORY}" \
    --min-replicas 1
fi

FQDN="$(az containerapp show -g "${RG}" -n "${APP}" --query "properties.configuration.ingress.fqdn" -o tsv)"

echo "Phoenix UI:       https://${FQDN}"
echo "Phoenix endpoint: https://${FQDN}/v1/traces"
echo
echo "Use this when rolling out the bot:"
echo "  export PHOENIX_ENDPOINT=https://${FQDN}/v1/traces"
