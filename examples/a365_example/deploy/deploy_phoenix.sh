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
#
# Deploy a lightweight Arize Phoenix server for NAT trace visualization.
# The HTTP trace endpoint is https://<fqdn>/v1/traces and the UI is https://<fqdn>.
#
# Set ACA_RG and optionally override:
#   ACA_ENVIRONMENT, PHOENIX_ACA_APP, PHOENIX_IMAGE, PHOENIX_CPU, PHOENIX_MEMORY

set -euo pipefail

RG="${ACA_RG:-}"
APP="${PHOENIX_ACA_APP:-phoenix-observability}"
IMAGE="${PHOENIX_IMAGE:-arizephoenix/phoenix:13.22}"
ENVIRONMENT="${ACA_ENVIRONMENT:-$(az containerapp env list -g "${RG}" --query "[0].name" -o tsv)}"
CPU="${PHOENIX_CPU:-1.0}"
MEMORY="${PHOENIX_MEMORY:-2Gi}"

if [[ -z "${RG}" ]]; then
  echo "ERROR: set ACA_RG" >&2
  exit 1
fi

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
