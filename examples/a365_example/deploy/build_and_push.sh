#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build the A365 bot image from the NeMo Agent Toolkit repo root and push to a registry.
#
# Usage (run from repository root):
#   ./examples/a365_example/deploy/build_and_push.sh <registry-login-server> [image_name] [tag]
#
# Examples:
#   ./examples/a365_example/deploy/build_and_push.sh myteam.azurecr.io
#   ./examples/a365_example/deploy/build_and_push.sh myteam.azurecr.io nat-a365-bot 20260327-1
#
# Prerequisites:
#   - Docker Desktop (or other daemon) running
#   - For Azure Container Registry: az login && az acr login -n <acr_short_name>
#     (script attempts az acr login when the server ends with .azurecr.io)
#
# Azure Container Apps expects linux/amd64. On Apple Silicon, the default image is arm64 and ACA will
# reject it ("no child with platform linux/amd64"). Override with DOCKER_PLATFORM=linux/arm64 only if
# you target arm64 in the cloud.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <registry-login-server> [image_name] [tag]" >&2
  echo "Example: $0 myacr.azurecr.io nat-a365-bot latest" >&2
  exit 1
fi

REGISTRY="$1"
IMAGE_NAME="${2:-nat-a365-bot}"
TAG="${3:-latest}"
FULL_IMAGE="${REGISTRY%/}/${IMAGE_NAME}:${TAG}"

# Common mistake: only two CLI args → tag defaults to "latest" while the intended tag
# was pasted into the image name, e.g. nat-a365-bot/mcp-foo → ACR repo
# "nat-a365-bot/mcp-foo" with tag "latest", but ACA_IMAGE uses "nat-a365-bot:mcp-foo"
# → MANIFEST_UNKNOWN on deploy.
if [[ $# -lt 3 && "${IMAGE_NAME}" == *"/"* ]]; then
  echo "ERROR: Pass the image tag as the third argument (do not put the tag in the image name)." >&2
  echo "  You passed only 2 arguments; tag defaulted to 'latest'." >&2
  echo "  Parsed repository: ${IMAGE_NAME}  tag: ${TAG}" >&2
  echo "  Example: $0 ${REGISTRY} nat-a365-bot mcp-$(date +%Y%m%d%H%M)" >&2
  exit 1
fi

cd "${REPO_ROOT}"

if [[ "${REGISTRY}" == *.azurecr.io ]]; then
  ACR_NAME="${REGISTRY%.azurecr.io}"
  echo "Logging in to ACR: ${ACR_NAME}"
  if command -v az >/dev/null 2>&1; then
    az acr login --name "${ACR_NAME}"
  else
    echo "WARN: az not found; run: az acr login -n ${ACR_NAME}" >&2
  fi
fi

PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
NAT_BUILD_STAMP="${NAT_BUILD_STAMP:-}"
if [[ -z "${NAT_BUILD_STAMP}" ]]; then
  NAT_BUILD_STAMP="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
fi
# BuildKit provenance/SBOM adds an extra manifest (unknown/unknown) to the OCI index; some Azure
# Container Apps versions then error with "no child with platform linux/amd64". Disable attestations.
echo "Build+push ${FULL_IMAGE} --platform=${PLATFORM} NAT_BUILD_STAMP=${NAT_BUILD_STAMP} (Dockerfile: ${DOCKERFILE})"
docker buildx build \
  --platform "${PLATFORM}" \
  --provenance=false \
  --sbom=false \
  --build-arg "NAT_BUILD_STAMP=${NAT_BUILD_STAMP}" \
  -f "${DOCKERFILE}" \
  -t "${FULL_IMAGE}" \
  --push \
  "${REPO_ROOT}"

echo "Done: ${FULL_IMAGE}"
