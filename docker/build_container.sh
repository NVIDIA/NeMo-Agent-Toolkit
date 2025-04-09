#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Get the path to REPO_ROOT without altering the docker context (in case we are in a submodule)
pushd ${SCRIPT_DIR} &> /dev/null
export REPO_ROOT=${REPO_ROOT:-"$(git rev-parse --show-toplevel)"}
popd &> /dev/null

HOST_ARCH=$(dpkg --print-architecture)
DOCKER_TARGET_ARCH=${DOCKER_TARGET_ARCH:-${HOST_ARCH}}

if [ ${DOCKER_TARGET_ARCH} != ${HOST_ARCH} ]; then
    echo -n "Performing cross-build for ${DOCKER_TARGET_ARCH} on ${HOST_ARCH}, please ensure qemu is installed, "
    echo "details in ${REPO_ROOT}/docs/source/advanced/running-ci-locally.md"
fi

git describe --tags --abbrev=0 2>/dev/null || echo "no-tag"

DOCKER_IMAGE_NAME=${DOCKER_IMAGE_NAME:-"agentiq"}
DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG:-"$(git describe --tags --abbrev=0 2> /dev/null)"}

DOCKER_EXTRA_ARGS=${DOCKER_EXTRA_ARGS:-""}

# Build the docker arguments
DOCKER_ARGS="-t ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}"
DOCKER_ARGS="${DOCKER_ARGS} --platform=linux/${DOCKER_TARGET_ARCH}"
DOCKER_ARGS="${DOCKER_ARGS} --network=host"

# Last add any extra args (duplicates override earlier ones)
DOCKER_ARGS="${DOCKER_ARGS} ${DOCKER_EXTRA_ARGS}"

# Export buildkit variable
export DOCKER_BUILDKIT=1

echo "Building ${DOCKER_IMAGE_NAME}:${DOCKER_TAG} with args..."
echo ""
echo "   COMMAND: docker build ${DOCKER_ARGS} -f ${SCRIPT_DIR}/Dockerfile ."
echo "   Note: add '--progress plain' to DOCKER_EXTRA_ARGS to show all container build output"

docker build ${DOCKER_ARGS} -f ${SCRIPT_DIR}/Dockerfile .
