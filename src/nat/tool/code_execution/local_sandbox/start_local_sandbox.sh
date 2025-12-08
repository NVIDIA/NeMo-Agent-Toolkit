#!/bin/bash

# Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Usage: ./start_local_sandbox.sh [SANDBOX_NAME] [OUTPUT_DATA_PATH]
# NOTE: needs to run from the root of the repo!

DOCKER_COMMAND=${DOCKER_COMMAND:-"docker"}
SANDBOX_NAME=${1:-'local-sandbox'}

# UWSGI_CHEAPER sets the number of initial uWSGI worker processes
# UWSGI_PROCESSES sets the maximum number of uWSGI worker processes
UWSGI_CHEAPER=${UWSGI_CHEAPER:-5}
UWSGI_PROCESSES=${UWSGI_PROCESSES:-10}

# Get the output_data directory path for mounting
# Priority: command line argument > environment variable > default path (current directory)
OUTPUT_DATA_PATH=${2:-${OUTPUT_DATA_PATH:-$(pwd)}}

echo "Starting sandbox with container name: ${SANDBOX_NAME}"
echo "Mounting output_data directory: ${OUTPUT_DATA_PATH}"

# Verify the path exists before mounting, create if it doesn't
if [ ! -d "${OUTPUT_DATA_PATH}" ]; then
    echo "Output data directory does not exist, creating: ${OUTPUT_DATA_PATH}"
    mkdir -p "${OUTPUT_DATA_PATH}"
fi

# Check if the Docker image already exists
if ! ${DOCKER_COMMAND} images ${SANDBOX_NAME} | grep -q "${SANDBOX_NAME}"; then
    echo "Docker image not found locally. Building ${SANDBOX_NAME}..."
    ${DOCKER_COMMAND} build --tag=${SANDBOX_NAME} \
        --build-arg="UWSGI_PROCESSES=${UWSGI_PROCESSES}" \
        --build-arg="UWSGI_CHEAPER=${UWSGI_CHEAPER}" \
        -f Dockerfile.sandbox .
else
    echo "Using existing Docker image: ${SANDBOX_NAME}"
fi

# Mount the output_data directory directly so files created in container appear in the local directory
${DOCKER_COMMAND} run --rm -ti --name=local-sandbox \
  --network=host \
  -v "${OUTPUT_DATA_PATH}:/workspace" \
  ${SANDBOX_NAME}
