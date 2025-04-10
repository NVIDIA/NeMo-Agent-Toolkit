#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Exit on error
set -e

# change this to ready to publish. this should be done programmatically once
# the release process is finalized.
if [[ "${CI_CRON_NIGHTLY}" == "true" || "${CI_COMMIT_BRANCH}" == "main" ]]; then
    RELEASE_STATUS=ready
else
    RELEASE_STATUS=preview
fi

# Define variables
AIQ_ARCH="any"
AIQ_OS="any"
AIQ_COMPONENT_NAME="agentiq"

WHEELS_DIR="${CI_PROJECT_DIR}/.tmp/wheels"
# Define the subdirectories to be exclude
EXCLUDE_SUBDIRS=("examples")
COMPONENT_NAME="agentiq"

# Exit if required secrets are not set
if [[ -z "${URM_USER}" || -z "${URM_API_KEY}" ]]; then
    echo "Error: URM_USER or URM_API_KEY is not set. Exiting."
    exit 1
fi

if [[ -z "${AIQ_ARTIFACTORY_URL}" || -z "${AIQ_ARTIFACTORY_NAME}" ]]; then
    echo "Error: AIQ_ARTIFACTORY_URL or AIQ_ARTIFACTORY_NAME is not set. Exiting."
    exit 1
fi

if [[ -z "${RELEASE_APPROVER}" ]]; then
    echo "Error: RELEASE_APPROVER is not set. Exiting."
    exit 1
fi

# Artifactory upload settings
UPLOAD_TO_ARTIFACTORY=${UPLOAD_TO_ARTIFACTORY:-true}
LIST_ARTIFACTORY_CONTENTS=${LIST_ARTIFACTORY_CONTENTS:-false}


# Exit early if neither upload nor listing is needed
if [[ "${UPLOAD_TO_ARTIFACTORY}" != "true" && "${LIST_ARTIFACTORY_CONTENTS}" != "true" ]]; then
    echo "Neither UPLOAD_TO_ARTIFACTORY nor LIST_ARTIFACTORY_CONTENTS is enabled."
    exit 0
fi

# Ensure wheels exist before uploading (including subdirectories)
if [[ ! -d "$WHEELS_DIR" || -z "$(find "$WHEELS_DIR" -type f -name "*.whl" 2>/dev/null)" ]]; then
    echo "No wheels found in $WHEELS_DIR or its subdirectories. Exiting."
    exit 1
fi

# Function to install JFrog CLI if needed
function install_jfrog_cli() {
    if ! command -v jf &> /dev/null; then
        echo "Installing JFrog CLI..."
        curl -fL https://install-cli.jfrog.io | sh || { echo "JFrog CLI installation failed"; exit 1; }
    fi
}
install_jfrog_cli

GIT_TAG=$(get_git_tag)
rapids-logger "Git Version: ${GIT_TAG}"

# Upload wheels if enabled
if [[ "${UPLOAD_TO_ARTIFACTORY}" == "true" ]]; then
    for SUBDIR in $(find "${WHEELS_DIR}" -mindepth 1 -maxdepth 1 -type d); do
        SUBDIR_NAME=$(basename "${SUBDIR}")

        # Skip directories listed in EXCLUDE_SUBDIRS
        if [[ " ${EXCLUDE_SUBDIRS[@]} " =~ " ${SUBDIR_NAME} " ]]; then
            echo "Skipping excluded directory: ${SUBDIR_NAME}"
            continue
        fi

        echo "Uploading wheels from ${SUBDIR} to Artifactory..."

        # Find all .whl files in the current subdirectory (no depth limit)
        find "${SUBDIR}" -type f -name "*.whl" | while read -r WHEEL_FILE; do
            # Extract relative path to preserve directory structure
            RELATIVE_PATH="${WHEEL_FILE#${WHEELS_DIR}/}"
            ARTIFACTORY_PATH="${AIQ_ARTIFACTORY_NAME}/${RELATIVE_PATH}"

            echo "Uploading ${WHEEL_FILE} to ${ARTIFACTORY_PATH}..."

            CI=true jf rt u --fail-no-op --url="${AIQ_ARTIFACTORY_URL}" \
                --user="${URM_USER}" --password="${URM_API_KEY}" \
                --flat=false "${WHEEL_FILE}" "${ARTIFACTORY_PATH}" \
                --target-props "arch=${AIQ_ARCH};os=${AIQ_OS};branch=${GIT_TAG};component_name=${AIQ_COMPONENT_NAME};version=${GIT_TAG};release_approver=${RELEASE_APPROVER};release_status=${RELEASE_STATUS}"
        done
    done
else
    echo "UPLOAD_TO_ARTIFACTORY is set to 'false'. Skipping upload."
fi

# List Artifactory contents (disabled by default as the output is very verbose)
if [[ "${LIST_ARTIFACTORY_CONTENTS}" == "true" ]]; then
    echo "Listing contents of Artifactory (${AIQ_ARTIFACTORY_NAME}):"
    CI=true jf rt s --url="${AIQ_ARTIFACTORY_URL}" \
        --user="${URM_USER}" --password="${URM_API_KEY}" \
        "${AIQ_ARTIFACTORY_NAME}/*/${GIT_TAG}/" --recursive
fi
