#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

GITLAB_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

source ${GITLAB_SCRIPT_DIR}/common.sh

GIT_TAG=$(get_git_tag)
IS_TAGGED=$(is_current_commit_release_tagged)
echo "Git Version: ${GIT_TAG} - Is Tagged: ${IS_TAGGED}"

# Define variables
NAT_ARCH="any"
NAT_OS="any"

# nvidia-nat itself and all of the plugins are under "nvidia-nat", while the compatibility packages are under "nat"
NAT_COMPONENTS=("nvidia-nat" "nat")

# We need to fix the name of the component in artifactory to aiqtoolkit
ARTIFACTORY_COMPONENT_FIXED_NAME="aiqtoolkit"

WHEELS_BASE_DIR="${CI_PROJECT_DIR}/.tmp/wheels"

# Define the subdirectories to be exclude
EXCLUDE_SUBDIRS=("examples")

# Exit if required secrets are not set
if [[ -z "${URM_USER}" || -z "${URM_API_KEY}" ]]; then
    echo "Error: URM_USER or URM_API_KEY is not set. Exiting."
    exit 1
fi

if [[ -z "${NAT_ARTIFACTORY_URL}" || -z "${NAT_ARTIFACTORY_NAME}" ]]; then
    echo "Error: NAT_ARTIFACTORY_URL or NAT_ARTIFACTORY_NAME is not set. Exiting."
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
if [[ ! -d "$WHEELS_BASE_DIR" || -z "$(find "$WHEELS_BASE_DIR" -type f -name "*.whl" 2>/dev/null)" ]]; then
    echo "No wheels found in $WHEELS_BASE_DIR or its subdirectories. Exiting."
    exit 1
fi

# Upload wheels if enabled
if [[ "${UPLOAD_TO_ARTIFACTORY}" == "true" ]]; then
    echo "Uploading wheels to Artifactory (${NAT_ARTIFACTORY_NAME}) for ${GIT_TAG}..."
    python ${GITLAB_SCRIPT_DIR}/artifactory_upload.py
else
    echo "UPLOAD_TO_ARTIFACTORY is set to 'false'. Skipping upload."
fi
