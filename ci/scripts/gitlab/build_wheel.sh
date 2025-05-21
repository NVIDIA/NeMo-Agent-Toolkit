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

set -e

GITLAB_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

source ${GITLAB_SCRIPT_DIR}/common.sh

GIT_TAG=$(get_git_tag)
IS_TAGGED=$(is_current_commit_tagged)
rapids-logger "Git Version: ${GIT_TAG} - Is Tagged: ${IS_TAGGED}"

if [[ "${CI_CRON_NIGHTLY}" == "1" || ( ${IS_TAGGED} == "1" && "${CI_COMMIT_BRANCH}" != "main" ) ]]; then
    export SETUPTOOLS_SCM_PRETEND_VERSION="${GIT_TAG}"
    export USE_FULL_VERSION="1"
fi

if [[ "${CI_CRON_NIGHTLY}" == "1" ]]; then
    create_env group:dev
    export SKIP_MD_UPDATE=1
    ${PROJECT_ROOT}/ci/release/update-version.sh "${GIT_TAG}"
fi

WHEELS_BASE_DIR="${CI_PROJECT_DIR}/.tmp/wheels"
WHEELS_DIR="${WHEELS_BASE_DIR}/aiqtoolkit"

create_env extra:all

build_wheel . "aiqtoolkit/${GIT_TAG}"


# Build all examples with a pyproject.toml in the first directory below examples
for AIQ_EXAMPLE in ${AIQ_EXAMPLES[@]}; do
    # places all wheels flat under example
    build_wheel ${AIQ_EXAMPLE} "examples"
done

# Build all packages with a pyproject.toml in the first directory below packages
for AIQ_PACKAGE in "${AIQ_PACKAGES[@]}"; do
    build_package_wheel ${AIQ_PACKAGE}
done

if [[ "${BUILD_AIQ_COMPAT}" == "true" ]]; then
    WHEELS_DIR="${WHEELS_BASE_DIR}/agentiq"
    for AIQ_COMPAT_PACKAGE in "${AIQ_COMPAT_PACKAGES[@]}"; do
        build_package_wheel ${AIQ_COMPAT_PACKAGE}
    done
fi

if [[ "${CI_COMMIT_BRANCH}" == "${CI_DEFAULT_BRANCH}" || "${CI_COMMIT_BRANCH}" == "main" ]]; then
    rapids-logger "Uploading Wheels"

    # Find and upload all .whl files from nested directories
    while read -r WHEEL_FILE; do
        echo "Uploading ${WHEEL_FILE}..."

        python -m twine upload \
            -u gitlab-ci-token \
            -p "${CI_JOB_TOKEN}" \
            --non-interactive \
            --repository-url "${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/packages/pypi" \
            "${WHEEL_FILE}"
    done < <(find "${WHEELS_BASE_DIR}" -type f -name "*.whl")
fi
