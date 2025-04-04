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

case "$1" in
    "" )
        STAGES=("bash")
        ;;
    "all" )
        STAGES=("checks" "tests" "docs" "build_wheel")
        ;;
    "checks" | "tests" | "docs" | "build_wheel" | "bash" )
        STAGES=("$1")
        ;;
    * )
        echo "Error: Invalid argument \"$1\" provided. Expected values: \"all\", \"checks\", \"tests\", " \
             "\"docs\", \"build_wheel\" or \"bash\""
        exit 1
        ;;
esac

# Use the HTTPS URL to avoid needing to expose SSH_AUTH_SOCK to the container
function git_ssh_to_https()
{
    local url=$1
    echo $url | sed -e 's|^git@github\.com:|https://github.com/|'
}

CI_ARCH=${CI_ARCH:-$(dpkg --print-architecture)}
AGENTIQ_ROOT=${AGENTIQ_ROOT:-$(git rev-parse --show-toplevel)}

GIT_URL=$(git remote get-url origin)
GIT_URL=$(git_ssh_to_https ${GIT_URL})

GIT_UPSTREAM_URL=$(git remote get-url upstream)
GIT_UPSTREAM_URL=$(git_ssh_to_https ${GIT_UPSTREAM_URL})

GIT_BRANCH=$(git branch --show-current)
GIT_COMMIT=$(git log -n 1 --pretty=format:%H)

# Specifies whether to mount the current git repo or to use a clean clone (the default)
USE_HOST_GIT=${USE_HOST_GIT:-0}

LOCAL_CI_TMP=${LOCAL_CI_TMP:-${AGENTIQ_ROOT}/.tmp/local_ci_tmp/${CI_ARCH}}
DOCKER_EXTRA_ARGS=${DOCKER_EXTRA_ARGS:-""}

CI_CONTAINER=${CI_CONTAINER:-"ghcr.io/astral-sh/uv:python3.12-bookworm"}


# These variables are common to all stages
BASE_ENV_LIST="--env LOCAL_CI_TMP=/ci_tmp"
BASE_ENV_LIST="${BASE_ENV_LIST} --env GIT_URL=${GIT_URL}"
BASE_ENV_LIST="${BASE_ENV_LIST} --env GIT_UPSTREAM_URL=${GIT_UPSTREAM_URL}"
BASE_ENV_LIST="${BASE_ENV_LIST} --env GIT_BRANCH=${GIT_BRANCH}"
BASE_ENV_LIST="${BASE_ENV_LIST} --env GIT_COMMIT=${GIT_COMMIT}"
BASE_ENV_LIST="${BASE_ENV_LIST} --env USE_HOST_GIT=${USE_HOST_GIT}"

for STAGE in "${STAGES[@]}"; do
    # Take a copy of the base env list, then make stage specific changes
    ENV_LIST="${BASE_ENV_LIST}"

    mkdir -p ${LOCAL_CI_TMP}
    cp ${AGENTIQ_ROOT}/ci/scripts/bootstrap_local_ci.sh ${LOCAL_CI_TMP}

    DOCKER_RUN_ARGS="--rm -ti --net=host --platform=linux/${CI_ARCH} -v "${LOCAL_CI_TMP}":/ci_tmp ${ENV_LIST} --env STAGE=${STAGE}"

    if [[ "${USE_HOST_GIT}" == "1" ]]; then
        DOCKER_RUN_ARGS="${DOCKER_RUN_ARGS} -v ${AGENTIQ_ROOT}:/agentiq"
    fi

    if [[ "${STAGE}" == "bash" ]]; then
        DOCKER_RUN_CMD="bash --init-file /ci_tmp/bootstrap_local_ci.sh"
    else
        DOCKER_RUN_CMD="/ci_tmp/bootstrap_local_ci.sh"
    fi

    echo "Running ${STAGE} stage in ${CI_CONTAINER}"
    docker run ${DOCKER_RUN_ARGS} ${DOCKER_EXTRA_ARGS} ${CI_CONTAINER} ${DOCKER_RUN_CMD}

    STATUS=$?
    if [[ ${STATUS} -ne 0 ]]; then
        echo "Error: docker exited with a non-zero status code for ${STAGE} of ${STATUS}"
        exit ${STATUS}
    fi
done
