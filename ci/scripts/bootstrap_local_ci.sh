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

if [[ "${USE_HOST_GIT}" == "1" ]]; then
    cd aiqtoolkit/
    git config --global --add safe.directory /aiqtoolkit

    # Avoids SSH host key verification prompt
    ssh-keyscan github.com >> /etc/ssh/ssh_known_hosts
else
    git clone ${GIT_URL} aiqtoolkit
    cd aiqtoolkit/
    git remote add upstream ${GIT_UPSTREAM_URL}
    git fetch upstream
    git checkout develop
    git checkout ${GIT_BRANCH}
    git pull
    git checkout ${GIT_COMMIT}
    git fetch --all --tags

    export CURRENT_BRANCH=${GIT_BRANCH}
    export COMMIT_SHA=${GIT_COMMIT}
fi

export WORKSPACE=$(pwd)
export LOCAL_CI=1
export WORKSPACE_TMP="${LOCAL_CI_TMP}/local_ci_workspace"
export UV_CACHE_DIR="${LOCAL_CI_TMP}/cache/uv"
export PRE_COMMIT_HOME="${LOCAL_CI_TMP}/cache/pre_commit"
mkdir -p ${UV_CACHE_DIR}

GH_SCRIPT_DIR="${WORKSPACE}/ci/scripts/github"


if [[ "${STAGE}" != "bash" ]]; then

    CI_SCRIPT="${GH_SCRIPT_DIR}/${STAGE}.sh"

    ${CI_SCRIPT}
fi
