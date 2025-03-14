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

GITLAB_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SCRIPT_DIR=$( dirname ${GITLAB_SCRIPT_DIR} )

source ${SCRIPT_DIR}/common.sh

AIQ_EXAMPLES=($(find ./examples/ -maxdepth 2 -name "pyproject.toml" | sort | xargs dirname))
AIQ_PACKAGES=($(find ./packages/ -maxdepth 2 -name "pyproject.toml" | sort | xargs dirname))


function create_env() {

    extras=()
    for arg in "$@"; do
        if [[ "${arg}" == "extra:all" ]]; then
            extras+=("--all-extras")
        elif [[ "${arg}" == "group:all" ]]; then
            extras+=("--all-groups")
        elif [[ "${arg}" == extra:* ]]; then
            extras+=("--extra" "${arg#extra:}")
        elif [[ "${arg}" == group:* ]]; then
            extras+=("--group" "${arg#group:}")
        else
            # Error out if we don't know what to do with the argument
            rapids-logger "Unknown argument to create_env: ${arg}. Must start with 'extra:' or 'group:'"
            exit 1
        fi
    done

    rapids-logger "Creating Environment with extras: ${@}"

    UV_SYNC_STDERROUT=$(uv sync ${extras[@]} 2>&1)

    # Environment should have already been created in the before_script
    if [[ "${UV_SYNC_STDERROUT}" =~ "warning:" ]]; then
        echo "Error, uv sync emitted warnings. These are usually due to missing lower bound constraints."
        echo "StdErr output:"
        echo "${UV_SYNC_STDERROUT}"
        exit 1
    fi

    rapids-logger "Final Environment"
    uv pip list
}

rapids-logger "Environment Variables"
printenv | sort
