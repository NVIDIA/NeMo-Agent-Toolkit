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

install_rapids_gha_tools

export NAT_AVOID_GH_CLI=1 # gh cli not working with gitlab, todo look into seeing if this can be fixed

function get_git_tag() {
    FT=$(git fetch --all --tags)

    # Get the latest Git tag, sorted by version, excluding lightweight tags
    GIT_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "no-tag")

    if [[ "${CI_CRON_NIGHTLY}" == "1" ]]; then
        if [[ ${GIT_TAG} == "no-tag" ]]; then
            rapids-logger "Error: No tag found. Exiting."
            exit 1;
        fi

        # If the branch is a nightly build create a version which will be accepted by pypi,
        # The sed script here is splitting on either the first dash or 'a',
        # transforming a tag like `v1.3.0-dev-17-g7681cf9f` into `v1.3.0a20250821`
        # and a tag like `v1.3.0a5` into `v1.3.0a20250821`
        # Note: We are intentionally not ceating an actual tag, just setting the variable
        GIT_TAG=$(echo $GIT_TAG | sed -E -e "s/(-|a).*/a$(date +"%Y%m%d")/")
    fi

    echo ${GIT_TAG}
}

function is_current_commit_release_tagged() {
    # Check if the current commit is tagged for release, either an RC tag or the release tag
    set +e
    GIT_TAG=$(git describe --tags --exact-match HEAD 2>/dev/null)
    local status_code=$?
    set -e

    # Convert the unix status code to a boolean value
    local is_tagged=0
    if [[ ${status_code} -eq 0 ]]; then
        local is_pre_release=0

        # Ensure we don't have a dev or alpha tag
        if [[ ! (${GIT_TAG} =~ "-dev" || ${GIT_TAG} =~ "a") ]]; then
            is_tagged=1
        fi
    fi
    echo ${is_tagged}
}

rapids-logger "Environment Variables"
printenv | sort
