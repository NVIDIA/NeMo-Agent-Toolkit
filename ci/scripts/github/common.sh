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

GITHUB_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SCRIPT_DIR=$( dirname ${GITHUB_SCRIPT_DIR} )

source ${SCRIPT_DIR}/common.sh

install_rapids_gha_tools

# Ensure the workspace tmp directory exists
mkdir -p ${WORKSPACE_TMP}

rapids-logger "Environment Variables"
printenv | sort

function get_git_tag() {
    # Get the latest Git tag, sorted by version, excluding lightweight tags
    git describe --first-parent --tags --abbrev=0 2>/dev/null || echo "no-tag"
}
