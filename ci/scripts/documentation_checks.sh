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

set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${SCRIPT_DIR}/common.sh

set +e

# Intentionally excluding CHANGELOG.md as it immutable
DOC_FILES=$(git ls-files "*.md" "*.rst" | grep -v -E '^(CHANGELOG|LICENSE)\.md$' | grep -v -E '^nv_internal/')

echo "Running spelling and grammer checks with Vale"
vale ${DOC_FILES}
VALE_RETVAL=$?

echo -e "\nRunning link checks with linkspector"
linkspector check -c ${PROJECT_ROOT}/ci/linkspector.yml
LINK_RETVAL=$?

if [[ ${PRE_COMMIT_VALE_RETVALRETVAL} -ne 0 || ${LINK_RETVAL} -ne 0 ]]; then
   echo ">>>> FAILED: checks"
   exit 1
fi
