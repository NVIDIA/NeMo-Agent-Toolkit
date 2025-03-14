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
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${SCRIPT_DIR}/common.sh

set +e
pre-commit run --all-files --show-diff-on-failure
PRE_COMMIT_RETVAL=$?

${SCRIPT_DIR}/python_checks.sh
PY_CHECKS_RETVAL=$?

if [[ ${PRE_COMMIT_RETVAL} -ne 0 || ${PY_CHECKS_RETVAL} -ne 0 ]]; then
   echo ">>>> FAILED: checks"
   exit 1
fi
