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

# Ignore errors
set +e
LC_ALL=C.UTF-8
LANG=C.UTF-8

python ${SCRIPT_DIR}/path_checks.py \
    --check-broken-symlinks \
    --check-paths-in-files

PATH_CHECKS_RETVAL=$?

if [[ "${PATH_CHECKS_RETVAL}" != "0" ]]; then
    echo ">>> FAILED: path checks"
else
    echo ">>> PASSED: path checks"
fi

exit ${PATH_CHECKS_RETVAL}
