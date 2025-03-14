#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Based on style.sh from Morpheus

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${SCRIPT_DIR}/common.sh

# Ignore errors and set path
set +e
LC_ALL=C.UTF-8
LANG=C.UTF-8

# Pre-populate the return values in case they are skipped
PYLINT_RETVAL=0

get_modified_files ${PYTHON_FILE_REGEX} MODIFIED_FILES

# When invoked by the git pre-commit hook CHANGED_FILES will already be defined
if [[ -n "${MODIFIED_FILES}" ]]; then
   echo -e "Running Python checks on ${#MODIFIED_FILES[@]} files:"

   for f in "${MODIFIED_FILES[@]}"; do
      echo "  $f"
   done

   if [[ "${SKIP_PYLINT}" == "" ]]; then
      PYLINT_OUTPUT=$(pylint ${MODIFIED_FILES[@]} 2>&1)
      PYLINT_RETVAL=$?
   fi

else
   echo "No modified Python files to check"
fi

if [[ "${SKIP_PYLINT}" != "" ]]; then
   echo -e "\n\n>>>> SKIPPED: pylint check\n\n"
elif [ "${PYLINT_RETVAL}" != "0" ]; then
   echo -e "\n\n>>>> FAILED: pylint style check; begin output\n\n"
   echo -e "${PYLINT_OUTPUT}"
   echo -e "\n\n>>>> FAILED: pylint style check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: pylint style check\n\n"
fi

RETVALS=(${PYLINT_RETVAL})
IFS=$'\n'
RETVAL=`echo "${RETVALS[*]}" | sort -nr | head -n1`

exit $RETVAL
