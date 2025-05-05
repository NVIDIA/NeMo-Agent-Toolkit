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

## Usage
# Either supply full versions:
#    `bash update-version.sh <new_version>`
#    Format is <maj>.<minor>.<patch> - no leading 'v'

set -e

# If the user has not supplied the versions, determine them from the git tags
if [[ "$#" -ne 1 ]]; then
   echo "No versions were provided."
   exit 1;
else
   NEXT_VERSION=$2
fi

NEXT_MAJOR=$(echo ${NEXT_VERSION} | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo ${NEXT_VERSION} | awk '{split($0, a, "."); print a[2]}')
NEXT_PATCH=$(echo ${NEXT_VERSION} | awk '{split($0, a, "."); print a[3]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}

# Inplace sed replace; workaround for Linux and Mac. Accepts multiple files
function sed_runner() {

   pattern=$1
   shift

   for f in $@ ; do
      sed -i.bak ''"$pattern"'' "$f" && rm -f "$f.bak"
   done
}

if [[ "${USE_FULL_VERSION}" == "1" ]]; then
   AIQ_VERSION=${NEXT_VERSION}
else
   AIQ_VERSION=${NEXT_SHORT_TAG}
fi


# Update the dependencies that the examples and packages depend on aiqtoolkit, we are explicitly specifying the
# `examples` and `packages` directories in order to avoid accidentally updating toml files of third-party packages in
# the `.venv` directory, and updating the root pyproject.toml file
AIQ_PACKAGE_TOMLS=$(find ./packages -name "pyproject.toml")
AIQ_EXAMPLE_TOMLS=$(find ./examples -name "pyproject.toml")
sed_runner "s|"aiqtoolkit\(\[\w*\]\)==.*"|"aiqtoolkit\1==${AIQ_VERSION}"|g" \
   ${AIQ_PACKAGE_TOMLS} \
   ${AIQ_EXAMPLE_TOMLS}
