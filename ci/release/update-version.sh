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
   NEXT_VERSION=$1
fi

export CUR_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# The root to the NAT repo
export PROJECT_ROOT=${PROJECT_ROOT:-"$(realpath ${CUR_DIR}/../..)"}

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

# Update the pypi description file
# Currently only the pypi.md file for the nvidia-nat package contains links to documentation
# Replace this with a `find ./ -name "pypi.md"` if this is needed for the other pypi.md files
if [[ -z "${SKIP_MD_UPDATE}" ]]; then
   sed_runner "s|https:\/\/docs.nvidia.com\/nemo\/agent-toolkit\/\([0-9|\.]\+\)|https:\/\/docs.nvidia.com\/nemo\/agent-toolkit\/${NEXT_VERSION}|g" src/nat/meta/pypi.md
fi



if [[ "${USE_FULL_VERSION}" == "1" ]]; then
   NAT_VERSION=${NEXT_VERSION}
   VERSION_MATCH="=="
else
   NAT_VERSION=${NEXT_SHORT_TAG}
   VERSION_MATCH="~="
fi

# Change directory to the repo root
pushd "${PROJECT_ROOT}" &> /dev/null

# Update the dependencies that the examples and packages depend on nvidia-nat, we are explicitly specifying the
# `examples` and `packages` directories in order to avoid accidentally updating toml files of third-party packages in
# the `.venv` directory, and updating the root pyproject.toml file. The sort is not really needed, but it makes the
# output deterministic and easier to read.
NAT_PACKAGE_TOMLS=($(find ./packages -name "pyproject.toml" | sort ))
NAT_EXAMPLE_TOMLS=($(find ./examples -name "pyproject.toml" | sort ))

for TOML_FILE in ${NAT_EXAMPLE_TOMLS[@]}; do
    ${CUR_DIR}/update_toml_dep.py \
      --toml-file-path=${TOML_FILE} \
      --new-version="${NAT_VERSION}" \
      --version-match="${VERSION_MATCH}"
done

for TOML_FILE in "${NAT_PACKAGE_TOMLS[@]}"; do
    ${CUR_DIR}/update_toml_dep.py \
      --toml-file-path=${TOML_FILE} \
      --new-version="${NAT_VERSION}" \
      --version-match="${VERSION_MATCH}"
done
