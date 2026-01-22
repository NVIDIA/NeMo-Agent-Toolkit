#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

GITHUB_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

source ${GITHUB_SCRIPT_DIR}/common.sh
WHEELS_BASE_DIR="${WORKSPACE_TMP}/wheels"
WHEELS_DIR="${WHEELS_BASE_DIR}/nvidia-nat"

GIT_TAG=$(get_git_tag)
rapids-logger "Git Version: ${GIT_TAG}"

create_env group:dev extra:all

# Update internal dependencies to the current git tag
set_versions

build_wheel . "nvidia-nat/${GIT_TAG}"


# Build all examples with a pyproject.toml in the first directory below examples
for NAT_EXAMPLE in ${NAT_EXAMPLES[@]}; do
    # places all wheels flat under example
    build_wheel ${NAT_EXAMPLE} "examples"
done


# Build all packages with a pyproject.toml in the first directory below packages
for NAT_PACKAGE in "${NAT_PACKAGES[@]}"; do
    build_package_wheel ${NAT_PACKAGE}
done

if [[ "${BUILD_NAT_COMPAT}" == "true" ]]; then
    WHEELS_DIR="${WHEELS_BASE_DIR}/nat"
    for NAT_COMPAT_PACKAGE in "${NAT_COMPAT_PACKAGES[@]}"; do
        build_package_wheel ${NAT_COMPAT_PACKAGE}
    done
fi

# Flatten out the wheels into a single directory for upload
BUILT_WHEELS=$(find "${WHEELS_BASE_DIR}"/**/ -type f -name "*.whl")
MOVED_WHEELS=()
for whl in ${BUILT_WHEELS}; do
    dest_wheel_name="${WHEELS_BASE_DIR}/$(basename "${whl}")"
    mv "${whl}" "${dest_wheel_name}"
    MOVED_WHEELS+=("${dest_wheel_name}")
done


# Test the built wheels
deactivate
TEMP_INSTALL_LOCATION="${WORKSPACE_TMP}/wheel_test_env"

PYTHON_VERSIONS_TO_TEST=("3.11" "3.12" "3.13")
for pyver in "${PYTHON_VERSIONS_TO_TEST[@]}"; do
    set +e
    # The managed python flag is needed since the OS's copy of python does not include C headers needed to build some
    # dependencies, specifically ruamel-yaml-clibz which is needed for semantic-kernel
    uv python find --managed-python ${pyver} &> /dev/null
    PYTHON_FIND_RESULT=$?
    set -e
    if [[ ${PYTHON_FIND_RESULT} -ne 0 ]]; then
        rapids-logger "Downloading Python version ${pyver}"

        # In common.sh we set this to never, we want to override that here
        UV_PYTHON_DOWNLOADS="manual" uv python install --managed-python ${pyver}
    fi
done

for whl in "${MOVED_WHEELS[@]}"; do

    for pyver in "${PYTHON_VERSIONS_TO_TEST[@]}"; do
        rapids-logger "Testing wheel: ${whl} with Python ${pyver}"
        UV_VENV_OUT=$(uv venv -p ${pyver} --seed "${TEMP_INSTALL_LOCATION}" 2>&1)
        UV_VENV_RESULT=$?

        if [[ ${UV_VENV_RESULT} -ne 0 ]]; then
            rapids-logger "Error, failed to create uv venv with Python ${pyver} for wheel ${whl} : ${UV_VENV_OUT}"
            exit ${UV_VENV_RESULT}
        fi

        source "${TEMP_INSTALL_LOCATION}/bin/activate"

        set +e
        UV_PIP_OUT=$(uv pip install --prerelease=allow --find-links "${WHEELS_BASE_DIR}" "${whl}" 2>&1)
        INSTALL_RESULT=$?

        if [[ ${INSTALL_RESULT} -ne 0 ]]; then
            rapids-logger "Error, failed to install wheel ${whl} with Python ${pyver} : ${UV_PIP_OUT}"
            exit ${INSTALL_RESULT}
        fi

        # run a simple command to verify installation
        PYTHON_IMPORT_OUT=$(python -c "import nat" 2>&1)
        IMPORT_TEST_RESULT=$?

       if [[ ${IMPORT_TEST_RESULT} -ne 0 ]]; then
            rapids-logger "Error, failed to import nat from wheel ${whl} with Python ${pyver} : ${PYTHON_IMPORT_OUT}"
            exit ${IMPORT_TEST_RESULT}
        fi

        REPORTED_VERSION=$(nat --version 2>&1)
        NAT_CMD_EXIT_CODE=$?

        if [[ ${NAT_CMD_EXIT_CODE} -ne 0 ]]; then
            rapids-logger "Error 'nat --version' command failed exit code ${NAT_CMD_EXIT_CODE} from wheel ${whl} with Python ${pyver} : ${REPORTED_VERSION}"
            exit ${NAT_CMD_EXIT_CODE}
        fi

        set -e
        deactivate
        rm -rf "${TEMP_INSTALL_LOCATION}"
    done
done
