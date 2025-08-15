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

export SCRIPT_DIR=${SCRIPT_DIR:-"$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"}

# The root to the NAT repo
export PROJECT_ROOT=${PROJECT_ROOT:-"$(realpath ${SCRIPT_DIR}/../..)"}

export PY_ROOT="${PROJECT_ROOT}/src"
export PROJ_TOML="${PROJECT_ROOT}/pyproject.toml"
export PY_DIRS="${PY_ROOT} ${PROJECT_ROOT}/packages ${PROJECT_ROOT}/tests ${PROJECT_ROOT}/ci/scripts "

# Determine the commits to compare against. If running in CI, these will be set. Otherwise, diff with main
export NAT_LOG_LEVEL=WARN
export CI_MERGE_REQUEST_TARGET_BRANCH_NAME=${CI_MERGE_REQUEST_TARGET_BRANCH_NAME:-"develop"}

if [[ "${GITLAB_CI}" == "true" ]]; then
   export BASE_SHA=${BASE_SHA:-${CI_MERGE_REQUEST_TARGET_BRANCH_SHA:-${CI_MERGE_REQUEST_DIFF_BASE_SHA:-$(${SCRIPT_DIR}/gitutils.py get_merge_target --current-branch=${CURRENT_BRANCH})}}}
   export COMMIT_SHA=${CI_COMMIT_SHA:-${COMMIT_SHA:-HEAD}}
else
   export BASE_SHA=${BASE_SHA:-$(${SCRIPT_DIR}/gitutils.py get_merge_target)}
   export COMMIT_SHA=${COMMIT_SHA:-${GITHUB_SHA:-HEAD}}
fi


export PYTHON_FILE_REGEX='^(\.\/)?(?!\.|build|external).*\.(py|pyx|pxd)$'

# Use these options to skip any of the checks
export SKIP_COPYRIGHT=${SKIP_COPYRIGHT:-""}


# Determine the merge base as the root to compare against. Optionally pass in a
# result variable otherwise the output is printed to stdout
function get_merge_base() {
   local __resultvar=$1
   local result=$(git merge-base ${BASE_SHA} ${COMMIT_SHA:-HEAD})

   if [[ "$__resultvar" ]]; then
      eval $__resultvar="'${result}'"
   else
      echo "${result}"
   fi
}

# Determine the changed files. First argument is the (optional) regex filter on
# the results. Second argument is the (optional) variable with the returned
# results. Otherwise the output is printed to stdout. Result is an array
function get_modified_files() {
   local  __resultvar=$2

   local GIT_DIFF_ARGS=${GIT_DIFF_ARGS:-"--name-only"}
   local GIT_DIFF_BASE=${GIT_DIFF_BASE:-$(get_merge_base)}

   # If invoked by a git-commit-hook, this will be populated
   local result=( $(git diff ${GIT_DIFF_ARGS} ${GIT_DIFF_BASE} | grep -P ${1:-'.*'}) )

   local files=()

   for i in "${result[@]}"; do
      if [[ -e "${i}" ]]; then
         files+=(${i})
      fi
   done

   if [[ "$__resultvar" ]]; then
      eval $__resultvar="( ${files[@]} )"
   else
      echo "${files[@]}"
   fi
}

# Determine a unified diff useful for clang-XXX-diff commands. First arg is
# optional file regex. Second argument is the (optional) variable with the
# returned results. Otherwise the output is printed to stdout
function get_unified_diff() {
   local  __resultvar=$2

   local result=$(git diff --no-color --relative -U0 $(get_merge_base) -- $(get_modified_files $1))

   if [[ "$__resultvar" ]]; then
      eval $__resultvar="'${result}'"
   else
      echo "${result}"
   fi
}

function get_num_proc() {
   NPROC_TOOL=`which nproc`
   NUM_PROC=${NUM_PROC:-`${NPROC_TOOL}`}
   echo "${NUM_PROC}"
}

function build_wheel() {
    rapids-logger "Building Wheel for $1"
    uv build --wheel --no-progress --out-dir "${WHEELS_DIR}/$2" --directory $1
}

function build_package_wheel()
{
    local pkg=$1
    pkg_dir_name="${pkg#packages/}"
    pkg_dir_name="${pkg#./packages/}"
    # Replace "nat_" with "nvidia_nat_"
    pkg_dir_name="${pkg_dir_name//nat_/nvidia_nat_}"

    # Remove compat/
    pkg_dir_name="${pkg_dir_name/compat\/}"
    build_wheel "${pkg}" "${pkg_dir_name}/${GIT_TAG}"
}

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

    rapids-logger "Creating uv env"
    VENV_DIR="${WORKSPACE_TMP}/.venv"
    uv venv --seed ${VENV_DIR}
    source ${VENV_DIR}/bin/activate

    rapids-logger "Creating Environment with extras: ${@}"

    UV_SYNC_STDERROUT=$(uv sync --active ${extras[@]} 2>&1)

    # Explicitly filter the warning about multiple packages providing a tests module, work-around for issue #611
    UV_SYNC_STDERROUT=$(echo "${UV_SYNC_STDERROUT}" | grep -v "warning: The module \`tests\` is provided by more than one package")


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

function install_rapids_gha_tools()
{
   echo "Installing Rapids GHA tools"
   wget https://github.com/rapidsai/gha-tools/releases/latest/download/tools.tar.gz -O - | tar -xz -C /usr/local/bin
}

function get_lfs_files() {
    rapids-logger "Installing git-lfs from apt"
    apt update
    apt install --no-install-recommends -y git-lfs

    if [[ "${USE_HOST_GIT}" == "1" ]]; then
        rapids-logger "Using host git, skipping git-lfs install"
    else
        rapids-logger "Fetching LFS files"
        git lfs install
        git lfs fetch
        git lfs pull
    fi

    rapids-logger "git lfs ls-files"
    git lfs ls-files
}

function cleanup {
   # Restore the original directory
   popd &> /dev/null
}


trap cleanup EXIT

# Change directory to the repo root
pushd "${PROJECT_ROOT}" &> /dev/null

NAT_EXAMPLES=($(find ./examples/ -maxdepth 4 -name "pyproject.toml" | sort | xargs dirname))
NAT_PACKAGES=($(find ./packages/ -maxdepth 2 -name "pyproject.toml" | sort | xargs dirname))
NAT_COMPAT_PACKAGES=($(find ./packages/compat -maxdepth 2 -name "pyproject.toml" | sort | xargs dirname))
