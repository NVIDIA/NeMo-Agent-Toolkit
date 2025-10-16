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

set +e

# Intentionally excluding CHANGELOG.md as it immutable
DOC_FILES=$(git ls-files "*.md" "*.rst" | grep -v -E '(^|/)(CHANGELOG|LICENSE)\.md$')
NOTEBOOK_FILES=$(git ls-files "*.ipynb")

if [[ -v ${WORKSPACE_TMP} ]]; then
    MKTEMP_ARGS=""
else
    MKTEMP_ARGS="--tmpdir=${WORKSPACE_TMP}"
fi

EXPORT_DIR=$(mktemp -d ${MKTEMP_ARGS} nat_converted_notebooks.XXXXXX)
if [[ ! -d "${EXPORT_DIR}" ]]; then
    echo "ERROR: Failed to create temporary directory" >&2
    exit 1
fi

jupyter nbconvert -y --log-level=WARN --to markdown --output-dir ${EXPORT_DIR} ${NOTEBOOK_FILES}
if [[ $? -ne 0 ]]; then
    echo "ERROR: Failed to convert notebooks" >&2
    rm -rf "${EXPORT_DIR}"
    exit 1
fi

CONVERTED_NOTEBOOK_FILES=$(find ${EXPORT_DIR} -type f  -name "*.md")

vale ${DOC_FILES} ${CONVERTED_NOTEBOOK_FILES}
RETVAL=$?

if [[ "${PRESERVE_TMP}" == "1" ]]; then
    echo "Preserving temporary directory: ${EXPORT_DIR}"
else
    rm -rf "${EXPORT_DIR}"
fi

exit $RETVAL
