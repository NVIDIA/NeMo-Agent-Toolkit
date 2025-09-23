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

GITLAB_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

source ${GITLAB_SCRIPT_DIR}/common.sh

create_env group:dev extra:all

rapids-logger "Git Version: $(git describe)"

rapids-logger "Running tests"
set +e

PYTEST_ARGS=""
REPORT_NAME="${CI_PROJECT_DIR}/pytest_junit_report.xml"
COV_REPORT_NAME="${CI_PROJECT_DIR}/pytest_coverage_report.xml"
if [ "${CI_CRON_NIGHTLY}" == "1" ]; then
       PYTEST_ARGS="--run_slow --run_integration"

       DATE_TAG=$(date +"%Y%m%d")
       REPORT_NAME="${CI_PROJECT_DIR}/pytest_junit_report_${DATE_TAG}.xml"
       COV_REPORT_NAME="${CI_PROJECT_DIR}/pytest_coverage_report_${DATE_TAG}.xml"
fi

pytest ${PYTEST_ARGS}  \
       --junit-xml=${REPORT_NAME} \
       --cov=nat --cov-report term-missing \
       --cov-report=xml:${COV_REPORT_NAME}
PYTEST_RESULTS=$?

if [ "${CI_CRON_NIGHTLY}" == "1" ]; then
       # Since this dependency is specific to only this script, we will just install it here
       rapids-logger "Installing slack-sdk"
       uv pip install "slack-sdk~=3.36"

       rapids-logger "Reporting test results"
       ${GITLAB_SCRIPT_DIR}/report_test_results.py ${REPORT_NAME} ${COV_REPORT_NAME}
       REPORT_RESULT=$?
       if [ ${REPORT_RESULT} -ne 0 ]; then
              rapids-logger "Failed to report test results to Slack"
              exit ${REPORT_RESULT}
       fi
fi

exit ${PYTEST_RESULTS}
