#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

create_env

rapids-logger "Running NIM model health check"
HEALTH_JSON=${CI_PROJECT_DIR:-${PROJECT_ROOT}}/model_health_results.json

set +e
python ${SCRIPT_DIR}/model_health_check.py --output-json "${HEALTH_JSON}"
HEALTH_RESULT=$?
set -e

set +e
install_slack_sdk
rapids-logger "Reporting model health results to Slack"
${GITLAB_SCRIPT_DIR}/report_test_results.py --model-health-json "${HEALTH_JSON}"
REPORT_RESULT=$?
set -e

if [ ${REPORT_RESULT} -ne 0 ]; then
    rapids-logger "Failed to report model health results to Slack"
fi

exit ${HEALTH_RESULT}
