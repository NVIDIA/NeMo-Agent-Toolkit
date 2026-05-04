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
set -uo pipefail

PYTHON_BIN="${NAT_HARBOR_PYTHON_BIN:-python}"

if [ "${NAT_HARBOR_ATIF_BRIDGE_ENABLED:-0}" = "1" ]; then
  "$PYTHON_BIN" -m nat_harbor.verifier.bridge_runner \
    --artifact-path "${NAT_HARBOR_ATIF_ARTIFACT_PATH:-trajectory.json}" \
    --evaluator-kind "${NAT_HARBOR_ATIF_EVALUATOR_KIND:-custom}" \
    --evaluator-ref "${NAT_HARBOR_ATIF_EVALUATOR_REF:-}" \
    --config-file "${NAT_HARBOR_ATIF_CONFIG_FILE:-}" \
    --evaluator-name "${NAT_HARBOR_ATIF_EVALUATOR_NAME:-}" \
    --output-dir /logs/verifier \
    --fallback-mode "${NAT_HARBOR_ATIF_FALLBACK_MODE:-fail}"
  status=$?
else
  "$PYTHON_BIN" /tests/evaluate.py
  status=$?

  mkdir -p /logs/verifier
  if [ "$status" -eq 0 ]; then
    echo 1 > /logs/verifier/reward.txt
  else
    echo 0 > /logs/verifier/reward.txt
  fi
fi

exit "$status"
