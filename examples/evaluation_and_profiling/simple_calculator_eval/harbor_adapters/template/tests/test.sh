#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -uo pipefail

if [ "${NAT_HARBOR_ATIF_BRIDGE_ENABLED:-0}" = "1" ]; then
  python -m nat_harbor.verifier.bridge_runner \
    --artifact-path "${NAT_HARBOR_ATIF_ARTIFACT_PATH:-trajectory.json}" \
    --evaluator-kind "${NAT_HARBOR_ATIF_EVALUATOR_KIND:-custom}" \
    --evaluator-ref "${NAT_HARBOR_ATIF_EVALUATOR_REF:-}" \
    --config-file "${NAT_HARBOR_ATIF_CONFIG_FILE:-}" \
    --evaluator-name "${NAT_HARBOR_ATIF_EVALUATOR_NAME:-}" \
    --output-dir /logs/verifier \
    --fallback-mode "${NAT_HARBOR_ATIF_FALLBACK_MODE:-fail}"
  status=$?
else
  python /tests/evaluate.py
  status=$?

  if [ "$status" -eq 0 ]; then
    echo 1 > /logs/verifier/reward.txt
  else
    echo 0 > /logs/verifier/reward.txt
  fi
fi

exit "$status"

