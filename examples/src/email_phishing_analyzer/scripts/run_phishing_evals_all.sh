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


#!/bin/bash
# run_model_comparison.sh

# SECURITY WARNING: Hard-coding API keys in scripts is not recommended for production
# Replace YOUR_API_KEY_HERE with your actual NVIDIA API key
HARDCODED_API_KEY="<YOUR_API_KEY_HERE>"

# Use environment variable or argument if provided, otherwise use hardcoded key
if [ ! -z "$1" ]; then
  export NVIDIA_API_KEY="$1"
  echo "Using API key from command line argument"
elif [ ! -z "$NVIDIA_API_KEY" ]; then
  echo "Using NVIDIA_API_KEY from environment"
else
  export NVIDIA_API_KEY="$HARDCODED_API_KEY"
  echo "Using hardcoded API key"
fi

# Define all config files
CONFIGS=(
  "examples/intermediate/evaluation_and_profiling/email_phishing_analyzer/configs/config-llama-3.1-8b-instruct.yml"
"examples/intermediate/evaluation_and_profiling/email_phishing_analyzer/configs/config-llama-3.3-70b-instruct.yml"
"examples/intermediate/evaluation_and_profiling/email_phishing_analyzer/configs/config-mixtral-8x22b-instruct-v0.1.yml"
"examples/intermediate/evaluation_and_profiling/email_phishing_analyzer/configs/config-phi-3-medium-4k-instruct.yml"
"examples/intermediate/evaluation_and_profiling/email_phishing_analyzer/configs/config-phi-3-mini-4k-instruct.yml"
)

# Create temp files for exit codes and store process IDs
EXIT_FILES=()
PIDS=()

echo "Starting evaluations in parallel..."

# Launch all evaluations in parallel
for config in "${CONFIGS[@]}"; do
  # Create temp file for this config
  EXIT_FILE=$(mktemp)
  EXIT_FILES+=("$EXIT_FILE")

  # Get config name for display
  CONFIG_NAME=$(basename "$config")

  # Run in background
  (
    echo "Running $CONFIG_NAME..."
    aiq eval --config_file="$config"
    echo $? > "$EXIT_FILE"
  ) &

  # Store process ID
  PIDS+=($!)
done

# Wait for all processes to complete
echo "Waiting for all evaluations to complete..."
wait "${PIDS[@]}"

# Check results
ALL_PASSED=true
FAILED_CONFIGS=()

for i in "${!CONFIGS[@]}"; do
  CONFIG_NAME=$(basename "${CONFIGS[$i]}")
  EXIT_CODE=$(cat "${EXIT_FILES[$i]}")

  if [ $EXIT_CODE -ne 0 ]; then
    ALL_PASSED=false
    FAILED_CONFIGS+=("$CONFIG_NAME (exit code $EXIT_CODE)")
  fi

  # Clean up temp files
  rm "${EXIT_FILES[$i]}"
done

# Print final results
if [ "$ALL_PASSED" = true ]; then
  echo -e "\nAll evaluations completed successfully!"
  exit 0
else
  echo -e "\nThe following evaluations failed:" >&2
  for failed in "${FAILED_CONFIGS[@]}"; do
    echo "  - $failed" >&2
  done
  exit 1
fi
