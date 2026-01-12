#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Test script for react_benchmark_agent with Dynamo integration
# This script will run all tests and report errors without exiting

# Show help if requested
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Test the react_benchmark_agent with Dynamo integration.

Environment Variables:
  DYNAMO_BACKEND    Backend to use (default: sglang)
  DYNAMO_MODEL      Model name (default: llama-3.3-70b)
  DYNAMO_PORT       Frontend port (default: 8099)

Options:
  -h, --help        Show this help message and exit

Example:
  DYNAMO_BACKEND=vllm DYNAMO_MODEL=llama-3.1-8b $0
EOF
}

if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    usage
    exit 0
fi

# Configuration via environment variables
BACKEND="${DYNAMO_BACKEND:-sglang}"
MODEL_NAME="${DYNAMO_MODEL:-llama-3.3-70b}"
DYNAMO_PORT="${DYNAMO_PORT:-8099}"

# Get script location and derive paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_DIR="$REPO_ROOT/examples/dynamo_integration/react_benchmark_agent/configs"

# Track failures
declare -a FAILURES=()
TOTAL_TESTS=0
PASSED_TESTS=0

echo "=========================================="
echo "Testing react_benchmark_agent with Dynamo"
echo "=========================================="
echo "Backend: $BACKEND"
echo "Model: $MODEL_NAME"
echo "Port: $DYNAMO_PORT"
echo "Example Dir: $EXAMPLE_DIR"
echo "=========================================="
echo ""

# Check if NAT is available
echo "0. Checking if NAT environment is active..."
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if ! command -v nat &> /dev/null; then
    echo "✗ NAT command not found"
    echo "  Please activate your NAT environment first:"
    echo "  cd $REPO_ROOT"
    echo "  source .venv/bin/activate"
    echo ""
    echo "  Then install this example:"
    echo "  cd $EXAMPLE_DIR"
    echo "  uv pip install -e ."
    FAILURES+=("NAT command not found - environment not activated")
else
    echo "✓ NAT command found"
    PASSED_TESTS=$((PASSED_TESTS + 1))
fi
echo ""

# Check if configuration files exist
echo "1. Checking if configuration files exist..."
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if [ -f "$CONFIG_DIR/config_dynamo_e2e_test.yml" ] && [ -f "$CONFIG_DIR/config_dynamo_prefix_e2e_test.yml" ]; then
    echo "✓ Configuration files found"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo "✗ Configuration files not found"
    echo "  Expected:"
    echo "    $CONFIG_DIR/config_dynamo_e2e_test.yml"
    echo "    $CONFIG_DIR/config_dynamo_prefix_e2e_test.yml"
    FAILURES+=("Configuration files not found")
fi
echo ""

# Check if Dynamo is running
echo "2. Checking if Dynamo frontend is running on port $DYNAMO_PORT..."
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if curl -s "http://localhost:$DYNAMO_PORT/health" > /dev/null 2>&1; then
    echo "✓ Dynamo frontend is running"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo "✗ Dynamo frontend is not responding on port $DYNAMO_PORT"
    echo "  Please start Dynamo according to setup instructions in:"
    echo "  $SCRIPT_DIR/README.md"
    echo ""
    echo "  For quick reference:"
    echo "    cd $SCRIPT_DIR"
    echo "    bash start_dynamo_unified.sh"
    echo "  Or with Thompson Sampling router:"
    echo "    bash start_dynamo_unified_thompson_hints.sh"
    FAILURES+=("Dynamo frontend not responding on port $DYNAMO_PORT")
fi
echo ""

# Test basic connectivity
echo "3. Testing basic Dynamo endpoint..."
TOTAL_TESTS=$((TOTAL_TESTS + 1))
response=$(curl -s "http://localhost:$DYNAMO_PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL_NAME\",
    \"messages\": [{\"role\": \"user\", \"content\": \"What is 1+1?\"}],
    \"stream\": false,
    \"max_tokens\": 20
  }" 2>&1)

if echo "$response" | grep -q "content"; then
    echo "✓ Dynamo endpoint is working"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo "✗ Dynamo endpoint returned an error:"
    echo "$response"
    FAILURES+=("Dynamo endpoint test failed - see error output above")
fi
echo ""

# Test NAT workflow with basic config
echo "4. Testing NAT workflow with Dynamo (basic config)..."
echo "   Config: $CONFIG_DIR/config_dynamo_e2e_test.yml"
echo ""
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if nat run --config_file "$CONFIG_DIR/config_dynamo_e2e_test.yml" --input "What is 1+1?" 2>&1; then
    echo ""
    echo "✓ Basic config test completed successfully"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo ""
    echo "✗ Basic config test failed"
    FAILURES+=("NAT workflow with basic config failed")
fi
echo ""

# Test NAT workflow with prefix hints
echo "5. Testing NAT workflow with Dynamo (with prefix hints)..."
echo "   Config: $CONFIG_DIR/config_dynamo_prefix_e2e_test.yml"
echo ""
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if nat run --config_file "$CONFIG_DIR/config_dynamo_prefix_e2e_test.yml" --input "What is 1+1?" 2>&1; then
    echo ""
    echo "✓ Prefix hints test completed successfully"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo ""
    echo "✗ Prefix hints test failed"
    FAILURES+=("NAT workflow with prefix hints failed")
fi
echo ""

# Print summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Total tests: $TOTAL_TESTS"
echo "Passed: $PASSED_TESTS"
echo "Failed: $((TOTAL_TESTS - PASSED_TESTS))"
echo ""

if [ ${#FAILURES[@]} -eq 0 ]; then
    echo "✓ All tests passed!"
    exit 0
else
    echo "✗ Some tests failed:"
    for i in "${!FAILURES[@]}"; do
        echo "  $((i + 1)). ${FAILURES[$i]}"
    done
    echo ""
    echo "Please fix the issues above and try again."
    echo "For detailed setup instructions, see:"
    echo "  $EXAMPLE_DIR/README.md"
    exit 1
fi
echo "=========================================="
echo ""
