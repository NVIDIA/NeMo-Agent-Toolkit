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

# Monitor Dynamo Custom Components
# Helper script to view logs and status of running Dynamo system

CONTAINER_NAME="dynamo-sglang"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo "========================================================="
    echo "$1"
    echo "========================================================="
}

print_status() {
    local status=$1
    local message=$2
    if [ "$status" == "ok" ]; then
        echo -e "${GREEN}✓${NC} $message"
    elif [ "$status" == "warn" ]; then
        echo -e "${YELLOW}⚠${NC} $message"
    else
        echo -e "${RED}✗${NC} $message"
    fi
}

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    print_status "error" "Container '$CONTAINER_NAME' is not running"
    echo ""
    echo "Start it with: ./start_dynamo_custom.sh"
    exit 1
fi

print_status "ok" "Container '$CONTAINER_NAME' is running"

# Menu
print_header "Dynamo Monitoring Menu"
echo "1. View Frontend logs (OpenAI API)"
echo "2. View Processor logs (tokenization, prefix tracking)"
echo "3. View Router logs (KV-aware routing)"
echo "4. View all component logs"
echo "5. View container logs (all processes)"
echo "6. Test health endpoint"
echo "7. Test basic inference"
echo "8. Check GPU usage"
echo "9. Check process status inside container"
echo "0. Exit"
echo ""
read -p "Select option [0-9]: " option

case $option in
    1)
        print_header "Frontend Logs (last 50 lines)"
        docker exec $CONTAINER_NAME cat /tmp/frontend.log 2>/dev/null | tail -n 50 || echo "Frontend log not available"
        echo ""
        read -p "Follow logs in real-time? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker exec $CONTAINER_NAME tail -f /tmp/frontend.log 2>/dev/null || echo "Cannot tail frontend log"
        fi
        ;;
    2)
        print_header "Processor Logs (last 50 lines)"
        docker exec $CONTAINER_NAME cat /tmp/processor.log 2>/dev/null | tail -n 50 || echo "Processor log not available"
        echo ""
        read -p "Follow logs in real-time? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker exec $CONTAINER_NAME tail -f /tmp/processor.log 2>/dev/null || echo "Cannot tail processor log"
        fi
        ;;
    3)
        print_header "Router Logs (last 50 lines)"
        docker exec $CONTAINER_NAME cat /tmp/router.log 2>/dev/null | tail -n 50 || echo "Router log not available"
        echo ""
        read -p "Follow logs in real-time? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker exec $CONTAINER_NAME tail -f /tmp/router.log 2>/dev/null || echo "Cannot tail router log"
        fi
        ;;
    4)
        print_header "All Component Logs"
        echo ""
        echo "=== Frontend Log (last 20 lines) ==="
        docker exec $CONTAINER_NAME cat /tmp/frontend.log 2>/dev/null | tail -n 20 || echo "Frontend log not available"
        echo ""
        echo "=== Processor Log (last 20 lines) ==="
        docker exec $CONTAINER_NAME cat /tmp/processor.log 2>/dev/null | tail -n 20 || echo "Processor log not available"
        echo ""
        echo "=== Router Log (last 20 lines) ==="
        docker exec $CONTAINER_NAME cat /tmp/router.log 2>/dev/null | tail -n 20 || echo "Router log not available"
        ;;
    5)
        print_header "Container Logs"
        docker logs --tail 100 $CONTAINER_NAME
        echo ""
        read -p "Follow logs in real-time? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker logs -f $CONTAINER_NAME
        fi
        ;;
    6)
        print_header "Health Check"
        echo "Testing: http://localhost:8099/health"
        echo ""
        http_code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8099/health 2>&1)
        if [ "$http_code" == "200" ]; then
            print_status "ok" "Health check passed (HTTP $http_code)"
        else
            print_status "error" "Health check failed (HTTP $http_code)"
        fi
        ;;
    7)
        print_header "Test Basic Inference"
        echo "Sending test request to http://localhost:8099/v1/chat/completions"
        echo ""
        response=$(curl -s http://localhost:8099/v1/chat/completions \
          -H "Content-Type: application/json" \
          -d '{
            "model": "llama-3.1-8b",
            "messages": [{"role": "user", "content": "Say hello in one word"}],
            "max_tokens": 10,
            "stream": false
          }' 2>&1)
        
        if echo "$response" | grep -q "content"; then
            print_status "ok" "Inference successful"
            echo ""
            echo "Full response:"
            echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
        else
            print_status "error" "Inference failed"
            echo ""
            echo "Response: $response"
        fi
        ;;
    8)
        print_header "GPU Usage"
        nvidia-smi
        ;;
    9)
        print_header "Process Status Inside Container"
        docker exec $CONTAINER_NAME ps aux | grep -E "(python|PID)" | grep -v grep
        ;;
    0)
        echo "Exiting."
        exit 0
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo ""
print_header "Monitoring Complete"
echo "Run this script again to see more options."
echo ""

