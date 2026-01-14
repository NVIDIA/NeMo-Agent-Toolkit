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

# Dynamo SGLang Shutdown Script
# Stops all components: Dynamo worker container, ETCD, and NATS
# Works for: UNIFIED, THOMPSON SAMPLING, and DISAGGREGATED modes

echo "========================================================="
echo "Stopping Dynamo SGLang FULL STACK"
echo "========================================================="
echo ""

# Stop Dynamo containers (check for both standard and thompson variants)
STOPPED_CONTAINER=false

if docker ps --format '{{.Names}}' | grep -q "^dynamo-sglang$"; then
    echo "Stopping Dynamo container (standard)..."
    docker stop dynamo-sglang
    docker rm dynamo-sglang
    echo "✓ Dynamo container stopped and removed"
    STOPPED_CONTAINER=true
fi

if docker ps --format '{{.Names}}' | grep -q "^dynamo-sglang-thompson$"; then
    echo "Stopping Dynamo container (Thompson Sampling)..."
    docker stop dynamo-sglang-thompson
    docker rm dynamo-sglang-thompson
    echo "✓ Dynamo Thompson container stopped and removed"
    STOPPED_CONTAINER=true
fi

if [ "$STOPPED_CONTAINER" = false ]; then
    echo "  (No Dynamo containers running)"
fi

# Stop ETCD
if docker ps --format '{{.Names}}' | grep -q "^etcd-dynamo$"; then
    echo ""
    echo "Stopping ETCD container..."
    docker stop etcd-dynamo
    docker rm etcd-dynamo
    echo "✓ ETCD container stopped and removed"
else
    echo "  (ETCD container not running)"
fi

# Stop NATS
if docker ps --format '{{.Names}}' | grep -q "^nats-dynamo$"; then
    echo ""
    echo "Stopping NATS container..."
    docker stop nats-dynamo
    docker rm nats-dynamo
    echo "✓ NATS container stopped and removed"
else
    echo "  (NATS container not running)"
fi

echo ""
echo "========================================================="
echo "✓ All components stopped!"
echo "========================================================="
echo ""
echo "To restart:"
echo "  Standard Unified:     bash start_dynamo_unified.sh"
echo "  Thompson Sampling:    bash start_dynamo_unified_thompson_hints.sh"
echo ""

