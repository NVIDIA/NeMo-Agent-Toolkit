#!/usr/bin/env python3
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
"""Simple MCP Load Test Example - Minimal Configuration."""

from nat.front_ends.mcp.load_test_utils import run_load_test

# Run a quick 10-second load test with 5 concurrent users
results = run_load_test(
    config_file="examples/getting_started/simple_calculator/configs/config.yml",
    tool_calls=[
        {
            "tool_name": "calculator_multiply", "args": {
                "text": "2 * 3"
            }
        },
        {
            "tool_name": "calculator_divide", "args": {
                "text": "10 / 2"
            }
        },
    ],
    num_concurrent_users=5,
    duration_seconds=10,
)

# Print results
print(f"\nTotal Requests: {results['summary']['total_requests']}")
print(f"Success Rate: {results['summary']['success_rate']}%")
print(f"Requests/Second: {results['summary']['requests_per_second']}")
print(f"Mean Latency: {results['latency_statistics']['mean_ms']:.2f} ms")
print(f"\nReports saved to: load_test_results/")
