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
"""Example MCP Load Test Script.

This script demonstrates how to run a load test against an MCP server.
Customize the configuration below for your specific use case.
"""

import logging
import sys
from pathlib import Path

from nat.front_ends.mcp.load_test_utils import run_load_test

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)


def main():
    """Run the load test with example configuration."""
    # Configuration
    config_file = "examples/getting_started/simple_calculator/configs/config.yml"

    # Verify config file exists
    if not Path(config_file).exists():
        logger.error("Config file not found: %s", config_file)
        logger.info("Please update the config_file path in this script")
        sys.exit(1)

    # Define tool calls to test
    # Each tool call should specify:
    # - tool_name: Name of the MCP tool
    # - args: Dictionary of arguments to pass
    # - weight: Relative frequency (optional, default 1.0)
    tool_calls = [
        {
            "tool_name": "calculator_multiply",
            "args": {
                "text": "2 * 3"
            },
            "weight": 2.0,  # Called twice as often as others
        },
        {
            "tool_name": "calculator_divide",
            "args": {
                "text": "10 / 2"
            },
            "weight": 1.0,
        },
        {
            "tool_name": "calculator_subtract",
            "args": {
                "text": "10 - 3"
            },
            "weight": 1.0,
        },
        {
            "tool_name": "calculator_inequality",
            "args": {
                "text": "5 > 3"
            },
            "weight": 1.0,
        },
    ]

    # Load test parameters
    num_concurrent_users = 10
    duration_seconds = 30
    server_port = 9901
    warmup_seconds = 5

    logger.info("Starting MCP load test")
    logger.info("Config file: %s", config_file)
    logger.info("Concurrent users: %d", num_concurrent_users)
    logger.info("Duration: %d seconds", duration_seconds)
    logger.info("Tool calls: %d types", len(tool_calls))

    try:
        # Run the load test
        results = run_load_test(
            config_file=config_file,
            tool_calls=tool_calls,
            num_concurrent_users=num_concurrent_users,
            duration_seconds=duration_seconds,
            server_port=server_port,
            warmup_seconds=warmup_seconds,
            output_dir="load_test_results",
        )

        # Print summary
        logger.info("Load test completed successfully!")
        logger.info("=" * 70)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 70)

        summary = results.get("summary", {})
        logger.info("Total Requests: %d", summary.get("total_requests", 0))
        logger.info("Successful: %d", summary.get("successful_requests", 0))
        logger.info("Failed: %d", summary.get("failed_requests", 0))
        logger.info("Success Rate: %.2f%%", summary.get("success_rate", 0))
        logger.info("Requests/Second: %.2f", summary.get("requests_per_second", 0))

        latency = results.get("latency_statistics", {})
        logger.info("\nLatency Statistics:")
        logger.info("  Mean: %.2f ms", latency.get("mean_ms", 0))
        logger.info("  Median: %.2f ms", latency.get("median_ms", 0))
        logger.info("  P95: %.2f ms", latency.get("p95_ms", 0))
        logger.info("  P99: %.2f ms", latency.get("p99_ms", 0))

        logger.info("\nReports saved to: load_test_results/")
        logger.info("=" * 70)

        return 0

    except Exception as e:
        logger.error("Load test failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
