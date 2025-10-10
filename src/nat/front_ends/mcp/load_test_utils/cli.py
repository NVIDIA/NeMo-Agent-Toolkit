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
"""MCP Load Test Entry Point Script.

Run load tests against MCP servers using YAML configuration files.

Usage:
    python cli.py --config_file=./configs/config.yml
    python cli.py -c configs/config.yml --verbose
"""

import argparse
import logging
import sys
from pathlib import Path

from nat.front_ends.mcp.load_test_utils import run_load_test_from_yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run MCP server load tests using YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with config file
  python cli.py --config_file=configs/config.yml

  # With verbose logging
  python cli.py --config_file=configs/config.yml --verbose

  # Short form
  python cli.py -c configs/config.yml

Configuration File Format:
  The YAML config file should contain:
    - config_file: Path to NAT workflow config
    - server: Server configuration (host, port, transport)
    - load_test: Test parameters (num_concurrent_users, duration_seconds, etc.)
    - output: Output directory configuration
    - tool_calls: List of tool calls to execute

  See configs/config.yml for a complete example.
        """,
    )

    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g., configs/config.yml)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )

    return parser.parse_args()


def print_summary(results: dict):
    """Print formatted summary of load test results.

    Args:
        results: Load test results dictionary
    """
    logger.info("=" * 70)
    logger.info("LOAD TEST RESULTS SUMMARY")
    logger.info("=" * 70)

    # Summary metrics
    summary = results.get("summary", {})
    logger.info("\nSummary Metrics:")
    logger.info("  Total Requests:      %d", summary.get("total_requests", 0))
    logger.info("  Successful:          %d", summary.get("successful_requests", 0))
    logger.info("  Failed:              %d", summary.get("failed_requests", 0))
    logger.info("  Success Rate:        %.2f%%", summary.get("success_rate", 0))
    logger.info("  Test Duration:       %.2f seconds", summary.get("test_duration_seconds", 0))
    logger.info("  Requests/Second:     %.2f", summary.get("requests_per_second", 0))

    # Latency statistics
    latency = results.get("latency_statistics", {})
    logger.info("\nLatency Statistics:")
    logger.info("  Mean:                %.2f ms", latency.get("mean_ms", 0))
    logger.info("  Median:              %.2f ms", latency.get("median_ms", 0))
    logger.info("  P95:                 %.2f ms", latency.get("p95_ms", 0))
    logger.info("  P99:                 %.2f ms", latency.get("p99_ms", 0))
    logger.info("  Min:                 %.2f ms", latency.get("min_ms", 0))
    logger.info("  Max:                 %.2f ms", latency.get("max_ms", 0))

    # Memory statistics
    memory = results.get("memory_statistics", {})
    if memory:
        logger.info("\nMemory Statistics:")
        logger.info("  Mean RSS:            %.2f MB", memory.get("rss_mean_mb", 0))
        logger.info("  Max RSS:             %.2f MB", memory.get("rss_max_mb", 0))
        logger.info("  Mean VMS:            %.2f MB", memory.get("vms_mean_mb", 0))
        logger.info("  Max VMS:             %.2f MB", memory.get("vms_max_mb", 0))
        logger.info("  Mean Memory Usage:   %.2f%%", memory.get("percent_mean", 0))
        logger.info("  Max Memory Usage:    %.2f%%", memory.get("percent_max", 0))

    # Per-tool statistics
    tool_stats = results.get("per_tool_statistics", {})
    if tool_stats:
        logger.info("\nPer-Tool Statistics:")
        for tool_name, stats in sorted(tool_stats.items()):
            logger.info("  %s:", tool_name)
            logger.info("    Calls:           %d", stats.get("total_calls", 0))
            logger.info("    Success Rate:    %.2f%%", stats.get("success_rate", 0))
            logger.info("    Mean Latency:    %.2f ms", stats.get("mean_latency_ms", 0))

    # Errors
    errors = results.get("errors", {})
    if errors:
        logger.info("\nErrors:")
        for error_msg, count in sorted(errors.items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info("  [%d] %s", count, error_msg[:80])

    # Output location
    test_config = results.get("test_configuration", {})
    output_dir = test_config.get("output_dir", "load_test_results")
    logger.info("\n" + "=" * 70)
    logger.info("Reports saved to: %s/", output_dir)
    logger.info("=" * 70)


def main():
    """Main entry point."""
    args = parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Validate config file path
    config_path = Path(args.config_file)

    if not config_path.exists():
        logger.error("Configuration file not found: %s", config_path)

        # Try to find it relative to script directory
        script_dir = Path(__file__).parent
        alternative_path = script_dir / args.config_file

        if alternative_path.exists():
            logger.info("Found config at: %s", alternative_path)
            config_path = alternative_path
        else:
            return 1

    logger.info("Starting MCP load test")
    logger.info("Configuration file: %s", config_path.absolute())

    try:
        # Run the load test
        results = run_load_test_from_yaml(str(config_path))

        # Print summary
        print_summary(results)

        # Return success
        return 0

    except KeyboardInterrupt:
        logger.warning("\nLoad test interrupted by user")
        return 130  # Standard exit code for SIGINT

    except Exception as e:
        logger.error("Load test failed: %s", e, exc_info=args.verbose)
        if not args.verbose:
            logger.info("Use --verbose flag for detailed error information")
        return 1


if __name__ == "__main__":
    sys.exit(main())
