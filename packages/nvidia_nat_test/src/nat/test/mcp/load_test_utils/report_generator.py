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
"""Report generation for MCP load tests."""

import statistics
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from nat.test.mcp.load_test_utils.load_tester import LoadTestConfig
    from nat.test.mcp.load_test_utils.load_tester import MemorySample
    from nat.test.mcp.load_test_utils.load_tester import ToolCallResult


def generate_summary_report(
    results: list['ToolCallResult'],
    test_duration: float,
    config: 'LoadTestConfig',
    memory_samples: list['MemorySample'] | None = None,
) -> dict[str, Any]:
    """Generate summary statistics from load test results.

    Args:
        results: List of tool call results
        test_duration: Actual duration of the test in seconds
        config: Load test configuration
        memory_samples: Optional list of memory usage samples

    Returns:
        Dictionary containing summary statistics
    """
    if not results:
        return {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "success_rate": 0.0,
            "test_duration_seconds": test_duration,
            "requests_per_second": 0.0,
        }

    total_requests = len(results)
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    successful_count = len(successful)
    failed_count = len(failed)
    success_rate = (successful_count / total_requests) * 100 if total_requests > 0 else 0.0

    latencies = [r.latency_ms for r in successful]

    if latencies:
        latency_stats = {
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "p95_ms": _percentile(latencies, 0.95),
            "p99_ms": _percentile(latencies, 0.99),
            "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        }
    else:
        latency_stats = {
            "min_ms": 0.0,
            "max_ms": 0.0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "stdev_ms": 0.0,
        }

    tool_stats: dict[str, dict[str, Any]] = {}
    tool_names = set(r.tool_name for r in results)

    for tool_name in tool_names:
        tool_results = [r for r in results if r.tool_name == tool_name]
        tool_successful = [r for r in tool_results if r.success]
        tool_latencies = [r.latency_ms for r in tool_successful]

        tool_stats[tool_name] = {
            "total_calls": len(tool_results),
            "successful_calls": len(tool_successful),
            "failed_calls": len(tool_results) - len(tool_successful),
            "success_rate": (len(tool_successful) / len(tool_results)) * 100 if tool_results else 0.0,
            "mean_latency_ms": statistics.mean(tool_latencies) if tool_latencies else 0.0,
            "median_latency_ms": statistics.median(tool_latencies) if tool_latencies else 0.0,
            "p95_latency_ms": _percentile(tool_latencies, 0.95) if tool_latencies else 0.0,
        }

    error_counts: dict[str, int] = {}
    for result in failed:
        error_msg = result.error or "Unknown error"
        error_key = error_msg[:100]
        error_counts[error_key] = error_counts.get(error_key, 0) + 1

    memory_stats: dict[str, Any] = {}
    if memory_samples:
        rss_values = [sample.rss_mb for sample in memory_samples]
        vms_values = [sample.vms_mb for sample in memory_samples]
        percent_values = [sample.percent for sample in memory_samples]

        if rss_values:
            memory_stats = {
                "samples_count": len(memory_samples),
                "rss_mean_mb": statistics.mean(rss_values),
                "rss_max_mb": max(rss_values),
                "rss_min_mb": min(rss_values),
                "vms_mean_mb": statistics.mean(vms_values),
                "vms_max_mb": max(vms_values),
                "percent_mean": statistics.mean(percent_values),
                "percent_max": max(percent_values),
            }

    report = {
        "test_configuration": {
            "config_file": config.config_file,
            "num_concurrent_users": config.num_concurrent_users,
            "duration_seconds": config.duration_seconds,
            "server_url": f"http://{config.server_host}:{config.server_port}",
            "transport": config.transport,
            "warmup_seconds": config.warmup_seconds,
            "tool_calls_configured": len(config.tool_calls),
            "output_dir": config.output_dir or "load_test_results",
        },
        "summary": {
            "total_requests":
                total_requests,
            "successful_requests":
                successful_count,
            "failed_requests":
                failed_count,
            "success_rate":
                round(success_rate, 2),
            "test_duration_seconds":
                round(test_duration, 2),
            "requests_per_second":
                round(total_requests / test_duration, 2) if test_duration > 0 else 0.0,
            "avg_concurrent_rps":
                round(total_requests / test_duration /
                      config.num_concurrent_users, 2) if test_duration > 0 and config.num_concurrent_users > 0 else 0.0,
        },
        "latency_statistics": {
            k: round(v, 2)
            for k, v in latency_stats.items()
        },
        "per_tool_statistics": tool_stats,
        "errors": error_counts,
    }

    if memory_stats:
        report["memory_statistics"] = {k: round(v, 2) if isinstance(v, float) else v for k, v in memory_stats.items()}

    return report


def _percentile(data: list[float], percentile: float) -> float:
    """Calculate percentile value.

    Args:
        data: List of values
        percentile: Percentile to calculate

    Returns:
        Percentile value
    """
    if not data:
        return 0.0

    sorted_data = sorted(data)
    index = int(len(sorted_data) * percentile)
    index = min(index, len(sorted_data) - 1)
    return sorted_data[index]
