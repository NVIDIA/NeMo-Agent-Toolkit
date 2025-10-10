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
    from nat.front_ends.mcp.load_test_utils.load_test import LoadTestConfig
    from nat.front_ends.mcp.load_test_utils.load_test import ToolCallResult


def generate_summary_report(
    results: list['ToolCallResult'],
    test_duration: float,
    config: 'LoadTestConfig',
) -> dict[str, Any]:
    """Generate summary statistics from load test results.

    Args:
        results: List of tool call results
        test_duration: Actual duration of the test in seconds
        config: Load test configuration

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

    # Calculate latency statistics (only for successful requests)
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

    # Per-tool statistics
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

    # Error analysis
    error_counts: dict[str, int] = {}
    for result in failed:
        error_msg = result.error or "Unknown error"
        # Truncate long error messages
        error_key = error_msg[:100]
        error_counts[error_key] = error_counts.get(error_key, 0) + 1

    return {
        "test_configuration": {
            "config_file": config.config_file,
            "num_concurrent_users": config.num_concurrent_users,
            "duration_seconds": config.duration_seconds,
            "server_url": f"http://{config.server_host}:{config.server_port}",
            "transport": config.transport,
            "warmup_seconds": config.warmup_seconds,
            "tool_calls_configured": len(config.tool_calls),
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


def _percentile(data: list[float], percentile: float) -> float:
    """Calculate percentile value.

    Args:
        data: Sorted or unsorted list of values
        percentile: Percentile to calculate (0.0 to 1.0)

    Returns:
        Percentile value
    """
    if not data:
        return 0.0

    sorted_data = sorted(data)
    index = int(len(sorted_data) * percentile)
    index = min(index, len(sorted_data) - 1)
    return sorted_data[index]


def generate_html_report(summary: dict[str, Any], config: 'LoadTestConfig') -> str:
    """Generate an HTML report from summary statistics.

    Args:
        summary: Summary statistics dictionary
        config: Load test configuration

    Returns:
        HTML report as a string
    """
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCP Load Test Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #76b900;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 30px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 8px;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background: #f9f9f9;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #76b900;
        }
        .metric-label {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }
        .metric-unit {
            font-size: 0.5em;
            color: #999;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #76b900;
            color: white;
            font-weight: 600;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .success {
            color: #4caf50;
        }
        .error {
            color: #f44336;
        }
        .config-item {
            margin: 10px 0;
            padding: 10px;
            background: #f9f9f9;
            border-radius: 4px;
        }
        .config-label {
            font-weight: 600;
            color: #555;
        }
        .config-value {
            color: #333;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>MCP Load Test Report</h1>
"""

    # Test Configuration
    html += "<h2>Test Configuration</h2>"
    config_data = summary.get("test_configuration", {})
    for key, value in config_data.items():
        label = key.replace("_", " ").title()
        html += f'<div class="config-item"><span class="config-label">{label}:</span> <span class="config-value">{value}</span></div>\n'

    # Summary Metrics
    html += "<h2>Summary Metrics</h2>"
    html += '<div class="metric-grid">'

    summary_data = summary.get("summary", {})
    metrics = [
        ("Total Requests", summary_data.get("total_requests", 0), ""),
        ("Success Rate", summary_data.get("success_rate", 0), "%"),
        ("Requests/Second", summary_data.get("requests_per_second", 0), "req/s"),
        ("Test Duration", summary_data.get("test_duration_seconds", 0), "s"),
    ]

    for label, value, unit in metrics:
        html += f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}<span class="metric-unit">{unit}</span></div>
        </div>
        """

    html += "</div>"

    # Latency Statistics
    html += "<h2>Latency Statistics</h2>"
    html += '<div class="metric-grid">'

    latency_data = summary.get("latency_statistics", {})
    latency_metrics = [
        ("Mean", latency_data.get("mean_ms", 0)),
        ("Median", latency_data.get("median_ms", 0)),
        ("P95", latency_data.get("p95_ms", 0)),
        ("P99", latency_data.get("p99_ms", 0)),
        ("Min", latency_data.get("min_ms", 0)),
        ("Max", latency_data.get("max_ms", 0)),
    ]

    for label, value in latency_metrics:
        html += f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value:.2f}<span class="metric-unit">ms</span></div>
        </div>
        """

    html += "</div>"

    # Per-Tool Statistics
    html += "<h2>Per-Tool Statistics</h2>"
    html += "<table>"
    html += """
        <tr>
            <th>Tool Name</th>
            <th>Total Calls</th>
            <th>Success Rate</th>
            <th>Mean Latency</th>
            <th>Median Latency</th>
            <th>P95 Latency</th>
        </tr>
    """

    tool_stats = summary.get("per_tool_statistics", {})
    for tool_name, stats in tool_stats.items():
        success_class = "success" if stats["success_rate"] >= 95 else "error"
        html += f"""
        <tr>
            <td>{tool_name}</td>
            <td>{stats['total_calls']}</td>
            <td class="{success_class}">{stats['success_rate']:.2f}%</td>
            <td>{stats['mean_latency_ms']:.2f} ms</td>
            <td>{stats['median_latency_ms']:.2f} ms</td>
            <td>{stats['p95_latency_ms']:.2f} ms</td>
        </tr>
        """

    html += "</table>"

    # Errors (if any)
    errors = summary.get("errors", {})
    if errors:
        html += "<h2>Errors</h2>"
        html += "<table>"
        html += "<tr><th>Error Message</th><th>Count</th></tr>"
        for error_msg, count in sorted(errors.items(), key=lambda x: x[1], reverse=True):
            html += f"<tr><td>{error_msg}</td><td class='error'>{count}</td></tr>"
        html += "</table>"

    html += """
    </div>
</body>
</html>
"""

    return html
