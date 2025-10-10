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
"""MCP Server Load Testing."""

import asyncio
import logging
import random
import subprocess
import time
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import TextContent

from nat.front_ends.mcp.load_test_utils.report_generator import generate_summary_report

logger = logging.getLogger(__name__)


@dataclass
class ToolCallConfig:
    """Configuration for a tool call in load testing."""

    tool_name: str
    """Name of the tool to call"""

    args: dict[str, Any] = field(default_factory=dict)
    """Arguments to pass to the tool"""

    weight: float = 1.0
    """Relative weight for this tool call (higher = more frequent)"""


@dataclass
class LoadTestConfig:
    """Configuration for MCP load testing."""

    config_file: str
    """Path to the NAT workflow config file"""

    tool_calls: list[ToolCallConfig]
    """List of tool calls to execute during load testing"""

    num_concurrent_users: int = 10
    """Number of concurrent users to simulate"""

    duration_seconds: int = 60
    """Duration of the load test in seconds"""

    server_host: str = "localhost"
    """MCP server host"""

    server_port: int = 9901
    """MCP server port"""

    transport: str = "streamable-http"
    """Transport type (streamable-http or sse)"""

    warmup_seconds: int = 5
    """Warmup period before starting measurements"""

    output_dir: str | None = None
    """Output directory for reports (default: load_test_results)"""


@dataclass
class ToolCallResult:
    """Result of a single tool call."""

    tool_name: str
    success: bool
    latency_ms: float
    timestamp: float
    error: str | None = None
    response: str | None = None


@dataclass
class MemorySample:
    """Memory usage sample at a point in time."""

    timestamp: float
    rss_mb: float  # Resident Set Size in MB
    vms_mb: float  # Virtual Memory Size in MB
    percent: float  # Memory usage percentage


class MCPLoadTest:
    """MCP Server Load Test Runner."""

    def __init__(self, config: LoadTestConfig):
        """Initialize load test runner.

        Args:
            config: Load test configuration
        """
        self.config = config
        self.results: list[ToolCallResult] = []
        self.memory_samples: list[MemorySample] = []
        self.server_process: subprocess.Popen | None = None
        self.server_url = self._get_server_url()
        self._memory_monitor_task: asyncio.Task | None = None

        # Set up output directory
        if config.output_dir:
            self.output_dir = Path(config.output_dir)
        else:
            self.output_dir = Path("load_test_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_server_url(self) -> str:
        """Get the MCP server URL based on transport type."""
        endpoint = "mcp" if self.config.transport == "streamable-http" else "sse"
        return f"http://{self.config.server_host}:{self.config.server_port}/{endpoint}"

    def _client_ctx(self):
        """Get the appropriate MCP client context manager based on transport type.

        Returns:
            Client context manager (streamablehttp_client or sse_client)
        """
        if self.config.transport == "streamable-http":
            return streamablehttp_client(url=self.server_url)
        else:
            from mcp.client.sse import sse_client
            return sse_client(url=self.server_url)

    async def _start_server(self) -> None:
        """Start the MCP server."""
        logger.info("Starting MCP server with config: %s", self.config.config_file)

        cmd = [
            "nat",
            "mcp",
            "serve",
            "--config_file",
            self.config.config_file,
            "--host",
            self.config.server_host,
            "--port",
            str(self.config.server_port),
            "--transport",
            self.config.transport,
        ]

        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for server to be ready
        max_retries = 30
        for i in range(max_retries):
            try:
                async with self._client_ctx() as ctx:
                    read, write = (ctx[0], ctx[1]) if isinstance(ctx, tuple) else ctx
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                logger.info("MCP server is ready")
                return
            except Exception as e:
                if i == max_retries - 1:
                    raise RuntimeError(f"Failed to start MCP server after {max_retries} retries") from e
                await asyncio.sleep(1)

    async def _stop_server(self) -> None:
        """Stop the MCP server."""
        if self.server_process:
            logger.info("Stopping MCP server")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Server didn't stop gracefully, killing it")
                self.server_process.kill()
                self.server_process.wait()

    def _select_tool_call(self) -> ToolCallConfig:
        """Select a tool call based on weights."""
        total_weight = sum(tc.weight for tc in self.config.tool_calls)
        r = random.uniform(0, total_weight)
        cumulative = 0.0

        for tool_call in self.config.tool_calls:
            cumulative += tool_call.weight
            if r <= cumulative:
                return tool_call

        # Fallback to first tool call
        return self.config.tool_calls[0]

    async def _call_tool(self, tool_call: ToolCallConfig) -> ToolCallResult:
        """Execute a single tool call and measure latency.

        Args:
            tool_call: Tool call configuration

        Returns:
            ToolCallResult with timing and success information
        """
        start_time = time.time()
        timestamp = start_time

        try:
            async with self._client_ctx() as ctx:
                read, write = (ctx[0], ctx[1]) if isinstance(ctx, tuple) else ctx
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_call.tool_name, tool_call.args)

            # Extract response
            outputs: list[str] = []
            for content in result.content:
                if isinstance(content, TextContent):
                    outputs.append(content.text)
                else:
                    outputs.append(str(content))

            response = "\n".join(outputs)
            is_error = getattr(result, "isError", False)

            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            return ToolCallResult(
                tool_name=tool_call.tool_name,
                success=not is_error,
                latency_ms=latency_ms,
                timestamp=timestamp,
                response=response if not is_error else None,
                error=response if is_error else None,
            )

        except Exception as e:
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            return ToolCallResult(
                tool_name=tool_call.tool_name,
                success=False,
                latency_ms=latency_ms,
                timestamp=timestamp,
                error=str(e),
            )

    async def _user_simulation(self, user_id: int, end_time: float) -> None:
        """Simulate a single user making repeated tool calls.

        Args:
            user_id: User identifier for logging
            end_time: Timestamp when to stop making calls
        """
        logger.debug("User %d starting simulation", user_id)

        while time.time() < end_time:
            tool_call = self._select_tool_call()
            result = await self._call_tool(tool_call)
            self.results.append(result)

            # Small random delay between calls to simulate realistic behavior
            await asyncio.sleep(random.uniform(0.1, 0.2))

        logger.debug("User %d finished simulation", user_id)

    async def _monitor_memory(self, end_time: float) -> None:
        """Monitor server memory usage during the test.

        Args:
            end_time: Timestamp when to stop monitoring
        """
        if not self.server_process:
            return

        try:
            process = psutil.Process(self.server_process.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            logger.warning("Cannot monitor memory for server process")
            return

        while time.time() < end_time:
            try:
                mem_info = process.memory_info()
                mem_percent = process.memory_percent()

                self.memory_samples.append(
                    MemorySample(
                        timestamp=time.time(),
                        rss_mb=mem_info.rss / (1024 * 1024),  # Convert to MB
                        vms_mb=mem_info.vms / (1024 * 1024),  # Convert to MB
                        percent=mem_percent,
                    ))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

            await asyncio.sleep(1.0)  # Sample every second

    async def run(self) -> dict[str, Any]:
        """Run the load test.

        Returns:
            Dictionary containing test results and statistics
        """
        logger.info("Starting load test with config: %s", self.config)

        # Start server
        await self._start_server()

        try:
            # Warmup period
            if self.config.warmup_seconds > 0:
                logger.info("Warming up for %d seconds", self.config.warmup_seconds)
                warmup_end = time.time() + self.config.warmup_seconds
                warmup_tasks = [
                    asyncio.create_task(self._user_simulation(i, warmup_end))
                    for i in range(min(3, self.config.num_concurrent_users))
                ]
                await asyncio.gather(*warmup_tasks)
                self.results.clear()  # Clear warmup results

            # Actual load test
            logger.info(
                "Starting load test: %d concurrent users for %d seconds",
                self.config.num_concurrent_users,
                self.config.duration_seconds,
            )

            test_start_time = time.time()
            test_end_time = test_start_time + self.config.duration_seconds

            # Start memory monitoring
            self._memory_monitor_task = asyncio.create_task(self._monitor_memory(test_end_time))

            # Create concurrent user simulations
            tasks = [
                asyncio.create_task(self._user_simulation(i, test_end_time))
                for i in range(self.config.num_concurrent_users)
            ]

            # Wait for all users to complete
            await asyncio.gather(*tasks)

            # Wait for memory monitoring to complete
            if self._memory_monitor_task:
                await self._memory_monitor_task

            test_duration = time.time() - test_start_time

            logger.info("Load test completed. Total calls: %d", len(self.results))

            # Generate reports
            summary = generate_summary_report(self.results, test_duration, self.config, self.memory_samples)
            self._save_reports(summary)

            return summary

        finally:
            # Stop server
            await self._stop_server()

    def _save_reports(self, summary: dict[str, Any]) -> None:
        """Save test reports to files.

        Args:
            summary: Test summary statistics
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results as CSV
        csv_file = self.output_dir / f"load_test_{timestamp}.csv"
        self._save_csv(csv_file)
        logger.info("Saved CSV report: %s", csv_file)

        # Save summary as text file
        summary_file = self.output_dir / f"load_test_{timestamp}_summary.txt"
        self._save_summary_text(summary_file, summary)
        logger.info("Saved summary report: %s", summary_file)

    def _save_csv(self, file_path: Path) -> None:
        """Save detailed results as CSV.

        Args:
            file_path: Path to save CSV file
        """
        import csv

        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "tool_name",
                "success",
                "latency_ms",
                "memory_rss_mb",
                "memory_vms_mb",
                "memory_percent",
                "error",
            ])

            for result in self.results:
                # Find the closest memory sample for this result
                memory_rss = ""
                memory_vms = ""
                memory_percent = ""

                if self.memory_samples:
                    # Find the closest memory sample by timestamp
                    closest_sample = min(
                        self.memory_samples,
                        key=lambda sample: abs(sample.timestamp - result.timestamp),
                    )
                    memory_rss = f"{closest_sample.rss_mb:.2f}"
                    memory_vms = f"{closest_sample.vms_mb:.2f}"
                    memory_percent = f"{closest_sample.percent:.2f}"

                writer.writerow([
                    result.timestamp,
                    result.tool_name,
                    result.success,
                    result.latency_ms,
                    memory_rss,
                    memory_vms,
                    memory_percent,
                    result.error or "",
                ])

    def _save_summary_text(self, file_path: Path, summary: dict[str, Any]) -> None:
        """Save summary report as text file.

        Args:
            file_path: Path to save summary file
            summary: Summary statistics dictionary
        """
        with open(file_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("MCP LOAD TEST SUMMARY\n")
            f.write("=" * 70 + "\n\n")

            # Test configuration
            f.write("TEST CONFIGURATION\n")
            f.write("-" * 70 + "\n")
            config_data = summary.get("test_configuration", {})
            for key, value in config_data.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")

            # Summary metrics
            f.write("SUMMARY METRICS\n")
            f.write("-" * 70 + "\n")
            summary_data = summary.get("summary", {})
            for key, value in summary_data.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")

            # Latency statistics
            f.write("LATENCY STATISTICS\n")
            f.write("-" * 70 + "\n")
            latency_data = summary.get("latency_statistics", {})
            for key, value in latency_data.items():
                f.write(f"{key.upper()}: {value:.2f} ms\n")
            f.write("\n")

            # Memory statistics
            memory_data = summary.get("memory_statistics", {})
            if memory_data:
                f.write("MEMORY STATISTICS\n")
                f.write("-" * 70 + "\n")
                for key, value in memory_data.items():
                    if isinstance(value, float):
                        f.write(f"{key.replace('_', ' ').title()}: {value:.2f} MB\n")
                    else:
                        f.write(f"{key.replace('_', ' ').title()}: {value}\n")
                f.write("\n")

            # Per-tool statistics
            f.write("PER-TOOL STATISTICS\n")
            f.write("-" * 70 + "\n")
            tool_stats = summary.get("per_tool_statistics", {})
            for tool_name, stats in tool_stats.items():
                f.write(f"\nTool: {tool_name}\n")
                for key, value in stats.items():
                    if isinstance(value, float):
                        f.write(f"  {key.replace('_', ' ').title()}: {value:.2f}\n")
                    else:
                        f.write(f"  {key.replace('_', ' ').title()}: {value}\n")

            # Errors
            errors = summary.get("errors", {})
            if errors:
                f.write("\nERRORS\n")
                f.write("-" * 70 + "\n")
                for error_msg, count in sorted(errors.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"{error_msg}: {count}\n")

            f.write("\n" + "=" * 70 + "\n")


def run_load_test(
    config_file: str,
    tool_calls: list[dict[str, Any]] | None = None,
    num_concurrent_users: int = 10,
    duration_seconds: int = 60,
    server_host: str = "localhost",
    server_port: int = 9901,
    transport: str = "streamable-http",
    warmup_seconds: int = 5,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Run an MCP load test with the specified configuration.

    Args:
        config_file: Path to NAT workflow config file
        tool_calls: List of tool call configurations.
        num_concurrent_users: Number of concurrent users to simulate
        duration_seconds: Duration of the load test
        server_host: MCP server host
        server_port: MCP server port
        transport: Transport type (streamable-http or sse)
        warmup_seconds: Warmup period before measurements
        output_dir: Output directory for reports

    Returns:
        Dictionary containing test results and statistics
    """
    # Convert tool_calls dict to ToolCallConfig objects
    if tool_calls is None:
        raise ValueError("tool_calls must be provided")

    tool_call_configs = [
        ToolCallConfig(
            tool_name=tc["tool_name"],
            args=tc.get("args", {}),
            weight=tc.get("weight", 1.0),
        ) for tc in tool_calls
    ]

    config = LoadTestConfig(
        config_file=config_file,
        tool_calls=tool_call_configs,
        num_concurrent_users=num_concurrent_users,
        duration_seconds=duration_seconds,
        server_host=server_host,
        server_port=server_port,
        transport=transport,
        warmup_seconds=warmup_seconds,
        output_dir=output_dir,
    )

    load_test = MCPLoadTest(config)
    return asyncio.run(load_test.run())


def run_load_test_from_yaml(yaml_config_path: str) -> dict[str, Any]:
    """Run an MCP load test using a YAML configuration file.

    Args:
        yaml_config_path: Path to YAML config file

    Returns:
        Dictionary containing test results and statistics
    """
    from nat.front_ends.mcp.load_test_utils.config_loader import load_config_from_yaml
    from nat.front_ends.mcp.load_test_utils.config_loader import validate_config

    config = load_config_from_yaml(yaml_config_path)
    validate_config(config)

    load_test = MCPLoadTest(config)
    return asyncio.run(load_test.run())
