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
"""Configuration loader for MCP load tests."""

import logging
from pathlib import Path
from typing import Any

import yaml

from nat.front_ends.mcp.load_test_utils.load_tester import LoadTestConfig
from nat.front_ends.mcp.load_test_utils.load_tester import ToolCallConfig

logger = logging.getLogger(__name__)


def load_config_from_yaml(config_path: str | Path) -> LoadTestConfig:
    """Load load test configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        LoadTestConfig object

    Raises:
        FileNotFoundError: If the config file doesn't exist
        ValueError: If the config file is invalid

    Example YAML structure:
        ```yaml
        # Path to NAT workflow config file
        config_file: "examples/getting_started/simple_calculator/configs/config.yml"

        # Server configuration
        server:
          host: "localhost"
          port: 9901
          transport: "streamable-http"  # or "sse"

        # Load test parameters
        load_test:
          num_concurrent_users: 10
          duration_seconds: 30
          warmup_seconds: 5

        # Output configuration
        output:
          directory: "load_test_results"

        # Tool calls to execute
        tool_calls:
          - tool_name: "calculator_multiply"
            args:
              text: "2 * 3"
            weight: 2.0
          - tool_name: "calculator_divide"
            args:
              text: "10 / 2"
            weight: 1.0
        ```
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info("Loading config from: %s", config_path)

    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    if not config_data:
        raise ValueError("Config file is empty")

    # Extract required field
    config_file = config_data.get("config_file")
    if not config_file:
        raise ValueError("'config_file' is required in the config")

    # Extract server configuration
    server_config = config_data.get("server", {})
    server_host = server_config.get("host", "localhost")
    server_port = server_config.get("port", 9901)
    transport = server_config.get("transport", "streamable-http")

    # Extract load test parameters
    load_test_config = config_data.get("load_test", {})
    num_concurrent_users = load_test_config.get("num_concurrent_users", 10)
    duration_seconds = load_test_config.get("duration_seconds", 60)
    warmup_seconds = load_test_config.get("warmup_seconds", 5)

    # Extract output configuration
    output_config = config_data.get("output", {})
    output_dir = output_config.get("directory", None)

    # Extract tool calls
    tool_calls_data = config_data.get("tool_calls", [])
    if not tool_calls_data:
        raise ValueError("At least one tool call must be specified in 'tool_calls'")

    tool_calls = []
    for tc in tool_calls_data:
        if "tool_name" not in tc:
            raise ValueError("Each tool call must have 'tool_name'")

        tool_calls.append(
            ToolCallConfig(
                tool_name=tc["tool_name"],
                args=tc.get("args", {}),
                weight=tc.get("weight", 1.0),
            ))

    return LoadTestConfig(
        config_file=config_file,
        tool_calls=tool_calls,
        num_concurrent_users=num_concurrent_users,
        duration_seconds=duration_seconds,
        server_host=server_host,
        server_port=server_port,
        transport=transport,
        warmup_seconds=warmup_seconds,
        output_dir=output_dir,
    )


def validate_config(config: LoadTestConfig) -> None:
    """Validate load test configuration.

    Args:
        config: LoadTestConfig to validate

    Raises:
        ValueError: If configuration is invalid
    """
    if not config.config_file:
        raise ValueError("config_file must be specified")

    if not Path(config.config_file).exists():
        raise ValueError(f"NAT workflow config file not found: {config.config_file}")

    if config.num_concurrent_users < 1:
        raise ValueError("num_concurrent_users must be at least 1")

    if config.duration_seconds < 1:
        raise ValueError("duration_seconds must be at least 1")

    if config.warmup_seconds < 0:
        raise ValueError("warmup_seconds must be non-negative")

    if not config.tool_calls:
        raise ValueError("At least one tool call must be specified")

    if config.transport not in ["streamable-http", "sse"]:
        raise ValueError("transport must be 'streamable-http' or 'sse'")

    if config.server_port < 1 or config.server_port > 65535:
        raise ValueError("server_port must be between 1 and 65535")

    logger.info("Configuration validated successfully")
