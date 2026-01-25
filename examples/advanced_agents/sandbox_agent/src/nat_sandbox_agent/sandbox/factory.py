# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Factory for creating sandbox instances."""

import os
from enum import Enum
from typing import Literal

from pydantic import BaseModel
from pydantic import Field

from nat_sandbox_agent.sandbox.base import BaseSandbox

# Environment variables to automatically pass to sandbox if set on host
DEFAULT_PASS_ENV_VARS = ["TAVILY_API_KEY"]


class SandboxType(str, Enum):
    """Supported sandbox types."""

    DOCKER = "docker"
    DAYTONA = "daytona"


class DockerSandboxConfig(BaseModel):
    """Configuration for Docker sandbox."""

    type: Literal["docker"] = "docker"
    image: str = Field(
        default="python:3.12-slim",
        description="Docker image to use for the sandbox.",
    )
    memory_limit: str = Field(
        default="512m",
        description="Memory limit for the container (e.g., '512m', '1g').",
    )
    cpu_limit: float = Field(
        default=1.0,
        description="CPU limit (number of CPUs).",
    )
    network_enabled: bool = Field(
        default=True,
        description="Whether to enable network access in the sandbox.",
    )
    work_dir: str = Field(
        default="/workspace",
        description="Working directory inside the container.",
    )
    auto_remove: bool = Field(
        default=False,
        description="Whether to auto-remove container when stopped.",
    )
    environment: dict[str, str] | None = Field(
        default=None,
        description="Environment variables to pass to the container.",
    )
    pass_env_vars: list[str] | None = Field(
        default=None,
        description="List of env var names to pass from host to container.",
    )
    volumes: dict[str, str] | None = Field(
        default=None,
        description="Volume mounts as dict {host_path: container_path}.",
    )


class DaytonaSandboxConfig(BaseModel):
    """Configuration for Daytona sandbox."""

    type: Literal["daytona"] = "daytona"
    api_key: str = Field(
        ...,
        description="Daytona API key for authentication.",
    )
    server_url: str = Field(
        default="https://api.daytona.io",
        description="Daytona API server URL.",
    )
    target: str = Field(
        default="us",
        description="Target region (e.g., 'us', 'eu').",
    )
    image: str = Field(
        default="daytonaio/workspace:latest",
        description="Docker image for the Daytona workspace.",
    )
    cpu: int = Field(
        default=2,
        description="Number of CPU cores.",
    )
    memory: int = Field(
        default=4,
        description="Memory in GB.",
    )
    disk: int = Field(
        default=10,
        description="Disk space in GB.",
    )
    auto_stop_interval: int = Field(
        default=30,
        description="Auto-stop interval in minutes (0 to disable).",
    )


# Union type for sandbox configuration
SandboxConfig = DockerSandboxConfig | DaytonaSandboxConfig


def _build_environment(config: DockerSandboxConfig) -> dict[str, str]:
    """Build environment variables dict for sandbox.

    Combines explicit environment vars with pass-through vars from host.
    """
    env = dict(config.environment or {})

    # Determine which env vars to pass from host
    pass_vars = config.pass_env_vars if config.pass_env_vars is not None else DEFAULT_PASS_ENV_VARS

    for var_name in pass_vars:
        if var_name not in env:  # Don't override explicit values
            value = os.environ.get(var_name)
            if value:
                env[var_name] = value

    return env


def create_sandbox(config: SandboxConfig) -> BaseSandbox:
    """Create a sandbox instance based on configuration.

    Args:
        config: Sandbox configuration (Docker or Daytona).

    Returns:
        BaseSandbox: A sandbox instance ready to be started.

    Raises:
        ValueError: If the sandbox type is not supported.
    """
    if config.type in (SandboxType.DOCKER, "docker"):
        from nat_sandbox_agent.sandbox.docker_sandbox import DockerSandbox

        environment = _build_environment(config)

        # Convert volumes from {host: container} to Docker format
        volumes = None
        if config.volumes:
            volumes = {
                host_path: {"bind": container_path, "mode": "rw"}
                for host_path, container_path in config.volumes.items()
            }

        return DockerSandbox(
            image=config.image,
            memory_limit=config.memory_limit,
            cpu_limit=config.cpu_limit,
            network_enabled=config.network_enabled,
            work_dir=config.work_dir,
            auto_remove=config.auto_remove,
            environment=environment,
            volumes=volumes,
        )

    elif config.type in (SandboxType.DAYTONA, "daytona"):
        from nat_sandbox_agent.sandbox.daytona_sandbox import DaytonaSandbox

        return DaytonaSandbox(
            api_key=config.api_key,
            server_url=config.server_url,
            target=config.target,
            image=config.image,
            cpu=config.cpu,
            memory=config.memory,
            disk=config.disk,
            auto_stop_interval=config.auto_stop_interval,
        )

    else:
        raise ValueError(f"Unknown sandbox type: {config.type}")


def create_sandbox_from_dict(config_dict: dict) -> BaseSandbox:
    """Create a sandbox instance from a dictionary configuration.

    Args:
        config_dict: Dictionary with sandbox configuration.
            Must include a 'type' key with value 'docker' or 'daytona'.

    Returns:
        BaseSandbox: A sandbox instance ready to be started.
    """
    sandbox_type = config_dict.get("type", "docker")

    if sandbox_type == "docker":
        config = DockerSandboxConfig(**config_dict)
    elif sandbox_type == "daytona":
        config = DaytonaSandboxConfig(**config_dict)
    else:
        raise ValueError(f"Unknown sandbox type: {sandbox_type}")

    return create_sandbox(config)
