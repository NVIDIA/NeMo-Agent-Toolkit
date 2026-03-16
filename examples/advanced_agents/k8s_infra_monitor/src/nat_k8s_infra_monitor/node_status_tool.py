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

"""Tool for retrieving Kubernetes node status and resource utilization."""

import json
import subprocess

from pydantic.fields import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.plugins.profiler.decorators.function_tracking import track_function

from . import utils


class NodeStatusToolConfig(FunctionBaseConfig, name="node_status_check"):
    """Configuration for the Kubernetes node status check tool."""

    offline_mode: bool = Field(default=True, description="Whether to run in offline mode")
    kubeconfig_path: str | None = Field(
        default=None,
        description="Path to kubeconfig file. If None, uses default kubectl config."
    )


@register_function(config_type=NodeStatusToolConfig)
async def node_status_check(config: NodeStatusToolConfig, builder: Builder):
    """Check Kubernetes node status including conditions and resource allocation."""

    @track_function()
    async def _run(query: str) -> str:
        """Retrieve node status and resource utilization for the Kubernetes cluster.

        Args:
            query: A description of what node information to retrieve, e.g.
                   'Check all node statuses' or 'Get resource usage for worker nodes'.

        Returns:
            A string containing node status information including conditions,
            allocatable resources, and current utilization.
        """
        if config.offline_mode:
            # Extract scenario_id from query if present
            scenario_id = _extract_scenario_id(query)
            offline_response = utils.get_offline_tool_response(scenario_id, "node_status_check")
            if offline_response:
                return offline_response
            return _get_default_healthy_response()

        return _run_live(config.kubeconfig_path)

    yield _run


def _extract_scenario_id(query: str) -> str:
    """Extract scenario_id from a query string."""
    try:
        data = json.loads(query)
        return data.get("scenario_id", "default")
    except (json.JSONDecodeError, TypeError):
        return "default"


def _run_live(kubeconfig_path: str | None) -> str:
    """Execute kubectl commands to gather live node data."""
    cmd_base = ["kubectl"]
    if kubeconfig_path:
        cmd_base.extend(["--kubeconfig", kubeconfig_path])

    sections: list[str] = []

    # Get node status
    try:
        result = subprocess.run(
            [*cmd_base, "get", "nodes", "-o", "wide", "--no-headers"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if result.returncode != 0:
            sections.append(
                "Error: kubectl failed while fetching node status\n"
                f"```\n{(result.stderr or result.stdout).strip()}\n```"
            )
        else:
            sections.append(f"## Node Status\n```\n{result.stdout.strip()}\n```")
    except subprocess.TimeoutExpired:
        sections.append("Error: kubectl timed out while fetching node status")

    # Get node resource usage via top
    try:
        result = subprocess.run(
            [*cmd_base, "top", "nodes", "--no-headers"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if result.returncode != 0:
            sections.append(
                "Error: kubectl top failed\n"
                f"```\n{(result.stderr or result.stdout).strip()}\n```"
            )
        else:
            sections.append(f"## Node Resource Usage\n```\n{result.stdout.strip()}\n```")
    except subprocess.TimeoutExpired:
        sections.append("Error: kubectl top timed out")

    return "\n\n".join(sections)


def _get_default_healthy_response() -> str:
    """Return a default healthy node status response for offline mode."""
    return (
        "## Node Status\n"
        "All 3 nodes are in Ready state.\n"
        "- control-plane-1: Ready, SchedulingDisabled (control-plane taint)\n"
        "- worker-1: Ready, 6 vCPU, 30Gi RAM, 74% CPU, 62% memory\n"
        "- worker-2: Ready, 6 vCPU, 20Gi RAM, 45% CPU, 51% memory\n\n"
        "## Node Resource Usage\n"
        "No resource pressure conditions detected on any node."
    )
