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

"""Tool for analyzing Kubernetes resource pressure and capacity."""

import json
import subprocess

from pydantic.fields import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.plugins.profiler.decorators.function_tracking import track_function

from . import utils


class ResourcePressureToolConfig(FunctionBaseConfig, name="resource_pressure_check"):
    """Configuration for the Kubernetes resource pressure check tool."""

    offline_mode: bool = Field(default=True, description="Whether to run in offline mode")
    kubeconfig_path: str | None = Field(default=None, description="Path to kubeconfig file")
    cpu_threshold_percent: int = Field(
        default=80, description="CPU utilization threshold to flag as pressure"
    )
    memory_threshold_percent: int = Field(
        default=85, description="Memory utilization threshold to flag as pressure"
    )


@register_function(config_type=ResourcePressureToolConfig)
async def resource_pressure_check(config: ResourcePressureToolConfig, builder: Builder):
    """Analyze cluster resource pressure including CPU, memory, and storage utilization."""

    @track_function()
    async def _run(query: str) -> str:
        """Check for resource pressure conditions across the Kubernetes cluster.

        Analyzes CPU and memory utilization on each node, identifies nodes approaching
        capacity limits, and checks for DiskPressure, MemoryPressure, or PIDPressure conditions.

        Args:
            query: A description of what resource information to check, e.g.
                   'Check cluster resource utilization' or 'Find nodes under memory pressure'.

        Returns:
            A string containing resource utilization data, pressure conditions,
            and capacity analysis for each node.
        """
        if config.offline_mode:
            scenario_id = _extract_scenario_id(query)
            offline_response = utils.get_offline_tool_response(scenario_id, "resource_pressure_check")
            if offline_response:
                return offline_response
            return _get_default_healthy_response()

        return _run_live(
            config.kubeconfig_path,
            config.cpu_threshold_percent,
            config.memory_threshold_percent,
        )

    yield _run


def _extract_scenario_id(query: str) -> str:
    """Extract scenario_id from a query string."""
    try:
        data = json.loads(query)
        return data.get("scenario_id", "default")
    except (json.JSONDecodeError, TypeError):
        return "default"


def _run_live(
    kubeconfig_path: str | None,
    cpu_threshold: int,
    memory_threshold: int,
) -> str:
    """Execute kubectl commands to gather live resource utilization data."""
    cmd_base = ["kubectl"]
    if kubeconfig_path:
        cmd_base.extend(["--kubeconfig", kubeconfig_path])

    sections: list[str] = []

    # Get node conditions (pressure indicators)
    try:
        result = subprocess.run(
            [*cmd_base, "get", "nodes", "-o",
             "jsonpath={range .items[*]}{.metadata.name}{'\\t'}"
             "{range .status.conditions[*]}{.type}={.status}{' '}{end}{'\\n'}{end}"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        conditions = result.stdout.strip()
        pressure_nodes = []
        for line in conditions.split("\n"):
            if not line.strip():
                continue
            if any(cond in line for cond in ["MemoryPressure=True", "DiskPressure=True", "PIDPressure=True"]):
                pressure_nodes.append(line)

        if pressure_nodes:
            sections.append("## Nodes Under Pressure\n" + "\n".join(f"- {n}" for n in pressure_nodes))
        else:
            sections.append("## Node Pressure Conditions\nNo nodes reporting pressure conditions.")
    except subprocess.TimeoutExpired:
        sections.append("Error: kubectl timed out while fetching node conditions")

    # Get resource utilization via kubectl top
    try:
        result = subprocess.run(
            [*cmd_base, "top", "nodes"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        top_output = result.stdout.strip()
        sections.append(f"## Node Resource Utilization\n```\n{top_output}\n```")

        # Parse and flag nodes exceeding thresholds
        high_usage = []
        for line in top_output.split("\n")[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 5:
                node_name = parts[0]
                try:
                    cpu_pct = int(parts[2].rstrip("%"))
                    mem_pct = int(parts[4].rstrip("%"))
                    if cpu_pct > cpu_threshold:
                        high_usage.append(f"  - {node_name}: CPU at {cpu_pct}% (threshold: {cpu_threshold}%)")
                    if mem_pct > memory_threshold:
                        high_usage.append(f"  - {node_name}: Memory at {mem_pct}% (threshold: {memory_threshold}%)")
                except ValueError:
                    continue

        if high_usage:
            sections.append("## Nodes Exceeding Thresholds\n" + "\n".join(high_usage))
    except subprocess.TimeoutExpired:
        sections.append("Error: kubectl top timed out")

    return "\n\n".join(sections) if sections else "No resource pressure detected."


def _get_default_healthy_response() -> str:
    """Return a default healthy resource pressure response for offline mode."""
    return (
        "## Node Pressure Conditions\n"
        "No nodes reporting pressure conditions (MemoryPressure, DiskPressure, PIDPressure all False).\n\n"
        "## Node Resource Utilization\n"
        "All nodes operating within normal resource thresholds.\n"
        "- worker-1: CPU 45%, Memory 62%\n"
        "- worker-2: CPU 32%, Memory 48%\n\n"
        "## Nodes Exceeding Thresholds\n"
        "None."
    )
