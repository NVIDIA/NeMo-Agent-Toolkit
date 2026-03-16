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

"""Tool for checking Kubernetes pod health across namespaces."""

import json
import subprocess

from pydantic.fields import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.plugins.profiler.decorators.function_tracking import track_function

from . import utils


class PodHealthToolConfig(FunctionBaseConfig, name="pod_health_check"):
    """Configuration for the Kubernetes pod health check tool."""

    offline_mode: bool = Field(default=True, description="Whether to run in offline mode")
    kubeconfig_path: str | None = Field(default=None, description="Path to kubeconfig file")
    namespaces: list[str] | None = Field(
        default=None,
        description="Specific namespaces to check. If None, checks all namespaces."
    )


@register_function(config_type=PodHealthToolConfig)
async def pod_health_check(config: PodHealthToolConfig, builder: Builder):
    """Check pod health across Kubernetes namespaces, identifying unhealthy or restarting pods."""

    @track_function()
    async def _run(query: str) -> str:
        """Retrieve pod health information across cluster namespaces.

        Identifies pods that are not in Running/Succeeded state, pods with high restart counts,
        and pods stuck in pending or crash-looping states.

        Args:
            query: A description of what pod health information to retrieve, e.g.
                   'Check for unhealthy pods' or 'Find pods with high restart counts'.

        Returns:
            A string containing pod health data including unhealthy pods,
            restart counts, and container status details.
        """
        if config.offline_mode:
            scenario_id = _extract_scenario_id(query)
            offline_response = utils.get_offline_tool_response(scenario_id, "pod_health_check")
            if offline_response:
                return offline_response
            return _get_default_healthy_response()

        return _run_live(config.kubeconfig_path, config.namespaces)

    yield _run


def _extract_scenario_id(query: str) -> str:
    """Extract scenario_id from a query string."""
    try:
        data = json.loads(query)
        return data.get("scenario_id", "default")
    except (json.JSONDecodeError, TypeError):
        return "default"


def _run_live(kubeconfig_path: str | None, namespaces: list[str] | None) -> str:
    """Execute kubectl commands to gather live pod health data."""
    cmd_base = ["kubectl"]
    if kubeconfig_path:
        cmd_base.extend(["--kubeconfig", kubeconfig_path])

    ns_flag = ["--all-namespaces"] if not namespaces else []
    sections: list[str] = []

    if namespaces:
        for ns in namespaces:
            try:
                result = subprocess.run(
                    [*cmd_base, "get", "pods", "-n", ns, "-o", "wide", "--no-headers"],
                    capture_output=True, text=True, timeout=30, check=False,
                )
                sections.append(f"### Namespace: {ns}\n```\n{result.stdout.strip()}\n```")
            except subprocess.TimeoutExpired:
                sections.append(f"### Namespace: {ns}\nError: kubectl timed out")
    else:
        try:
            result = subprocess.run(
                [*cmd_base, "get", "pods", *ns_flag, "-o", "wide", "--no-headers",
                 "--field-selector=status.phase!=Running,status.phase!=Succeeded"],
                capture_output=True, text=True, timeout=30, check=False,
            )
            unhealthy = result.stdout.strip()
            if unhealthy:
                sections.append(f"## Unhealthy Pods\n```\n{unhealthy}\n```")
            else:
                sections.append("## Unhealthy Pods\nNo unhealthy pods found across all namespaces.")
        except subprocess.TimeoutExpired:
            sections.append("Error: kubectl timed out while fetching pod status")

    # Check for pods with high restart counts (> 5)
    try:
        result = subprocess.run(
            [*cmd_base, "get", "pods", "--all-namespaces", "-o",
             "jsonpath={range .items[*]}{.metadata.namespace}{' '}"
             "{.metadata.name}{' '}{range .status.containerStatuses[*]}"
             "{.restartCount}{' '}{end}{'\\n'}{end}"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        high_restarts = []
        for line in result.stdout.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 3:
                ns, name = parts[0], parts[1]
                restarts = [int(r) for r in parts[2:] if r.isdigit()]
                if any(r > 5 for r in restarts):
                    high_restarts.append(f"  {ns}/{name}: {max(restarts)} restarts")
        if high_restarts:
            sections.append("## High Restart Pods\n" + "\n".join(high_restarts))
    except (subprocess.TimeoutExpired, ValueError):
        pass

    return "\n\n".join(sections) if sections else "All pods are healthy across all namespaces."


def _get_default_healthy_response() -> str:
    """Return a default healthy pod status response for offline mode."""
    return (
        "## Pod Health Summary\n"
        "All pods are in Running or Succeeded state across all namespaces.\n\n"
        "## High Restart Pods\n"
        "No pods with excessive restart counts detected."
    )
