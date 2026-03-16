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

"""Tool for collecting and analyzing Kubernetes cluster events."""

import json
import subprocess

from pydantic.fields import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.plugins.profiler.decorators.function_tracking import track_function

from . import utils


class EventCollectorToolConfig(FunctionBaseConfig, name="event_collector"):
    """Configuration for the Kubernetes event collector tool."""

    offline_mode: bool = Field(default=True, description="Whether to run in offline mode")
    kubeconfig_path: str | None = Field(default=None, description="Path to kubeconfig file")
    event_limit: int = Field(default=50, gt=0, description="Maximum number of recent events to retrieve")


@register_function(config_type=EventCollectorToolConfig)
async def event_collector(config: EventCollectorToolConfig, builder: Builder):
    """Collect recent Kubernetes events, focusing on warnings and errors."""

    @track_function()
    async def _run(query: str) -> str:
        """Retrieve recent Kubernetes cluster events filtered by severity.

        Collects Warning and Normal events from the cluster, useful for diagnosing
        scheduling failures, image pull errors, OOM kills, and other operational issues.

        Args:
            query: A description of what events to look for, e.g.
                   'Get recent warning events' or 'Check events in namespace monitoring'.

        Returns:
            A string containing formatted cluster events grouped by type,
            with timestamps and affected resources.
        """
        if config.offline_mode:
            scenario_id = _extract_scenario_id(query)
            offline_response = utils.get_offline_tool_response(scenario_id, "event_collector")
            if offline_response:
                return offline_response
            return _get_default_healthy_response()

        return _run_live(config.kubeconfig_path, config.event_limit)

    yield _run


def _extract_scenario_id(query: str) -> str:
    """Extract scenario_id from a query string."""
    try:
        data = json.loads(query)
        return data.get("scenario_id", "default")
    except (json.JSONDecodeError, TypeError):
        return "default"


def _run_live(kubeconfig_path: str | None, event_limit: int) -> str:
    """Execute kubectl commands to gather live cluster events."""
    cmd_base = ["kubectl"]
    if kubeconfig_path:
        cmd_base.extend(["--kubeconfig", kubeconfig_path])

    sections: list[str] = []

    # Get warning events
    try:
        result = subprocess.run(
            [*cmd_base, "get", "events", "--all-namespaces",
             "--field-selector=type=Warning",
             "--sort-by=.lastTimestamp",
             "--no-headers"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if result.returncode != 0:
            sections.append(
                "Error: kubectl failed while fetching warning events\n"
                f"```\n{(result.stderr or result.stdout).strip()}\n```"
            )
        else:
            warnings = result.stdout.strip()
            if warnings:
                lines = warnings.split("\n")[:event_limit]
                sections.append(f"## Warning Events ({len(lines)} most recent)\n```\n" + "\n".join(lines) + "\n```")
            else:
                sections.append("## Warning Events\nNo warning events found.")
    except subprocess.TimeoutExpired:
        sections.append("Error: kubectl timed out while fetching events")

    # Get recent events summary
    try:
        result = subprocess.run(
            [*cmd_base, "get", "events", "--all-namespaces",
             "--sort-by=.lastTimestamp",
             "-o", "custom-columns="
             "NAMESPACE:.metadata.namespace,"
             "TYPE:.type,"
             "REASON:.reason,"
             "OBJECT:.involvedObject.kind/.involvedObject.name,"
             "MESSAGE:.message",
             "--no-headers"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if result.returncode != 0:
            sections.append(
                "Error: kubectl failed while fetching recent events\n"
                f"```\n{(result.stderr or result.stdout).strip()}\n```"
            )
        elif result.stdout.strip():
            lines = result.stdout.strip().split("\n")[:event_limit]
            sections.append(
                f"## Recent Events ({len(lines)} most recent)\n```\n" + "\n".join(lines) + "\n```"
            )
    except subprocess.TimeoutExpired:
        pass

    return "\n\n".join(sections) if sections else "No cluster events found."


def _get_default_healthy_response() -> str:
    """Return a default healthy event response for offline mode."""
    return (
        "## Warning Events\n"
        "No warning events found.\n\n"
        "## Recent Events\n"
        "Recent events are routine: Pulled, Created, Started, Scheduled. "
        "No abnormal patterns detected."
    )
