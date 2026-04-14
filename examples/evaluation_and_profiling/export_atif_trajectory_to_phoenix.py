#!/usr/bin/env python3
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
"""Export an ATIF trajectory JSON file to Phoenix for visualization.

Usage::

    # Single file
    python export_atif_trajectory_to_phoenix.py trajectory.json

    # Multiple files
    python export_atif_trajectory_to_phoenix.py *.json

    # Custom endpoint and project
    python export_atif_trajectory_to_phoenix.py trajectory.json \\
        --endpoint http://localhost:6006/v1/traces \\
        --project my-project

Prerequisites:
    - A running Phoenix server (e.g. ``python -m phoenix.server.main serve``)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from nat.plugins.phoenix.atif_trajectory_phoenix_exporter import ATIFTrajectoryPhoenixExporter


def _rebase_timestamps(trajectory: dict) -> None:
    """Shift all epoch timestamps so the trajectory ends at 'now'.

    This makes traces visible in Phoenix's default time window which
    typically only shows recent activity.  ISO ``step.timestamp`` fields
    are removed since they would conflict with the rebased epochs.
    """
    import time as _time

    # Collect all epoch timestamps to find the range
    epochs: list[float] = []
    for step in trajectory.get("steps", []):
        extra = step.get("extra") or {}
        inv = extra.get("invocation") or {}
        for key in ("start_timestamp", "end_timestamp"):
            if inv.get(key):
                epochs.append(inv[key])
        for ti in extra.get("tool_invocations") or []:
            if ti:
                for key in ("start_timestamp", "end_timestamp"):
                    if ti.get(key):
                        epochs.append(ti[key])

    if not epochs:
        return

    max_epoch = max(epochs)
    offset = _time.time() - max_epoch

    # Shift all epoch timestamps
    for step in trajectory.get("steps", []):
        step.pop("timestamp", None)  # remove ISO timestamps to avoid conflicts
        extra = step.get("extra") or {}
        inv = extra.get("invocation")
        if inv:
            if inv.get("start_timestamp"):
                inv["start_timestamp"] += offset
            if inv.get("end_timestamp"):
                inv["end_timestamp"] += offset
        for ti in extra.get("tool_invocations") or []:
            if ti:
                if ti.get("start_timestamp"):
                    ti["start_timestamp"] += offset
                if ti.get("end_timestamp"):
                    ti["end_timestamp"] += offset

    # Recurse into subagent trajectories
    for sub in trajectory.get("subagent_trajectories", []):
        _rebase_timestamps(sub)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export ATIF trajectory JSON files to Phoenix for trace visualization.", )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="One or more ATIF trajectory JSON files to export.",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:6006/v1/traces",
        help="Phoenix endpoint URL (default: http://localhost:6006/v1/traces).",
    )
    parser.add_argument(
        "--project",
        default="atif-trajectories",
        help="Phoenix project name (default: atif-trajectories).",
    )
    parser.add_argument(
        "--rebase-time",
        action="store_true",
        help="Shift all timestamps to end at 'now' so traces appear in Phoenix's default time window.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    exporter = ATIFTrajectoryPhoenixExporter(
        endpoint=args.endpoint,
        project=args.project,
    )

    for path in args.files:
        if not path.exists():
            logging.error("File not found: %s", path)
            sys.exit(1)

        with open(path) as f:
            trajectory = json.load(f)

        if args.rebase_time:
            _rebase_timestamps(trajectory)

        agent_name = trajectory.get("agent", {}).get("name", "unknown")
        session_id = trajectory.get("session_id", "unknown")
        num_steps = len(trajectory.get("steps", []))

        logging.info(
            "Exporting %s  (agent=%s, steps=%d, session=%s)",
            path.name,
            agent_name,
            num_steps,
            session_id,
        )
        exporter.export(trajectory)

    logging.info("Done — open %s and select project '%s'", args.endpoint.rsplit("/v1/traces", 1)[0], args.project)


if __name__ == "__main__":
    main()
