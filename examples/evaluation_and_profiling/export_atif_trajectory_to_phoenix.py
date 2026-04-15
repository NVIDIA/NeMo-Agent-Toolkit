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
