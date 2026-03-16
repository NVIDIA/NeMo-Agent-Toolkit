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

import json
import logging
from pathlib import Path

logger = logging.getLogger("nat_k8s_infra_monitor")

# Offline data storage — populated at startup when running in offline mode
_offline_data: dict[str, dict] = {}


def log_header(text: str, dash_length: int = 80, level: int = logging.INFO) -> None:
    """Log a formatted header."""
    logger.log(level, "-" * dash_length)
    logger.log(level, text)
    logger.log(level, "-" * dash_length)


def preload_offline_data(offline_data_path: str | None) -> int:
    """Load offline scenario data from a JSON file into memory.

    The JSON file should contain a list of scenario objects, each with a ``scenario_id``
    key and tool response data keyed by tool name.

    Returns:
        The number of scenarios loaded.
    """
    _offline_data.clear()

    if offline_data_path is None:
        return 0

    path = Path(offline_data_path)
    if not path.exists():
        logger.warning("Offline data file not found: %s", offline_data_path)
        return 0

    with open(path, encoding="utf-8") as fh:
        scenarios = json.load(fh)

    for scenario in scenarios:
        scenario_id = scenario.get("scenario_id", "")
        _offline_data[scenario_id] = scenario

    logger.info("Loaded %d offline scenarios from %s", len(_offline_data), offline_data_path)
    return len(_offline_data)


def get_offline_tool_response(scenario_id: str, tool_name: str) -> str | None:
    """Retrieve a cached offline response for a given scenario and tool."""
    scenario = _offline_data.get(scenario_id)
    if scenario is None:
        return None
    return scenario.get(tool_name)
