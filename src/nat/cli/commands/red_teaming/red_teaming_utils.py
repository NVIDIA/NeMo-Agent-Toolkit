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
"""Utility functions for red team evaluation CLI."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

from nat.eval.runners.red_teaming_runner import RedTeamingRunnerConfig

logger = logging.getLogger(__name__)


def load_red_teaming_config(config_file: Path) -> RedTeamingRunnerConfig:
    """Load a RedTeamingRunnerConfig from a YAML or JSON file.

    Args:
        config_file: Path to the configuration file (YAML or JSON)

    Returns:
        Parsed RedTeamingRunnerConfig object

    Raises:
        ValueError: If the file format is invalid or parsing fails
        FileNotFoundError: If the file doesn't exist
    """
    # Ensure plugins are discovered and registered before parsing the config.
    # This triggers rebuild_annotations() which allows Pydantic to resolve
    # discriminated unions (e.g., _type: nim -> NIMConfig).
    from nat.runtime.loader import PluginTypes
    from nat.runtime.loader import discover_and_register_plugins
    discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)

    logger.info("Loading red teaming config from: %s", config_file)

    if not config_file.exists():
        raise FileNotFoundError(f"Red teaming config file not found: {config_file}")

    with open(config_file, encoding='utf-8') as f:
        if config_file.suffix in ('.yml', '.yaml'):
            config_data = yaml.safe_load(f)
        elif config_file.suffix == '.json':
            config_data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {config_file.suffix}. "
                             "Use .yml, .yaml, or .json")

    if not isinstance(config_data, dict):
        raise ValueError(f"Red teaming config file must contain a dictionary, got {type(config_data)}")

    try:
        config = RedTeamingRunnerConfig(**config_data)
    except Exception as e:
        raise ValueError(f"Failed to parse red teaming config: {e}") from e

    logger.info("Loaded red teaming config with %d scenarios", len(config.scenarios))
    return config
