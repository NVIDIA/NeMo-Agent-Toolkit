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

"""Utility functions for red team evaluation workflows."""

import json
import logging
from pathlib import Path

from nat.data_models.config import Config
from nat.eval.red_teaming_evaluator import RedTeamingScenarioBase
from nat.middleware.red_teaming_middleware import RedTeamingMiddlewareConfig

logger = logging.getLogger(__name__)


def load_red_team_scenarios(scenarios_file: Path) -> list[RedTeamingScenarioBase]:
    """
    Load red team scenario entries from JSON file.

    Args:
        scenarios_file: Path to JSON file containing scenario entries

    Returns:
        List of red team scenario entries

    Raises:
        ValueError: If JSON file is invalid or contains validation errors
    """
    logger.info(f"Loading red team scenarios from: {scenarios_file}")

    with open(scenarios_file, encoding='utf-8') as f:
        scenarios_data = json.load(f)

    if not isinstance(scenarios_data, list):
        raise ValueError(
            f"Red team scenarios file must contain a JSON array, got {type(scenarios_data)}"
        )

    # Parse into RedTeamingScenarioBase objects
    scenarios = []
    for idx, entry_data in enumerate(scenarios_data):
        try:
            scenario = RedTeamingScenarioBase(**entry_data)
            scenarios.append(scenario)
        except Exception as e:
            raise ValueError(
                f"Invalid scenario entry at index {idx}: {e}"
            ) from e

    # Validate: warn if multiple null middleware
    null_middleware_scenarios = [s for s in scenarios if s.middleware_name is None]
    if len(null_middleware_scenarios) > 1:
        logger.warning(
            f"Found {len(null_middleware_scenarios)} scenarios with null middleware_name "
            f"(baseline scenarios): {[s.scenario_id for s in null_middleware_scenarios]}. "
            "It's recommended to have only one baseline scenario."
        )

    logger.info(f"Loaded {len(scenarios)} scenarios successfully")
    return scenarios


def validate_base_config(config: Config) -> None:
    """
    Validate that the base configuration meets requirements for red teaming evaluation.

    Args:
        config: The workflow configuration to validate

    Raises:
        ValueError: If the config doesn't contain at least one middleware or
            doesn't contain a red_teaming_evaluator
    """
    # Validate middleware requirement
    if not config.middleware or len(config.middleware) == 0:
        raise ValueError(
            "base config must contain at least one middleware. "
            "Red teaming evaluation requires middleware to be configured."
        )
    has_red_teaming_middleware = False
    for _, middleware_config in config.middleware.items():
        if isinstance(middleware_config, RedTeamingMiddlewareConfig):
            has_red_teaming_middleware = True
            break
    if not has_red_teaming_middleware:
        raise ValueError(
            f"base config must contain at least one middleware of type "
            f"RedTeamingMiddleware. Available middleware: {list(config.middleware.keys())}"
        )

    # Check for red_teaming_evaluator
    has_red_teaming_evaluator = False
    for evaluator_name, evaluator_config in config.eval.evaluators.items():
        if hasattr(evaluator_config, 'type') and evaluator_config.type == 'red_teaming_evaluator':
            has_red_teaming_evaluator = True
            break

    if not has_red_teaming_evaluator:
        available_evaluator_types = [
            getattr(eval_config, 'type', 'unknown')
            for eval_config in config.eval.evaluators.values()
            if hasattr(eval_config, 'type')
        ]
        raise ValueError(
            "base config must contain at least one evaluator of type "
            "'red_teaming_evaluator'. "
            f"Found evaluator types: {available_evaluator_types if available_evaluator_types else 'none'}."
        )

