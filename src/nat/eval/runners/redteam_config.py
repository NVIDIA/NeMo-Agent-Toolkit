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

import logging
import typing
from pathlib import Path

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from nat.data_models.config import Config
from nat.eval.config import EvaluationRunConfig
from nat.eval.red_teaming_evaluator.filter_conditions import IntermediateStepsFilterCondition

logger = logging.getLogger(__name__)


class InterceptScenarioEntry(BaseModel):
    """
    A single intercept scenario entry from the JSON dataset.

    Each entry defines how to configure one intercept for one target function/function_group.

    Attributes:
        scenario_id: Unique identifier for this scenario
        intercept_name: Name of the intercept to apply, or None for baseline (no intercepts)
        target_function: Name of the function to apply the intercept to
            (mutually exclusive with target_function_group)
        target_function_group: Name of the function group to apply the intercept to
            (mutually exclusive with target_function)
        payload: Payload value for the intercept (only used when intercept_name is not None)
        evaluation_instructions: Optional scenario-specific instructions for the evaluator to check
            if the intercept was successful in producing the expected behavior
    """
    scenario_id: str = Field(description="Unique identifier for this scenario")
    intercept_name: str | None = Field(
        default=None,
        description="Name of the intercept to apply, or null for baseline scenario"
    )
    target_function: str | None = Field(
        default=None,
        description="Name of the function to apply the intercept to"
    )
    target_function_group: str | None = Field(
        default=None,
        description="Name of the function group to apply the intercept to"
    )
    payload: typing.Any = Field(
        default=None,
        description="Payload value for the intercept"
    )
    evaluation_instructions: str | None = Field(
        default=None,
        description="Scenario-specific instructions for evaluating whether the intercept produced the expected behavior"
    )
    filter_conditions: list[IntermediateStepsFilterCondition] | None = Field(
        default=None,
        description="Optional filter conditions for selecting specific intermediate steps to evaluate"
    )

    @model_validator(mode='after')
    def validate_entry(self) -> 'InterceptScenarioEntry':
        """Validate the entry configuration."""
        # If intercept_name is null, this is a baseline scenario
        if self.intercept_name is None:
            if self.target_function is not None or self.target_function_group is not None:
                raise ValueError(
                    f"Scenario '{self.scenario_id}': When intercept_name is null (baseline), "
                    "target_function and target_function_group must also be null"
                )
            return self

        # If intercept_name is provided, exactly one target must be specified
        if self.target_function is None and self.target_function_group is None:
            raise ValueError(
                f"Scenario '{self.scenario_id}': When intercept_name is provided, "
                "either target_function or target_function_group must be specified"
            )

        if self.target_function is not None and self.target_function_group is not None:
            raise ValueError(
                f"Scenario '{self.scenario_id}': Cannot specify both target_function "
                "and target_function_group. Use only one."
            )

        return self

    def apply_to_config(self, config: Config) -> Config:
        """
        Apply this scenario to a workflow configuration.

        This method performs all necessary transformations including:
        1. Applying intercept overrides (if intercept_name is specified)
        2. Injecting evaluation instructions (if evaluation_instructions is specified)
        3. Injecting filter conditions (if filter_conditions is specified)

        Args:
            config: The workflow configuration to transform

        Returns:
            Transformed configuration with scenario applied

        Note:
            This method modifies the config in-place and returns it.
        """
        logger.info(f"Applying scenario '{self.scenario_id}' to configuration")

        # Apply intercept override if this isn't a baseline scenario
        if self.intercept_name is not None:
            self._apply_intercept_override(config)

        # Inject evaluation instructions and/or filter conditions if provided
        if self.evaluation_instructions is not None or self.filter_conditions is not None:
            self._inject_evaluation_config(config)

        return config

    def _apply_intercept_override(self, config: Config) -> None:
        """
        Apply intercept override to the configuration.

        This method:
        1. Validates that the intercept exists
        2. Removes the intercept from all functions/function_groups (clear overlaps)
        3. Applies the intercept only to the specified target
        4. Updates the intercept's payload

        Args:
            config: Configuration to modify (modified in-place)

        Raises:
            ValueError: If intercept or target doesn't exist in config
        """
        # Handle baseline scenario (null intercept)
        if self.intercept_name is None:
            logger.info(
                f"Baseline scenario '{self.scenario_id}': using config as-is (no intercept modifications)"
            )
            return

        # Validate that the intercept exists in function_intercepts
        if self.intercept_name not in config.function_intercepts:
            raise ValueError(
                f"Scenario '{self.scenario_id}': Intercept '{self.intercept_name}' "
                f"not found in function_intercepts. Available intercepts: "
                f"{list(config.function_intercepts.keys())}"
            )

        # Step 1: Clear the intercept from all functions and function groups
        logger.info(f"Clearing intercept '{self.intercept_name}' from all functions/groups")
        for func_name, func_config in config.functions.items():
            if self.intercept_name in func_config.intercepts:
                func_config.intercepts = [
                    ic for ic in func_config.intercepts if ic != self.intercept_name
                ]
                logger.debug(f"Removed '{self.intercept_name}' from function '{func_name}'")

        for group_name, group_config in config.function_groups.items():
            if self.intercept_name in group_config.intercepts:
                group_config.intercepts = [
                    ic for ic in group_config.intercepts if ic != self.intercept_name
                ]
                logger.debug(f"Removed '{self.intercept_name}' from function group '{group_name}'")

        # Step 2: Apply the intercept to the target
        if self.target_function is not None:
            # Target is a function
            if self.target_function not in config.functions:
                raise ValueError(
                    f"Scenario '{self.scenario_id}': Target function '{self.target_function}' "
                    f"not found in config. Available functions: {list(config.functions.keys())}"
                )

            logger.info(f"Adding intercept '{self.intercept_name}' to function '{self.target_function}'")
            target_config = config.functions[self.target_function]
            if self.intercept_name not in target_config.intercepts:
                target_config.intercepts.append(self.intercept_name)

        elif self.target_function_group is not None:
            # Target is a function group
            if self.target_function_group not in config.function_groups:
                raise ValueError(
                    f"Scenario '{self.scenario_id}': Target function group '{self.target_function_group}' "
                    f"not found in config. Available function groups: {list(config.function_groups.keys())}"
                )

            logger.info(
                f"Adding intercept '{self.intercept_name}' to function group '{self.target_function_group}'"
            )
            target_config = config.function_groups[self.target_function_group]
            if self.intercept_name not in target_config.intercepts:
                target_config.intercepts.append(self.intercept_name)

        # Step 3: Update the intercept's payload if provided
        if self.payload is not None:
            logger.info(f"Updating payload for intercept '{self.intercept_name}': {self.payload}")
            intercept_config = config.function_intercepts[self.intercept_name]
            config_dict = intercept_config.model_dump()

            # Update with the payload
            if isinstance(self.payload, dict):
                # If payload is a dict, merge it into the config
                config_dict.update(self.payload)
            else:
                # Otherwise, assume it's a value for the 'payload' field
                config_dict['payload'] = self.payload

            # Recreate the intercept config from the updated dict
            config_type = type(intercept_config)
            config.function_intercepts[self.intercept_name] = config_type(**config_dict)

    def _inject_evaluation_config(self, config: Config) -> None:
        """
        Inject scenario-specific evaluation configuration into red_teaming_evaluator configs.

        This includes evaluation instructions and filter conditions.

        Args:
            config: Configuration to modify (modified in-place)
        """
        logger.info("Injecting scenario-specific evaluation configuration")

        # Check if eval config exists
        if not config.eval or not config.eval.evaluators:
            logger.warning("No evaluators found in config. Skipping scenario config injection.")
            return

        # Find and update red_teaming_evaluator if it exists.
        for evaluator_name, evaluator_config in config.eval.evaluators.items():
            # Check if this is a red_teaming_evaluator
            if hasattr(evaluator_config, 'type') and evaluator_config.type == 'red_teaming_evaluator':
                logger.info(f"Updating evaluator '{evaluator_name}' with scenario-specific configuration")

                # Get the config as a dict
                config_dict = evaluator_config.model_dump()

                # Update scenario_specific_instructions if provided
                if self.evaluation_instructions is not None:
                    config_dict['scenario_specific_instructions'] = self.evaluation_instructions
                    logger.debug(f"Injected evaluation instructions into '{evaluator_name}'")

                # Update filter_conditions if provided
                if self.filter_conditions is not None:
                    # Convert filter conditions to dicts for serialization
                    config_dict['filter_conditions'] = [
                        fc.model_dump() for fc in self.filter_conditions
                    ]
                    logger.debug(f"Injected {len(self.filter_conditions)} filter conditions into '{evaluator_name}'")

                # Recreate the evaluator config
                config_type = type(evaluator_config)
                config.eval.evaluators[evaluator_name] = config_type(**config_dict)

                logger.debug(f"Successfully injected configuration into evaluator '{evaluator_name}'")


class RedTeamingEvaluationConfig(BaseModel):
    """
    Configuration for red teaming evaluation runs with function intercepts.

    This config allows running multiple evaluation scenarios where each scenario
    tests different function intercept configurations. Scenarios are defined in a
    JSON file similar to an evaluation dataset.

    Attributes:
        base_evaluation_config: Base evaluation configuration that will be modified
            for each red teaming scenario.
        intercept_scenarios_file: Path to JSON file containing intercept scenario entries.
            Each entry defines how to configure intercepts for one test scenario.
    """
    base_evaluation_config: EvaluationRunConfig
    intercept_scenarios_file: Path = Field(
        description="Path to JSON file containing intercept scenario entries"
    )

    @model_validator(mode='after')
    def validate_file_exists(self) -> 'RedTeamingEvaluationConfig':
        """Validate that the scenarios file exists."""
        if not self.intercept_scenarios_file.exists():
            raise FileNotFoundError(
                f"Intercept scenarios file not found: {self.intercept_scenarios_file}"
            )
        if not self.intercept_scenarios_file.is_file():
            raise ValueError(
                f"Intercept scenarios path is not a file: {self.intercept_scenarios_file}"
            )
        return self
