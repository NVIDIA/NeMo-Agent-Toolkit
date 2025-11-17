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


class RedTeamScenarioEntry(BaseModel):
    """
    A single intercept scenario entry from the JSON dataset.

    Each entry defines how to configure one intercept for one target function/function_group.

    Attributes:
        scenario_id: Unique identifier for this scenario
        middleware_name: Name of the middleware to apply, or None for baseline (no middleware)
        target_function_or_group: Name of the function or function group to apply the middleware to.
            The system will automatically determine whether it's a function or function group
            by checking both namespaces. If None, this is a baseline scenario (no middleware).
        middleware_config_override: Parts of the middleware config to override.
            Keys are the middleware config keys to override, and should be compatible
            with the middleware defined by middleware_name.
            If None, the middleware config is not modified.
        evaluation_instructions: Optional scenario-specific instructions for the evaluator to check
            if the middleware was successful in producing the expected behavior
    """
    scenario_id: str = Field(description="Unique identifier for this scenario")
    middleware_name: str | None = Field(
        default=None,
        description="Name of the middleware to apply, or null for baseline scenario"
    )
    target_function_or_group: str | None = Field(
        default=None,
        description="Name of the function or function group to apply the middleware to. "
        "The system will automatically determine whether it's a function or function group "
        "by checking both namespaces. If None, this is a baseline scenario (no middleware)."
    )
    middleware_config_override: dict[str, typing.Any] | None = Field(
        default=None,
        description="Optional configuration overrides for the middleware."
        "Keys are the middleware config keys to override, and should be compatible"
        "with the middleware defined by middleware_name."
    )
    evaluation_instructions: str | None = Field(
        default=None,
        description="Scenario-specific instructions for evaluating whether the scenario produced the expected behavior"
    )
    filter_conditions: list[IntermediateStepsFilterCondition] | None = Field(
        default=None,
        description="Optional filter conditions for selecting specific intermediate steps to evaluate"
    )

    @model_validator(mode='after')
    def validate_entry(self) -> 'RedTeamScenarioEntry':
        """Validate the entry configuration."""
        # If middleware_name is null, this is a baseline scenario
        if self.middleware_name is None:
            if self.target_function_or_group is not None:
                raise ValueError(
                    f"Scenario '{self.scenario_id}': When middleware_name is null (baseline), "
                    "target_function_or_group must also be null"
                )
            return self

        # If middleware_name is provided, target must be specified
        if self.target_function_or_group is None:
            raise ValueError(
                f"Scenario '{self.scenario_id}': When middleware_name is provided, "
                "target_function_or_group must be specified"
            )

        return self

    def apply_to_config(self, config: Config) -> Config:
        """
        Apply this scenario to a workflow configuration.

        This method performs all necessary transformations including:
        1. Applying middleware overrides (if middleware_name is specified)
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
        if self.middleware_name is not None:
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
        4. Updates the intercept's configuration. Using middleware_config_override if provided.

        Args:
            config: Configuration to modify (modified in-place)

        Raises:
            ValueError: If intercept or target doesn't exist in config
        """
        # Handle baseline scenario (null intercept)
        if self.middleware_name is None:
            logger.info(
                f"Baseline scenario '{self.scenario_id}': using config as-is (no intercept modifications)"
            )
            return
        if not config.middleware:
            raise ValueError(
                "No middleware found in base workflow configuration."
                "Red teaming evaluation requires at least one middleware."
            )
        # Validate that the intercept exists in middleware
        if self.middleware_name not in config.middleware:
            raise ValueError(
                f"Scenario '{self.scenario_id}': Middleware '{self.middleware_name}' "
                f"not found in middleware. Available middleware: "
                f"{list(config.middleware.keys())}"
            )

        # Step 1: Clear the intercept from all functions and function groups
        logger.info(f"Clearing middleware '{self.middleware_name}' from all functions/groups")
        for func_name, func_config in config.functions.items():
            if self.middleware_name in func_config.middleware:
                func_config.middleware = [
                    mc for mc in func_config.middleware if mc != self.middleware_name
                ]
                logger.debug(f"Removed '{self.middleware_name}' from function '{func_name}'")

        for group_name, group_config in config.function_groups.items():
            if self.middleware_name in group_config.middleware:
                group_config.middleware = [
                    mc for mc in group_config.middleware if mc != self.middleware_name
                ]
                logger.debug(f"Removed '{self.middleware_name}' from function group '{group_name}'")

        # Step 2: Apply the intercept to the target
        # Determine if target is a function or function group by checking both namespaces
        target_name = self.target_function_or_group
        is_function = target_name in config.functions
        is_function_group = target_name in config.function_groups

        if is_function and is_function_group:
            # Ambiguity: name exists in both namespaces
            raise ValueError(
                f"Scenario '{self.scenario_id}': Target '{target_name}' exists in both "
                "functions and function_groups. This creates ambiguity. Please ensure "
                "function and function group names are unique."
            )

        if is_function:
            # Target is a function
            logger.info(f"Adding middleware '{self.middleware_name}' to function '{target_name}'")
            target_config = config.functions[target_name]
            if self.middleware_name not in target_config.middleware:
                target_config.middleware.append(self.middleware_name)

        elif is_function_group:
            # Target is a function group
            logger.info(
                f"Adding middleware '{self.middleware_name}' to function group '{target_name}'"
            )
            target_config = config.function_groups[target_name]
            if self.middleware_name not in target_config.middleware:
                target_config.middleware.append(self.middleware_name)

        else:
            # Target not found in either namespace
            available_functions = list(config.functions.keys())
            available_groups = list(config.function_groups.keys())
            raise ValueError(
                f"Scenario '{self.scenario_id}': Target '{target_name}' not found in config. "
                f"Available functions: {available_functions}, "
                f"Available function groups: {available_groups}"
            )

        # Step 3: Update the intercept's payload if provided
        if self.middleware_config_override is not None:
            logger.info(f"Updating payload for middleware '{self.middleware_name}': {self.middleware_config_override}")
            middleware_config = config.middleware[self.middleware_name]
            config_dict = middleware_config.model_dump(exclude_unset=False)

            # Update config_dict using middleware_config_override,
            # but raise an error if override contains a key not in config_dict.
            if self.middleware_config_override:
                for key, value in self.middleware_config_override.items():
                    if key not in config_dict:
                        raise KeyError(
                            "middleware_config_override contains key '{key}' not present in the middleware config"
                            f"for '{self.middleware_name}'."
                        )
                    config_dict[key] = value

            # Recreate the intercept config from the updated dict
            config_type = type(middleware_config)
            config.middleware[self.middleware_name] = config_type(**config_dict)

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
            raise ValueError("No evaluators found in base workflow configuration. "
                             "Red teaming evaluation requires at least one evaluator of type 'red_teaming_evaluator'.")

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
    Configuration for red teaming evaluation runs with middleware functionality.

    This config allows running multiple evaluation scenarios where each scenario
    tests different middleware configurations. Scenarios are defined in a
    JSON file similar to an evaluation dataset.

    Attributes:
        base_evaluation_config: Base evaluation configuration that will be modified
            for each red teaming scenario.
        red_team_scenarios_file: Path to JSON file containing red team scenario entries.
            Each entry defines how to configure middleware for one test scenario.
    """
    base_evaluation_config: EvaluationRunConfig
    red_team_scenarios_file: Path = Field(
        description="Path to JSON file containing red team scenario entries"
    )

    @model_validator(mode='after')
    def validate_file_exists(self) -> 'RedTeamingEvaluationConfig':
        """Validate that the scenarios file exists."""
        if not self.red_team_scenarios_file.exists():
            raise FileNotFoundError(
                f"Red team scenarios file not found: {self.red_team_scenarios_file}"
            )
        if not self.red_team_scenarios_file.is_file():
            raise ValueError(
                f"Red team scenarios path is not a file: {self.red_team_scenarios_file}"
            )
        return self

