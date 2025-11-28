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

from __future__ import annotations

import logging
import typing

from pydantic import BaseModel
from pydantic import Field

from nat.data_models.config import Config
from nat.data_models.evaluate import EvalConfig
from nat.data_models.intermediate_step import IntermediateStep
from nat.eval.evaluator.evaluator_model import EvalOutputItem
from nat.eval.red_teaming_evaluator.filter_conditions import IntermediateStepsFilterCondition
from nat.middleware.red_teaming_middleware import RedTeamingMiddlewareConfig

logger = logging.getLogger(__name__)


def _update_config_dict_value(config_dict: dict[str, typing.Any], path: str, value: typing.Any) -> None:
    """Update a value in a nested config dictionary at the specified path.

    Similar to LayeredConfig._update_config_value, navigates nested dicts using dot notation.

    Args:
        config_dict: The configuration dictionary to update
        path: String representing the path using dot notation (e.g., "functions.my_func.middleware")
        value: The new value to set at the specified path

    Example:
        If config_dict is {"functions": {"my_func": {"middleware": ["a", "b"]}}}
        and path is "functions.my_func.middleware" with value ["a", "b", "c"],
        this will update config_dict to {"functions": {"my_func": {"middleware": ["a", "b", "c"]}}}
    """
    parts = path.split('.')
    current = config_dict
    # Navigate through nested dictionaries until reaching the parent of target
    for part in parts[:-1]:
        current = current[part]
    # Update the value at the target location
    current[parts[-1]] = value


class ConditionEvaluationResult(EvalOutputItem):
    """
    Evaluation results for a single IntermediateStep that meets the filtering condition.
    Attributes:
        id: Identifier from the input item
        score: Average score across all filter conditions
        reasoning: Reasoning for given score.
        intermediate_step: IntermediateStep that was selected and evaluated (based on reduction strategy).
            The evaluated output can be accessed through intermediate_step.payload.output.
    """
    intermediate_step: IntermediateStep | None = Field(
        default=None,
        description="The single IntermediateStep that was selected and evaluated (based on reduction strategy)"
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if any step of the evaluation has failed"
    )

    @classmethod
    def empty(cls, id: str, error: str | None = None) -> ConditionEvaluationResult:
        """
        Create an empty ConditionEvaluationResult.

        Returns:
            Empty ConditionEvaluationResult instance
        """
        return cls(
            id=id,
            score=0.0,
            reasoning = {},
            error_message=error,
            intermediate_step=None
        )


class RedTeamingEvalOutputItem(EvalOutputItem):
    """
    Extended evaluation output item for red teaming evaluations.

    Organizes results by filter condition name, with each condition containing
    its score, the evaluated output, and the single intermediate step that was selected.

    Attributes:
        id: Identifier from the input item
        score: Average score across all filter conditions
        reasoning: Summary information for compatibility
        results_by_condition: Map from condition name to evaluation results
    """
    results_by_condition: dict[str, ConditionEvaluationResult] = Field(
        description="Results organized by filter condition name"
    )


class RedTeamingScenarioBase(BaseModel):
    """
    A single red teaming scenario entry.

    Each entry defines how to configure one scenario for red teaming evaluation,
    including middleware configuration and evaluation parameters.

    Attributes:
        scenario_id: Unique identifier for this scenario
        middleware_name: Name of the middleware to apply, or None for baseline (no middleware)
        middleware_config_override: Parts of the middleware config to override.
            Keys are the middleware config keys to override, and should be compatible
            with the middleware defined by middleware_name.
            If None, the middleware config is not modified.
        evaluation_instructions: Optional scenario-specific instructions for the evaluator to check
            if the middleware was successful in producing the expected behavior
        filter_conditions: Optional filter conditions for selecting specific intermediate steps to evaluate
    """
    scenario_id: str = Field(description="Unique identifier for this scenario")
    middleware_name: str | None = Field(
        default=None,
        description="Name of the middleware to apply, or null for baseline scenario"
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
            Each transformation dumps config to dict, mutates, and reconstructs for validation.
        """
        logger.info(f"Applying scenario '{self.scenario_id}' to configuration")

        # Apply intercept override if this isn't a baseline scenario
        if self.middleware_name is not None:
            config = self._apply_intercept_override(config)

        # Inject evaluation instructions and/or filter conditions if provided
        if self.evaluation_instructions is not None or self.filter_conditions is not None:
            config = self._inject_evaluation_config(config)

        return config

    def _validate_middleware_overwrite(self, config: Config) -> dict[str, typing.Any]:
        """Validate middleware overwrite configuration.

        Validates that the middleware exists, is accessible, and is a red teaming middleware.
        Returns the middleware config dict for further use.

        Args:
            config: Configuration to validate

        Returns:
            Middleware config dict if validation passes

        Raises:
            ValueError: If middleware validation fails
        """

        if not config.middleware:
            raise ValueError(
                "No middleware found in base workflow configuration. "
                "Red teaming evaluation requires at least one middleware."
            )

        # Validate that the intercept exists in middleware
        if self.middleware_name not in config.middleware:
            raise ValueError(
                f"Scenario '{self.scenario_id}': Middleware '{self.middleware_name}' "
                f"not found in middleware. Available middleware: "
                f"{list(config.middleware.keys())}"
            )

        red_teaming_middleware_config = config.middleware[self.middleware_name]
        if not isinstance(red_teaming_middleware_config, RedTeamingMiddlewareConfig):
            raise ValueError(
                f"Scenario '{self.scenario_id}': Middleware '{self.middleware_name}' "
                f"is not a red teaming middleware. Available middleware: "
                f"{list(config.middleware.keys())}"
            )

        # Return the middleware config as a dict
        return red_teaming_middleware_config.model_dump(exclude_unset=False)

    def _find_target_path(
        self, config_dict: dict[str, typing.Any], target_name: str
    ) -> str:
        """Find the path to a target component in the config dict.

        Searches for target_name in functions, function_groups, and workflow.
        Returns the dot-notation path to the target.

        Args:
            config_dict: Configuration dictionary to search
            target_name: Name of the target component to find

        Returns:
            Dot-notation path to the target (e.g., "functions.my_func", "function_groups.my_group", "workflow")

        Raises:
            ValueError: If target is ambiguous (exists in multiple locations) or not found
        """
        found_paths: list[str] = []

        # Check functions
        if "functions" in config_dict and target_name in config_dict["functions"]:
            found_paths.append(f"functions.{target_name}")

        # Check function_groups
        if "function_groups" in config_dict and target_name in config_dict["function_groups"]:
            found_paths.append(f"function_groups.{target_name}")

        # Check workflow (special case - workflow itself is the target)
        if target_name == "workflow" and "workflow" in config_dict:
            found_paths.append("workflow")

        # Validate results
        if len(found_paths) > 1:
            raise ValueError(
                f"Scenario '{self.scenario_id}': Target '{target_name}' exists in multiple locations: "
                f"{', '.join(found_paths)}. This creates ambiguity. "
                "Please ensure component names are unique."
            )

        if len(found_paths) == 0:
            # Collect available targets for error message
            available_targets: list[str] = []
            if "functions" in config_dict:
                available_targets.extend([f"functions.{name}" for name in config_dict["functions"].keys()])
            if "function_groups" in config_dict:
                available_targets.extend([f"function_groups.{name}" for name in config_dict["function_groups"].keys()])
            if "workflow" in config_dict:
                available_targets.append("workflow")

            raise ValueError(
                f"Scenario '{self.scenario_id}': Target '{target_name}' not found in config. "
                f"Available targets: {', '.join(available_targets)}"
            )

        return found_paths[0]

    def _apply_intercept_override(self, config: Config) -> Config:
        """
        Apply intercept override to the configuration.

        This method:
        1. Validates that the intercept exists
        2. Removes the intercept from all middleware-capable components (functions, function_groups, workflow)
        3. Applies the intercept only to the specified target
        4. Updates the intercept's configuration using middleware_config_override if provided

        Args:
            config: Configuration to transform

        Returns:
            Transformed configuration with middleware overrides applied

        Raises:
            ValueError: If intercept or target doesn't exist in config
        """
        # Validate middleware overwrite (handles baseline scenario check)
        # Handle baseline scenario (null intercept)
        if not self.middleware_name:
            logger.info(
                f"Baseline scenario '{self.scenario_id}': using config as-is (no intercept modifications)"
            )
            return config
        middleware_config_dict = self._validate_middleware_overwrite(config)

        # Early return for baseline scenario (no changes needed)
        if not middleware_config_dict:
            return config

        # Dump config to dict for manipulation
        # Use mode='python' to preserve Python objects that can be reconstructed
        config_dict = config.model_dump(mode='python', exclude_unset=False)

        # Step 1: Clear the middleware from all middleware-capable components
        logger.info(f"Clearing middleware '{self.middleware_name}' from all components")
        for component_type in ["functions", "function_groups"]:
            if component_type in config_dict:
                for component_name, component_config in config_dict[component_type].items():
                    if "middleware" in component_config and self.middleware_name in component_config["middleware"]:
                        # Remove middleware from list
                        component_config["middleware"] = [
                            mc for mc in component_config["middleware"] if mc != self.middleware_name
                        ]
                        logger.debug(f"Removed '{self.middleware_name}' from {component_type}['{component_name}']")

        # Clear from workflow if present
        if "workflow" in config_dict and "middleware" in config_dict["workflow"]:
            if self.middleware_name in config_dict["workflow"]["middleware"]:
                config_dict["workflow"]["middleware"] = [
                    mc for mc in config_dict["workflow"]["middleware"] if mc != self.middleware_name
                ]
                logger.debug(f"Removed '{self.middleware_name}' from workflow")

        # Step 2: Determine target name
        if self.middleware_config_override and 'target_function_or_group' in self.middleware_config_override:
            target_name = self.middleware_config_override.get('target_function_or_group', '').split('.')[0]
        elif middleware_config_dict.get('target_function_or_group'):
            target_name = middleware_config_dict['target_function_or_group'].split('.')[0]
        else:
            raise ValueError(
                f"Scenario '{self.scenario_id}': No target function or group specified "
                "in middleware_config_override or middleware config"
            )

        # Step 3: Find target path and apply middleware
        target_path = self._find_target_path(config_dict, target_name)
        logger.info(f"Adding middleware '{self.middleware_name}' to {target_path}")

        # Get current middleware list for the target
        target_middleware_path = f"{target_path}.middleware"
        parts = target_middleware_path.split('.')
        current = config_dict
        for part in parts[:-1]:
            current = current[part]
        current_middleware = current.get(parts[-1], [])
        current_middleware = list(current_middleware) + [self.middleware_name]
        _update_config_dict_value(config_dict, target_middleware_path, current_middleware)

        # Step 4: Apply middleware_config_override if provided
        if self.middleware_config_override:
            logger.info(
                f"Updating config for middleware '{self.middleware_name}': {self.middleware_config_override}"
            )

            # Validate that all override keys exist in the middleware config
            for key in self.middleware_config_override.keys():
                if key not in middleware_config_dict:
                    raise KeyError(
                        f"middleware_config_override contains key '{key}' not present in the middleware config "
                        f"for '{self.middleware_name}'. Available keys: {list(middleware_config_dict.keys())}"
                    )

            # Apply overrides using path-based updates
            for key, value in self.middleware_config_override.items():
                middleware_path = f"middleware.{self.middleware_name}.{key}"
                _update_config_dict_value(config_dict, middleware_path, value)

        # Reconstruct config from dict
        return Config(**config_dict)

    def _inject_evaluation_config(self, config: Config) -> Config:
        """
        Inject scenario-specific evaluation configuration into red_teaming_evaluator configs.

        This includes evaluation instructions and filter conditions.

        Args:
            config: Configuration to transform

        Returns:
            Transformed configuration with evaluation config injected
        """
        logger.info("Injecting scenario-specific evaluation configuration")

        # Dump config to dict for manipulation
        # Use mode='python' to preserve Python objects that can be reconstructed
        config_dict = config.model_dump(mode='python', exclude_unset=False)

        # Check if eval config exists
        eval_config = config_dict.get("eval")
        if not eval_config or not eval_config.get("evaluators"):
            raise ValueError(
                "No evaluators found in base workflow configuration. "
                "Red teaming evaluation requires at least one evaluator of type 'red_teaming_evaluator'."
            )

        # Find and update red_teaming_evaluator configs
        evaluators_updated = False
        for evaluator_name, evaluator_config in eval_config["evaluators"].items():
            # Check if this is a red_teaming_evaluator
            if evaluator_config.get("type") == "red_teaming_evaluator":
                logger.info(f"Updating evaluator '{evaluator_name}' with scenario-specific configuration")

                # Update scenario_specific_instructions if provided
                if self.evaluation_instructions is not None:
                    instructions_path = f"eval.evaluators.{evaluator_name}.scenario_specific_instructions"
                    _update_config_dict_value(config_dict, instructions_path, self.evaluation_instructions)
                    logger.debug(f"Injected evaluation instructions into '{evaluator_name}'")

                # Update filter_conditions if provided
                if self.filter_conditions is not None:
                    # Convert filter conditions to dicts for serialization
                    filter_conditions_dict = [fc.model_dump() for fc in self.filter_conditions]
                    filter_path = f"eval.evaluators.{evaluator_name}.filter_conditions"
                    _update_config_dict_value(config_dict, filter_path, filter_conditions_dict)
                    logger.debug(f"Injected {len(self.filter_conditions)} filter conditions into '{evaluator_name}'")

                evaluators_updated = True
                logger.debug(f"Successfully injected configuration into evaluator '{evaluator_name}'")

        if not evaluators_updated:
            raise ValueError(
                "No red_teaming_evaluator found in evaluators. "
                f"Available evaluator types: {[e.get('type') for e in eval_config['evaluators'].values()]}"
            )

        # Reconstruct config from dict
        return Config(**config_dict)


