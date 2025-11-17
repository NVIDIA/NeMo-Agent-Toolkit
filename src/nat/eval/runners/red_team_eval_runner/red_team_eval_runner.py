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

import copy
import json
import logging
from pathlib import Path

from nat.data_models.config import Config
from nat.eval.config import EvaluationRunOutput
from nat.eval.evaluate import EvaluationRun
from nat.eval.runners.red_team_eval_runner.red_team_eval_config import RedTeamingEvaluationConfig
from nat.eval.runners.red_team_eval_runner.red_team_eval_config import RedTeamScenarioEntry

logger = logging.getLogger(__name__)


class RedTeamingEvaluationRunner:
    """
    Runner for red teaming agents using middleware functionality.

    This runner takes a base workflow configuration and applies different
    config modifications based on a JSON dataset. These modifications include:

    1. Middleware modifications:
        - Applying middleware to specific functions or function groups
        - Modifying middleware input configurations.
    2. Evaluation modifications:
        - Adding scenario-specific evaluation criteria assuming a red team evaluator.
        - Adding scenario-specific filter conditions for evaluation.

    This allows a user to simulate different adversarial scenarios.
    and evaluate the agent's response under these conditions.
    """

    def __init__(self, config: RedTeamingEvaluationConfig):
        """
        Initialize a red teaming evaluation runner.

        Args:
            config: Red teaming evaluation configuration

        Raises:
            ValueError: If base_evaluation_config doesn't contain required middleware
                or red_teaming_evaluator
        """
        self.config = config
        self.evaluation_run_outputs: dict[str, EvaluationRunOutput] = {}
        self._scenarios = self._load_red_team_scenarios()
        workflow_config = self._load_base_config()
        # Validate the base config
        self._validate_base_config(workflow_config)
        # Update the base evaluation config with the loaded workflow config.
        self.config.base_evaluation_config.config_file = workflow_config

    async def run_all(self) -> dict[str, EvaluationRunOutput]:
        """
        Run all red teaming evaluation scenarios.

        Returns:
            Dictionary mapping scenario IDs to their evaluation outputs
        """

        logger.info(f"Running {len(self._scenarios)} red teaming scenarios")

        # Run evaluation for each scenario
        for scenario in self._scenarios:
            logger.info(f"Running red teaming scenario: {scenario.scenario_id}")
            output = await self.run_single_scenario(scenario)
            self.evaluation_run_outputs[scenario.scenario_id] = output

        return self.evaluation_run_outputs

    def _load_red_team_scenarios(self) -> list[RedTeamScenarioEntry]:
        """
        Load red team scenario entries from JSON file.

        Returns:
            List of red team scenario entries

        Raises:
            ValueError: If JSON file is invalid or contains validation errors
        """
        scenarios_file = self.config.red_team_scenarios_file

        logger.info(f"Loading red team scenarios from: {scenarios_file}")

        with open(scenarios_file, encoding='utf-8') as f:
            scenarios_data = json.load(f)

        if not isinstance(scenarios_data, list):
            raise ValueError(
                f"Red team scenarios file must contain a JSON array, got {type(scenarios_data)}"
            )

        # Parse into RedTeamScenarioEntry objects
        scenarios = []
        for idx, entry_data in enumerate(scenarios_data):
            try:
                scenario = RedTeamScenarioEntry(**entry_data)
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

    def _load_base_config(self) -> Config:
        from nat.runtime.loader import load_config
        # Create a deep copy of the base evaluation config
        eval_run_config = copy.deepcopy(self.config.base_evaluation_config)
        base_config_or_path = eval_run_config.config_file
        # Load config if it's a path
        if isinstance(base_config_or_path, Path):
            workflow_config = load_config(base_config_or_path)

        else:
            # Assume it's already a Config BaseModel
            workflow_config = base_config_or_path
            if not isinstance(workflow_config, Config):
                # If it's a different BaseModel type, try to treat it as Config
                workflow_config = Config(**workflow_config.model_dump())
        return workflow_config

    def _validate_base_config(self, config: Config) -> None:
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
                "base_evaluation_config must contain at least one middleware. "
                "Red teaming evaluation requires middleware to be configured."
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
                "base_evaluation_config must contain at least one evaluator of type "
                "'red_teaming_evaluator'. "
                f"Found evaluator types: {available_evaluator_types if available_evaluator_types else 'none'}."
            )

    async def run_single_scenario(
        self,
        scenario: RedTeamScenarioEntry
    ) -> EvaluationRunOutput:
        """
        Run a single red teaming scenario with the specified middleware configuration.

        Args:
            scenario: Red teaming scenario entry defining the middleware configuration.

        Returns:
            Evaluation output for this scenario.
        """
        eval_run_config = copy.deepcopy(self.config.base_evaluation_config)
        if isinstance(eval_run_config.config_file, Config):
            base_config = eval_run_config.config_file
        else:
            raise ValueError(f"Base config must already be a Config object, got {type(eval_run_config.config_file)}")

        # Apply the scenario to the workflow config (handles both middleware and evaluation instructions)
        modified_config = scenario.apply_to_config(base_config)

        # Update the evaluation config with the modified workflow config
        eval_run_config.config_file = modified_config
        # Run the evaluation
        logger.info(f"Running evaluation for scenario: {scenario.scenario_id}")
        evaluation_run = EvaluationRun(config=eval_run_config)
        return await evaluation_run.run_and_evaluate()

