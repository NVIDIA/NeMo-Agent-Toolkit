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
from nat.eval.runners.redteam_config import InterceptScenarioEntry
from nat.eval.runners.redteam_config import RedTeamingEvaluationConfig

logger = logging.getLogger(__name__)


class RedTeamingEvaluationRunner:
    """
    Runner for red teaming evaluations with function intercepts.

    This runner takes a base workflow configuration and applies different
    function intercept modifications based on a JSON dataset. Each entry
    in the dataset defines one test scenario.
    """

    def __init__(self, config: RedTeamingEvaluationConfig):
        """
        Initialize a red teaming evaluation runner.

        Args:
            config: Red teaming evaluation configuration
        """
        self.config = config
        self.evaluation_run_outputs: dict[str, EvaluationRunOutput] = {}

    async def run_all(self) -> dict[str, EvaluationRunOutput]:
        """
        Run all red teaming evaluation scenarios.

        Returns:
            Dictionary mapping scenario IDs to their evaluation outputs
        """
        # Load intercept scenarios from JSON file
        scenarios = self._load_intercept_scenarios()

        logger.info(f"Running {len(scenarios)} red teaming scenarios")

        # Run evaluation for each scenario
        for scenario in scenarios:
            logger.info(f"Running red teaming scenario: {scenario.scenario_id}")
            output = await self.run_single_scenario(scenario)
            self.evaluation_run_outputs[scenario.scenario_id] = output

        return self.evaluation_run_outputs

    def _load_intercept_scenarios(self) -> list[InterceptScenarioEntry]:
        """
        Load intercept scenario entries from JSON file.

        Returns:
            List of intercept scenario entries

        Raises:
            ValueError: If JSON file is invalid or contains validation errors
        """
        scenarios_file = self.config.intercept_scenarios_file

        logger.info(f"Loading intercept scenarios from: {scenarios_file}")

        with open(scenarios_file, encoding='utf-8') as f:
            scenarios_data = json.load(f)

        if not isinstance(scenarios_data, list):
            raise ValueError(
                f"Intercept scenarios file must contain a JSON array, got {type(scenarios_data)}"
            )

        # Parse into InterceptenarioEntry objects
        scenarios = []
        for idx, entry_data in enumerate(scenarios_data):
            try:
                scenario = InterceptScenarioEntry(**entry_data)
                scenarios.append(scenario)
            except Exception as e:
                raise ValueError(
                    f"Invalid scenario entry at index {idx}: {e}"
                ) from e

        # Validate: warn if multiple null intercepts
        null_intercept_scenarios = [s for s in scenarios if s.intercept_name is None]
        if len(null_intercept_scenarios) > 1:
            logger.warning(
                f"Found {len(null_intercept_scenarios)} scenarios with null intercept_name "
                f"(baseline scenarios): {[s.scenario_id for s in null_intercept_scenarios]}. "
                "It's recommended to have only one baseline scenario."
            )

        logger.info(f"Loaded {len(scenarios)} scenarios successfully")
        return scenarios

    async def run_single_scenario(
        self,
        scenario: InterceptScenarioEntry
    ) -> EvaluationRunOutput:
        """
        Run a single red teaming scenario with the specified intercept configuration.

        Args:
            scenario: Intercept scenario entry defining the configuration

        Returns:
            Evaluation output for this scenario
        """
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

        # Apply the scenario to the workflow config (handles both intercepts and instructions)
        modified_config = scenario.apply_to_config(workflow_config)

        # Update the evaluation config with the modified workflow config
        eval_run_config.config_file = modified_config
        # Run the evaluation
        logger.info(f"Running evaluation for scenario: {scenario.scenario_id}")
        evaluation_run = EvaluationRun(config=eval_run_config)
        return await evaluation_run.run_and_evaluate()
