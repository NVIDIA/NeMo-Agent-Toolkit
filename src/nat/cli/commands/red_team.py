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

import asyncio
import copy
import logging
import shutil
from pathlib import Path

import click
import yaml

from nat.cli.cli_utils.red_teaming_utils import load_red_team_scenarios
from nat.cli.cli_utils.red_teaming_utils import validate_base_config
from nat.data_models.config import Config
from nat.eval.config import EvaluationRunConfig
from nat.eval.red_teaming_evaluator import RedTeamingScenarioBase
from nat.eval.runners.config import MultiEvaluationRunConfig
from nat.eval.runners.multi_eval_runner import MultiEvaluationRunner

logger = logging.getLogger(__name__)


@click.group(
    name=__name__,
    invoke_without_command=True,
    help="Run red teaming evaluation with multiple scenarios."
)
@click.option(
    "--config_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="A JSON/YAML file that sets the parameters for the workflow and evaluation.",
)
@click.option(
    "--scenarios_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="A JSON file containing red team scenario entries.",
)
@click.option(
    "--dataset",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=False,
    help="A json file with questions and ground truth answers. This will override the dataset path in the config file.",
)
@click.option(
    "--result_json_path",
    type=str,
    default="$",
    help=("A JSON path to extract the result from the workflow. Use this when the workflow returns "
          "multiple objects or a dictionary. For example, '$.output' will extract the 'output' field "
          "from the result."),
)
@click.option(
    "--endpoint",
    type=str,
    default=None,
    help="Use endpoint for running the workflow. Example: http://localhost:8000/generate",
)
@click.option(
    "--endpoint_timeout",
    type=int,
    default=300,
    help="HTTP response timeout in seconds. Only relevant if endpoint is specified.",
)
@click.option(
    "--reps",
    type=int,
    default=1,
    help="Number of repetitions for the evaluation.",
)
@click.option(
    "--override",
    type=(str, str),
    multiple=True,
    help="Override config values using dot notation (e.g., --override llms.nim_llm.temperature 0.7)",
)
@click.pass_context
def red_team_command(ctx, **kwargs) -> None:
    """Run red teaming evaluation with multiple scenarios."""
    pass


async def run_red_team_evaluation(config: MultiEvaluationRunConfig):
    """Run red team evaluation using MultiEvaluationRunner."""
    multi_eval_runner = MultiEvaluationRunner(config=config)
    return await multi_eval_runner.run_all()


@red_team_command.result_callback(replace=True)
def process_red_team_eval(
    processors,
    *,
    config_file: Path,
    scenarios_file: Path,
    dataset: Path,
    result_json_path: str,
    endpoint: str,
    endpoint_timeout: int,
    reps: int,
    override: tuple[tuple[str, str], ...],
):
    """
    Process the red team eval command and execute the evaluation.
    """
    from nat.cli.cli_utils.config_override import load_and_override_config
    from nat.runtime.loader import load_config
    from nat.utils.data_models.schema_validator import validate_schema

    # Step 1: Apply overrides to base config if any overrides are provided
    if override:
        logger.info(f"Applying {len(override)} override(s) to base configuration")
        config_dict = load_and_override_config(config_file, override)
        base_config = validate_schema(config_dict, Config)
    else:
        logger.info("No overrides provided, loading base configuration directly")
        base_config = load_config(config_file)

    # Validate the base config
    validate_base_config(base_config)

    # Extract base output directory
    base_output_dir = base_config.eval.general.output_dir
    logger.info(f"Base output directory: {base_output_dir}")

    # Step 2: Check if base directory exists and handle edge cases
    if base_output_dir.exists():
        raise click.ClickException(
            f"Output directory already exists: {base_output_dir}. "
            "Please remove it or specify a different output directory using --override."
        )

    if base_output_dir is None:
        logger.warning("No output directory configured. Results will not be saved to disk.")

    # Load scenarios
    scenarios = load_red_team_scenarios(scenarios_file)

    # Step 3: Create base output directory and save metadata
    if base_output_dir:
        base_output_dir.mkdir(parents=True, exist_ok=False)
        logger.info(f"Created base output directory: {base_output_dir}")

        # Save base config (with overrides applied) to output directory
        base_config_output_path = base_output_dir / "base_config.yml"
        with open(base_config_output_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(base_config.model_dump(mode='json'), f, default_flow_style=False)
        logger.info(f"Saved base configuration to: {base_config_output_path}")

        # Copy scenarios file to output directory
        scenarios_output_path = base_output_dir / "scenarios.json"
        shutil.copy(scenarios_file, scenarios_output_path)
        logger.info(f"Saved scenarios to: {scenarios_output_path}")

    def create_multi_eval_configs(scenario: RedTeamingScenarioBase) -> EvaluationRunConfig:
        """
        Create EvaluationRunConfig for a single scenario.

        Args:
            scenario: Red team scenario to apply

        Returns:
            EvaluationRunConfig instance for this scenario
        """
        logger.info(f"Creating evaluation config for scenario: {scenario.scenario_id}")

        # Create a deep copy of the base config (with overrides already applied)
        config_copy = copy.deepcopy(base_config)

        # Apply the scenario to the config
        modified_config = scenario.apply_to_config(config_copy)

        # Modify output directories to append scenario_id
        if base_output_dir:
            scenario_output_dir = base_output_dir / scenario.scenario_id
            modified_config.eval.general.output_dir = scenario_output_dir
            if modified_config.eval.general.output:
                modified_config.eval.general.output.dir = scenario_output_dir
            logger.info(f"Scenario '{scenario.scenario_id}' output directory: {scenario_output_dir}")

            # Save scenario-specific config to scenario output directory
            scenario_output_dir.mkdir(parents=True, exist_ok=False)
            scenario_config_path = scenario_output_dir / "scenario_config.yml"
            with open(scenario_config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(modified_config.model_dump(mode='json'), f, default_flow_style=False)
            logger.info(f"Saved scenario configuration to: {scenario_config_path}")

        # Validate the modified config
        try:
            validate_schema(modified_config.model_dump(mode='json'), Config)
        except Exception as e:
            raise click.ClickException(
                f"Modified configuration for scenario '{scenario.scenario_id}' failed validation: {e}"
            )

        # Create a new EvaluationRunConfig for this scenario
        # Note: We pass empty tuple for override since overrides are already applied
        scenario_eval_config = EvaluationRunConfig(
            config_file=modified_config,
            result_json_path=result_json_path,
            dataset=str(dataset) if dataset else None,
            endpoint=endpoint,
            endpoint_timeout=endpoint_timeout,
            reps=reps,
            override=(),  # Overrides already applied
        )
        return scenario_eval_config

    # Create multiple evaluation configs from scenarios
    eval_configs = {scenario.scenario_id: create_multi_eval_configs(scenario) for scenario in scenarios}

    # Create multi evaluation config
    multi_eval_config = MultiEvaluationRunConfig(configs=eval_configs)

    # Run the multi evaluation
    logger.info(f"Running red team evaluation with {len(eval_configs)} scenarios")
    asyncio.run(run_red_team_evaluation(multi_eval_config))
    logger.info("Red team evaluation completed")

