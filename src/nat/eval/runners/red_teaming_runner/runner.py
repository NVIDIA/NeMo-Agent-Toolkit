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

"""Red teaming runner for executing multi-scenario red teaming evaluations."""

from __future__ import annotations

import logging
import typing
import uuid
import warnings
from datetime import datetime
from pathlib import Path

import yaml

from nat.data_models.config import Config
from nat.data_models.evaluate import EvalGeneralConfig
from nat.eval.config import EvaluationRunConfig
from nat.eval.config import EvaluationRunOutput
from nat.eval.red_teaming_evaluator.register import RedTeamingEvaluatorConfig
from nat.eval.runners.config import MultiEvaluationRunConfig
from nat.eval.runners.multi_eval_runner import MultiEvaluationRunner
from nat.eval.runners.red_teaming_runner.config import RED_TEAMING_EVALUATOR_LLM_NAME
from nat.eval.runners.red_teaming_runner.config import RedTeamingRunnerConfig
from nat.eval.runners.red_teaming_runner.config import RedTeamingScenario
from nat.middleware.red_teaming_middleware_config import RedTeamingMiddlewareConfig
from nat.utils.data_models.schema_validator import validate_schema
from nat.eval.evaluator.evaluator_model import EvalOutput

logger = logging.getLogger(__name__)


class RedTeamingRunner:
    """Runner for executing red teaming evaluations across multiple scenarios.

    This runner encapsulates all the logic for:
    - Generating workflow configurations for each scenario
    - Setting up output directories
    - Saving configuration files
    - Running evaluations via MultiEvaluationRunner

    Example usage:
        runner = RedTeamingRunner(
            config=rt_config,
            base_config=base_config,
            dataset_path="/path/to/dataset.json",
        )
        results = await runner.run()
    """

    def __init__(
        self,
        config: RedTeamingRunnerConfig | None,
        base_config: Config,
        dataset_path: str | None = None,
        result_json_path: str = "$",
        endpoint: str | None = None,
        endpoint_timeout: int = 300,
        reps: int = 1,
        overrides: tuple[tuple[str, str], ...] = (),
    ):
        """Initialize the RedTeamingRunner.

        Args:
            config: Red teaming configuration with scenarios. If None, the base_config
                is used as a single pre-configured scenario.
            base_config: The base workflow configuration to transform for each scenario.
            dataset_path: Optional dataset path (overrides config dataset).
            result_json_path: JSON path to extract the result from the workflow.
            endpoint: Optional endpoint URL for running the workflow.
            endpoint_timeout: HTTP response timeout in seconds.
            reps: Number of repetitions for the evaluation.
            overrides: Config overrides using dot notation (path, value) tuples.
        """
        self.config = config
        self.base_config = base_config
        self.dataset_path = dataset_path
        self.result_json_path = result_json_path
        self.endpoint = endpoint
        self.endpoint_timeout = endpoint_timeout
        self.reps = reps
        self.overrides = overrides

        self._scenario_configs: dict[str, Config] | None = None
        self._base_output_dir: Path | None = None

    async def run(self) -> dict[str, EvaluationRunOutput]:
        """Run the red teaming evaluation across all scenarios.

        Returns:
            Dictionary mapping scenario_id to EvaluationRunOutput.

        Raises:
            ValueError: If configuration validation fails.
        """
        # Generate scenario configs
        scenario_configs = self.generate_workflow_configs()

        # Apply overrides to all scenario configs
        scenario_configs = self._apply_overrides_to_all(scenario_configs)

        # Setup output directory
        base_output_dir = self.setup_output_directory(scenario_configs)

        # Save configs
        self.save_configs(base_output_dir, scenario_configs)

        # Build evaluation configs
        eval_configs = self._build_evaluation_configs(base_output_dir, scenario_configs)

        # Run evaluation
        multi_eval_config = MultiEvaluationRunConfig(configs=eval_configs)
        logger.info("Running red team evaluation with %d scenario(s)", len(eval_configs))

        runner = MultiEvaluationRunner(config=multi_eval_config)
        results = await runner.run_all()
        logger.info("Red team evaluation completed")
        summary = self._compute_result_summary(results)
        self._log_results_summary(summary, base_output_dir)
        return results

    def generate_workflow_configs(self) -> dict[str, Config]:
        """Generate workflow configurations for each scenario.

        If config is None, returns the base_config as a single scenario
        after validating it has the required red teaming components.

        Returns:
            Dictionary mapping scenario_id to the transformed Config.

        Raises:
            ValueError: If validation fails.
        """
        if self.config is None:
            # No red_team_config - use base_config directly as single scenario
            self._validate_base_config_for_direct_use(self.base_config)
            return {"single_scenario": self.base_config}

        # Warn about other evaluators in base config
        self._warn_about_other_evaluators(self.base_config)

        # Validate: dataset must be defined somewhere
        self._validate_dataset_exists(self.base_config, self.dataset_path)

        scenario_configs: dict[str, Config] = {}

        for scenario_key, scenario in self.config.scenarios.items():
            scenario_id = scenario.scenario_id or scenario_key
            logger.info("Generating config for scenario: %s", scenario_id)

            # Deep copy the base config
            config_dict = self.base_config.model_dump(mode='python', exclude_unset=False)

            # Add evaluator LLM with fixed name
            config_dict["llms"][RED_TEAMING_EVALUATOR_LLM_NAME] = (
                self.config.evaluator_llm.model_dump(mode='python')
            )
            logger.debug("Added evaluator LLM as '%s'", RED_TEAMING_EVALUATOR_LLM_NAME)

            # Apply middleware if not a baseline scenario
            if scenario.middleware is not None:
                middleware_name = f"red_teaming_{scenario_id}"
                middleware_config = scenario.middleware.model_dump(mode='python')

                # Add middleware to the middleware section
                if "middleware" not in config_dict:
                    config_dict["middleware"] = {}
                config_dict["middleware"][middleware_name] = middleware_config

                # Attach middleware to ALL functions, function_groups, and workflow
                self._attach_middleware_everywhere(config_dict, middleware_name)
                logger.debug("Attached middleware '%s' to all components", middleware_name)

            # Inject evaluator config
            self._inject_evaluator_config(config_dict, scenario)

            # Merge general eval settings if provided
            if self.config.general is not None:
                self._merge_general_config(config_dict, self.config.general)

            # Reconstruct config from dict
            scenario_configs[scenario_id] = Config(**config_dict)
            logger.info("Generated config for scenario '%s'", scenario_id)

        return scenario_configs

    def setup_output_directory(self, scenario_configs: dict[str, Config]) -> Path:
        """Set up the base output directory.

        If the directory already exists, creates a new directory with a timestamp
        and unique identifier suffix.

        Args:
            scenario_configs: The generated scenario configurations.

        Returns:
            The base output directory path.
        """
        # Determine base output directory from first scenario
        first_scenario = next(iter(scenario_configs.values()))
        base_output_dir = first_scenario.eval.general.output_dir

        if base_output_dir.exists():
            # Generate a unique directory name with timestamp and 4-digit UID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            short_uid = uuid.uuid4().hex[:4]
            new_dir_name = f"{base_output_dir.name}_{timestamp}_{short_uid}"
            base_output_dir = base_output_dir.parent / new_dir_name

            warnings.warn(
                f"Output directory already exists. Creating new directory: {base_output_dir}",
                UserWarning,
                stacklevel=2
            )

        base_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created output directory: %s", base_output_dir)

        self._base_output_dir = base_output_dir
        return base_output_dir

    def save_configs(
        self,
        base_output_dir: Path,
        scenario_configs: dict[str, Config],
    ) -> None:
        """Save base config, red team config, and scenario configs to disk.

        Args:
            base_output_dir: The base output directory.
            scenario_configs: The generated scenario configurations.
        """
        # Save base config
        with open(base_output_dir / "base_config.yml", 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.base_config.model_dump(mode='json'), f, default_flow_style=False)

        # Save red team config if present
        if self.config:
            with open(base_output_dir / "red_team_config.yml", 'w', encoding='utf-8') as f:
                yaml.safe_dump(self.config.model_dump(mode='json'), f, default_flow_style=False)

        # Save scenario configs
        for scenario_id, scenario_config in scenario_configs.items():
            scenario_output_dir = base_output_dir / scenario_id
            scenario_output_dir.mkdir(parents=True, exist_ok=True)
            with open(scenario_output_dir / "scenario_config.yml", 'w', encoding='utf-8') as f:
                yaml.safe_dump(scenario_config.model_dump(mode='json'), f, default_flow_style=False)

    def _apply_overrides_to_all(
        self,
        scenario_configs: dict[str, Config],
    ) -> dict[str, Config]:
        """Apply CLI overrides to all scenario configs.

        Args:
            scenario_configs: The scenario configurations to modify.

        Returns:
            The modified scenario configurations.
        """
        if not self.overrides:
            return scenario_configs

        result = {}
        for scenario_id, config in scenario_configs.items():
            config_dict = config.model_dump(mode='json')
            for path, value in self.overrides:
                self._update_config_value(config_dict, path, value)
            result[scenario_id] = Config(**config_dict)
        return result

    def _build_evaluation_configs(
        self,
        base_output_dir: Path,
        scenario_configs: dict[str, Config],
    ) -> dict[str, EvaluationRunConfig]:
        """Build EvaluationRunConfig for each scenario.

        Args:
            base_output_dir: The base output directory.
            scenario_configs: The generated scenario configurations.

        Returns:
            Dictionary mapping scenario_id to EvaluationRunConfig.

        Raises:
            ValueError: If config validation fails.
        """
        eval_configs: dict[str, EvaluationRunConfig] = {}

        for scenario_id, scenario_config in scenario_configs.items():
            # Set scenario-specific output directory
            scenario_output_dir = base_output_dir / scenario_id
            scenario_config.eval.general.output_dir = scenario_output_dir
            if scenario_config.eval.general.output:
                scenario_config.eval.general.output.dir = scenario_output_dir

            # Validate
            try:
                validate_schema(scenario_config.model_dump(mode='json'), Config)
            except Exception as e:
                raise ValueError(f"Config for scenario '{scenario_id}' failed validation: {e}") from e

            eval_configs[scenario_id] = EvaluationRunConfig(
                config_file=scenario_config,
                result_json_path=self.result_json_path,
                dataset=self.dataset_path,
                endpoint=self.endpoint,
                endpoint_timeout=self.endpoint_timeout,
                reps=self.reps,
                override=(),
            )

        return eval_configs

    def _validate_base_config_for_direct_use(self, config: Config) -> None:
        """Validate that a workflow config is compatible with red teaming.

        A workflow config is compatible if it contains:
        - At least one RedTeamingMiddleware (or subclass)
        - At least one red_teaming_evaluator

        This is used when the user provides a pre-configured workflow instead
        of a RedTeamingRunnerConfig.

        Args:
            config: The workflow configuration to validate.

        Raises:
            ValueError: If the config is not red-team compatible.
        """
        errors: list[str] = []

        # Check for red teaming middleware
        has_red_teaming_middleware = False
        if config.middleware:
            for middleware_name, middleware_config in config.middleware.items():
                if isinstance(middleware_config, RedTeamingMiddlewareConfig):
                    has_red_teaming_middleware = True
                    logger.debug("Found red teaming middleware: %s", middleware_name)
                    break

        if not has_red_teaming_middleware:
            middleware_types = []
            if config.middleware:
                middleware_types = [
                    type(m).__name__ for m in config.middleware.values()
                ]
            errors.append(
                f"Config must contain at least one middleware of type RedTeamingMiddleware "
                f"(or subclass). Found middleware types: {middleware_types or 'none'}"
            )

        # Check for red teaming evaluator
        has_red_teaming_evaluator = False
        if config.eval and config.eval.evaluators:
            for evaluator_name, evaluator_config in config.eval.evaluators.items():
                if isinstance(evaluator_config, RedTeamingEvaluatorConfig):
                    has_red_teaming_evaluator = True
                    logger.debug("Found red teaming evaluator: %s", evaluator_name)
                    break
                # Also check by type string for backwards compatibility
                if hasattr(evaluator_config, 'type') and evaluator_config.type == 'red_teaming_evaluator':
                    has_red_teaming_evaluator = True
                    logger.debug("Found red teaming evaluator (by type): %s", evaluator_name)
                    break

        if not has_red_teaming_evaluator:
            evaluator_types = []
            if config.eval and config.eval.evaluators:
                evaluator_types = [
                    getattr(e, 'type', type(e).__name__)
                    for e in config.eval.evaluators.values()
                ]
            errors.append(
                f"Config must contain at least one evaluator of type red_teaming_evaluator. "
                f"Found evaluator types: {evaluator_types or 'none'}"
            )

        if errors:
            raise ValueError(
                "Workflow config is not red-team compatible:\n- " +
                "\n- ".join(errors)
            )

        logger.info("Workflow config validated for red teaming")

    def _warn_about_other_evaluators(self, base_config: Config) -> None:
        """Warn if the base config contains other evaluators.

        Red teaming evaluation is potentially incompatible with other evaluators
        due to its adversarial nature.

        Args:
            base_config: The base workflow configuration to validate.
        """
        if base_config.eval and base_config.eval.evaluators:
            other_evaluators = list(base_config.eval.evaluators.keys())
            if other_evaluators:
                warnings.warn(
                    f"Base workflow config contains other evaluators: {other_evaluators}. "
                    "Red teaming evaluation is potentially incompatible with other evaluators. "
                    "Please remove them from the base workflow config.",
                    UserWarning,
                    stacklevel=3
                )

    def _validate_dataset_exists(
        self,
        base_config: Config,
        dataset_path: str | None,
    ) -> None:
        """Validate that a dataset is defined somewhere.

        Dataset can be defined in:
        - CLI --dataset argument (dataset_path)
        - RedTeamingRunnerConfig.general.dataset
        - base_config.eval.general.dataset

        Args:
            base_config: The base workflow configuration.
            dataset_path: Optional dataset path from CLI.

        Raises:
            ValueError: If no dataset is defined anywhere.
        """
        # Check CLI argument
        if dataset_path:
            return

        # Check RedTeamingRunnerConfig.general.dataset
        if self.config and self.config.general and self.config.general.dataset:
            return

        # Check base_config.eval.general.dataset
        if (base_config.eval and
            base_config.eval.general and
            base_config.eval.general.dataset):
            return

        raise ValueError(
            "No dataset defined. Please provide a dataset via:\n"
            "  - CLI: --dataset <path>\n"
            "  - RedTeamingRunnerConfig: general.dataset\n"
            "  - Base workflow: eval.general.dataset"
        )

    def _merge_general_config(
        self,
        config_dict: dict[str, typing.Any],
        general: EvalGeneralConfig,
    ) -> None:
        """Merge general eval settings into the config dict.

        This performs a union of the base workflow's eval.general with the
        RedTeamingRunnerConfig.general, where RedTeamingRunnerConfig values
        take precedence. Only explicitly set values override base values.

        Args:
            config_dict: The configuration dictionary to modify (in place).
            general: The EvalGeneralConfig from RedTeamingRunnerConfig.
        """
        # Ensure eval.general exists
        if "eval" not in config_dict:
            config_dict["eval"] = {}
        if "general" not in config_dict["eval"]:
            config_dict["eval"]["general"] = {}

        # Get the new general config as dict, excluding unset values
        # This ensures we only override values that were explicitly set
        general_dict = general.model_dump(mode='python', exclude_unset=False)

        # Log which fields are being overridden
        existing_general = config_dict["eval"]["general"]
        overridden_fields = [
            key for key in general_dict.keys()
            if key in existing_general and existing_general[key] != general_dict[key]
        ]
        if overridden_fields:
            logger.info(
                "Merging RedTeamingRunnerConfig.general into base config. "
                "Overriding fields: %s", overridden_fields
            )

        # Merge: base config values as defaults, RedTeamingRunnerConfig values override
        config_dict["eval"]["general"].update(general_dict)

    def _attach_middleware_everywhere(
        self,
        config_dict: dict[str, typing.Any],
        middleware_name: str,
    ) -> None:
        """Attach middleware to all functions, function_groups, and workflow.

        The middleware's internal target_function_or_group handles runtime
        activation - this just ensures the middleware is registered everywhere.

        Args:
            config_dict: The configuration dictionary to modify (in place).
            middleware_name: Name of the middleware to attach.
        """
        # Attach to all functions
        if "functions" in config_dict:
            for func_config in config_dict["functions"].values():
                if "middleware" not in func_config:
                    func_config["middleware"] = []
                if middleware_name not in func_config["middleware"]:
                    func_config["middleware"].append(middleware_name)

        # Attach to all function_groups
        if "function_groups" in config_dict:
            for group_config in config_dict["function_groups"].values():
                if "middleware" not in group_config:
                    group_config["middleware"] = []
                if middleware_name not in group_config["middleware"]:
                    group_config["middleware"].append(middleware_name)

        # Attach to workflow
        if "workflow" in config_dict:
            if "middleware" not in config_dict["workflow"]:
                config_dict["workflow"]["middleware"] = []
            if middleware_name not in config_dict["workflow"]["middleware"]:
                config_dict["workflow"]["middleware"].append(middleware_name)

    def _inject_evaluator_config(
        self,
        config_dict: dict[str, typing.Any],
        scenario: RedTeamingScenario,
    ) -> None:
        """Inject the evaluator configuration into the workflow config.

        Creates a red_teaming_evaluator in the eval section with:
        - Base evaluator config from RedTeamingRunnerConfig
        - Fixed LLM name 'red_teaming_evaluator_llm'
        - Scenario-specific overrides for evaluation_instructions and filter_conditions

        Args:
            config_dict: The configuration dictionary to modify (in place).
            scenario: The scenario containing potential overrides.
        """
        if self.config is None:
            return

        # Ensure eval section exists
        if "eval" not in config_dict:
            config_dict["eval"] = {}
        if "evaluators" not in config_dict["eval"]:
            config_dict["eval"]["evaluators"] = {}

        # Build evaluator config from base
        evaluator_dict = self.config.evaluator.model_dump(mode='python', exclude_unset=False)

        # Force the LLM name to the fixed value
        evaluator_dict["llm_name"] = RED_TEAMING_EVALUATOR_LLM_NAME

        # Apply scenario-specific overrides
        if scenario.evaluation_instructions is not None:
            evaluator_dict["scenario_specific_instructions"] = scenario.evaluation_instructions
            logger.debug("Applied scenario evaluation_instructions override")

        if scenario.filter_conditions is not None:
            evaluator_dict["filter_conditions"] = [
                fc.model_dump(mode='python') for fc in scenario.filter_conditions
            ]
            logger.debug("Applied scenario filter_conditions override")

        # Add evaluator to config
        config_dict["eval"]["evaluators"]["red_teaming_evaluator"] = evaluator_dict

    def _update_config_value(self, config_dict: dict[str, typing.Any], path: str, value: typing.Any) -> None:
        """Update a single value in the config dictionary at the specified path.

        Args:
            config_dict: The configuration dictionary to update.
            path: The path to the value to update.
            value: The new value to set at the specified path.
        """

        parts = path.split('.')
        current = config_dict
        # Navigate through nested dictionaries until reaching the parent of target
        for part in parts[:-1]:
            current = current[part]
        # Update the value at the target location
        current[parts[-1]] = value

    def _find_red_teaming_evaluator_results(self, results: dict[str, EvaluationRunOutput]) -> dict[str, EvalOutput]:
        """Find the red teaming evaluator results in the results.

        Args:
            results: The results of the red teaming evaluation.

        Returns:
            The red teaming evaluator results.
        """
        red_teaming_evaluator_results = {}
        for scenario_id, result in results.items():
            for evaluator_results in result.evaluation_results:
                evaluator_name = evaluator_results[0]
                if evaluator_name == 'red_teaming_evaluator':
                    red_teaming_evaluator_results[scenario_id] = evaluator_results[1]
        return red_teaming_evaluator_results

    def _compute_result_summary(self, results: dict[str, EvaluationRunOutput]) -> dict[str, typing.Any]:
        """Compute the result summary for the red teaming evaluation.

        Args:
            results: The results of the red teaming evaluation.

        Returns:
            The result summary.
        """

        summary = {}
        per_scenario_summary = {}
        mean_scores = []
        evaluator_results = self._find_red_teaming_evaluator_results(results)
        for scenario_id, result in evaluator_results.items():
            scenario_scores = []
            for eval_output_item in result.eval_output_items:
                scenario_scores.append(eval_output_item.score)
            # Eval result per evaluator.
            mean_scores.append(result.average_score)
            per_scenario_summary[scenario_id] = {
                "mean_score": result.average_score,
                "max_score": max(scenario_scores),
                "min_score": min(scenario_scores),
                "scores": scenario_scores,
            }
        overall_sum = sum(mean_scores)
        overall_mean = overall_sum / len(mean_scores) if overall_sum > 0 else 0.0
        summary['overall_score'] = overall_mean
        summary['per_scenario_summary'] = per_scenario_summary
        return summary

    def _log_results_summary(self, summary: dict[str, typing.Any], output_dir: Path) -> None:
        """Log a nicely formatted summary of the red teaming evaluation results.

        Args:
            summary: The computed summary dictionary with overall_score and per_scenario_summary.
            output_dir: The base output directory where results are saved.
        """
        per_scenario = summary.get('per_scenario_summary', {})
        overall_score = summary.get('overall_score', 0.0)

        # Build the output lines
        lines = [
            "",
            "=" * 70,
            "  RED TEAMING EVALUATION RESULTS",
            "=" * 70,
            "",
        ]

        # Overall score
        lines.append(f"  Overall Score: {overall_score:.4f}")
        lines.append("")

        # Build table
        if per_scenario:
            # Calculate column widths
            scenario_ids = list(per_scenario.keys())
            max_scenario_len = max(len(sid) for sid in scenario_ids)
            scenario_col_width = max(max_scenario_len, len("Scenario"))

            # Table header
            lines.append("  " + "-" * (scenario_col_width + 45))
            header = f"  {'Scenario':<{scenario_col_width}}  |  {'Mean':>8}  |  {'Max':>8}  |  {'Min':>8}"
            lines.append(header)
            lines.append("  " + "-" * (scenario_col_width + 45))

            # Table rows
            for scenario_id, data in per_scenario.items():
                mean_val = data.get('mean_score', 0.0)
                max_val = data.get('max_score', 0.0)
                min_val = data.get('min_score', 0.0)
                row = (
                    f"  {scenario_id:<{scenario_col_width}}  |  "
                    f"{mean_val:>8.4f}  |  {max_val:>8.4f}  |  {min_val:>8.4f}"
                )
                lines.append(row)

            lines.append("  " + "-" * (scenario_col_width + 45))

        lines.append("")
        lines.append(f"  Output Directory: {output_dir.resolve()}")
        lines.append("")
        lines.append("=" * 70)
        lines.append("")

        # Log the formatted output
        logger.info("\n".join(lines))

__all__ = [
    "RedTeamingRunner",
]

