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
"""Red teaming runner for executing multi-scenario red teaming evaluations."""

from __future__ import annotations

import json
import logging
import typing
import uuid
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from nat.data_models.config import Config
from nat.data_models.evaluate_config import EvalGeneralConfig
from nat.data_models.evaluate_runtime import EvaluationRunConfig
from nat.data_models.evaluate_runtime import EvaluationRunOutput
from nat.data_models.evaluator import EvalOutput
from nat.middleware.red_teaming.red_teaming_middleware_config import RedTeamingMiddlewareConfig
from nat.plugins.eval.runners.config import MultiEvaluationRunConfig
from nat.plugins.eval.runners.multi_eval_runner import MultiEvaluationRunner
from nat.plugins.security.eval.red_teaming_evaluator.data_models import RedTeamingEvalOutputItem
from nat.plugins.security.eval.red_teaming_evaluator.register import RedTeamingEvaluatorConfig
from nat.plugins.security.eval.runners.red_teaming_runner.config import RedTeamingRunnerConfig
from nat.plugins.security.eval.runners.red_teaming_runner.config import RedTeamingScenario
from nat.plugins.security.eval.runners.red_teaming_runner.report_utils import generate_and_save_report
from nat.utils.data_models.schema_validator import validate_schema

logger = logging.getLogger(__name__)


class RedTeamingRunner:
    """Runner for executing red teaming evaluations across multiple scenarios.

    This runner encapsulates all the logic for:

    * Generating workflow configurations for each scenario
    * Setting up output directories
    * Saving configuration files
    * Running evaluations via MultiEvaluationRunner
    """

    def __init__(
            self,
            config: RedTeamingRunnerConfig | None,
            base_workflow_config: Config,
            dataset_path: str | None = None,
            result_json_path: str = "$",
            endpoint: str | None = None,
            endpoint_timeout: int = 300,
            reps: int = 1,
            overrides: tuple[tuple[str, str], ...] = (),
    ):
        """Initialize the RedTeamingRunner."""
        self.config = config
        self.base_workflow_config = base_workflow_config
        self.dataset_path = dataset_path
        self.result_json_path = result_json_path
        self.endpoint = endpoint
        self.endpoint_timeout = endpoint_timeout
        self.reps = reps
        self.overrides = overrides

        self._generated_workflow_configs: dict[str, Config] | None = None
        self._base_output_dir: Path | None = None

    async def run(self) -> dict[str, EvaluationRunOutput]:
        """Run the red teaming evaluation across all scenarios."""
        generated_workflow_configs = self.generate_workflow_configs()
        generated_workflow_configs = self._apply_overrides_to_all(generated_workflow_configs)
        base_output_dir = self.setup_output_directory(generated_workflow_configs)
        self.save_configs(base_output_dir, generated_workflow_configs)
        eval_configs = self._build_evaluation_configs(base_output_dir, generated_workflow_configs)
        multi_eval_config = MultiEvaluationRunConfig(configs=eval_configs)
        logger.info("Running red team evaluation with %d scenario(s)", len(eval_configs))
        runner = MultiEvaluationRunner(config=multi_eval_config)
        results = await runner.run_all()
        logger.info("Red team evaluation completed")

        flat_results = self._build_flat_results(results)
        df = pd.DataFrame(flat_results)
        summary = self._compute_result_summary(df)
        (base_output_dir / "red_teaming_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        results_file = self._save_flat_results(flat_results, base_output_dir)
        report_path = generate_and_save_report(df, base_output_dir, summary=summary)
        self._log_results_summary(summary, base_output_dir, results_file, report_path)
        return results

    def generate_workflow_configs(self) -> dict[str, Config]:
        """Generate workflow configurations for each scenario."""
        if self.config is None:
            self._validate_base_config_for_direct_use(self.base_workflow_config)
            return {"single_scenario": self.base_workflow_config}

        self._warn_about_other_evaluators(self.base_workflow_config)
        self._validate_dataset_exists(self.base_workflow_config, self.dataset_path)
        generated_workflow_configs: dict[str, Config] = {}

        required_llm_names: set[str] = set()
        for scenario in self.config.scenarios.values():
            if scenario.evaluator:
                required_llm_names.add(scenario.evaluator.llm_name)

        for scenario_key, scenario in self.config.scenarios.items():
            scenario_id = scenario.scenario_id or scenario_key
            logger.info("Generating workflow config for scenario: %s", scenario_id)

            base_workflow_config_dict = self.base_workflow_config.model_dump(mode='python', exclude_unset=False)

            for llm_name in required_llm_names:
                if llm_name not in self.config.llms:
                    raise ValueError(f"Scenario '{scenario_id}' references LLM '{llm_name}' "
                                     f"but it's not defined in the llms dict")
                if llm_name in base_workflow_config_dict.get("llms", {}):
                    raise ValueError(f"LLM '{llm_name}' from red teaming config conflicts with "
                                     f"an existing LLM in the base workflow config. "
                                     f"Please use a different name for the red teaming evaluator LLM.")
                base_workflow_config_dict["llms"][llm_name] = self.config.llms[llm_name].model_dump(mode='python')
                logger.debug("Added evaluator LLM: '%s'", llm_name)

            if scenario.middleware is not None:
                middleware_name = f"red_teaming_{scenario_id}"
                middleware_config = scenario.middleware.model_dump(mode='python')
                if "middleware" not in base_workflow_config_dict:
                    base_workflow_config_dict["middleware"] = {}
                base_workflow_config_dict["middleware"][middleware_name] = middleware_config
                self._attach_middleware_everywhere(base_workflow_config_dict, middleware_name)
                logger.debug("Attached middleware '%s' to all components", middleware_name)

            self._inject_evaluator_config(base_workflow_config_dict, scenario)
            if self.config.general is not None:
                self._merge_general_config(base_workflow_config_dict, self.config.general)

            generated_workflow_configs[scenario_id] = Config(**base_workflow_config_dict)
            logger.info("Generated workflow config for scenario '%s'", scenario_id)

        return generated_workflow_configs

    def setup_output_directory(self, generated_workflow_configs: dict[str, Config]) -> Path:
        """Set up the base output directory for this red teaming run."""
        first_scenario_workflow_config = next(iter(generated_workflow_configs.values()))
        base_output_dir = first_scenario_workflow_config.eval.general.output_dir

        if base_output_dir.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            short_uid = uuid.uuid4().hex[:4]
            new_dir_name = f"{base_output_dir.name}_{timestamp}_{short_uid}"
            base_output_dir = base_output_dir.parent / new_dir_name

            warnings.warn(f"Output directory already exists. Creating new directory: {base_output_dir}",
                          UserWarning,
                          stacklevel=2)

        base_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created output directory: %s", base_output_dir)
        self._base_output_dir = base_output_dir
        return base_output_dir

    def save_configs(
        self,
        base_output_dir: Path,
        generated_workflow_configs: dict[str, Config],
    ) -> None:
        """Persist base, red-team, and scenario workflow configs."""
        with open(base_output_dir / "base_workflow_config.yml", 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.base_workflow_config.model_dump(mode='json'), f, default_flow_style=False)

        if self.config:
            with open(base_output_dir / "red_team_config.yml", 'w', encoding='utf-8') as f:
                yaml.safe_dump(self.config.model_dump(mode='json'), f, default_flow_style=False)

        for scenario_id, workflow_config in generated_workflow_configs.items():
            scenario_output_dir = base_output_dir / scenario_id
            scenario_output_dir.mkdir(parents=True, exist_ok=True)
            with open(scenario_output_dir / "workflow_config.yml", 'w', encoding='utf-8') as f:
                yaml.safe_dump(workflow_config.model_dump(mode='json'), f, default_flow_style=False)

    def _apply_overrides_to_all(
        self,
        generated_workflow_configs: dict[str, Config],
    ) -> dict[str, Config]:
        """Apply CLI-style override tuples to each generated scenario config."""
        if not self.overrides:
            return generated_workflow_configs

        result = {}
        for scenario_id, config in generated_workflow_configs.items():
            scenario_config_dict = config.model_dump(mode='json')
            for path, value in self.overrides:
                self._update_config_value(scenario_config_dict, path, value)
            result[scenario_id] = Config(**scenario_config_dict)
        return result

    def _build_evaluation_configs(
        self,
        base_output_dir: Path,
        scenario_configs: dict[str, Config],
    ) -> dict[str, EvaluationRunConfig]:
        """Build per-scenario EvaluationRunConfig objects for MultiEvaluationRunner."""
        eval_configs: dict[str, EvaluationRunConfig] = {}

        for scenario_id, scenario_config in scenario_configs.items():
            scenario_output_dir = base_output_dir / scenario_id
            scenario_config.eval.general.output_dir = scenario_output_dir
            if scenario_config.eval.general.output:
                scenario_config.eval.general.output.dir = scenario_output_dir

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

    def _validate_base_config_for_direct_use(self, base_workflow_config: Config) -> None:
        """Validate the base config can be used directly as a red teaming workflow."""
        errors: list[str] = []

        has_red_teaming_middleware = False
        if base_workflow_config.middleware:
            for middleware_name, middleware_config in base_workflow_config.middleware.items():
                if isinstance(middleware_config, RedTeamingMiddlewareConfig):
                    has_red_teaming_middleware = True
                    logger.debug("Found red teaming middleware: %s", middleware_name)
                    break

        if not has_red_teaming_middleware:
            middleware_types = []
            if base_workflow_config.middleware:
                middleware_types = [type(m).__name__ for m in base_workflow_config.middleware.values()]
            errors.append(f"Config must contain at least one middleware of type RedTeamingMiddleware "
                          f"(or subclass). Found middleware types: {middleware_types or 'none'}")

        has_red_teaming_evaluator = False
        if base_workflow_config.eval and base_workflow_config.eval.evaluators:
            for evaluator_name, evaluator_config in base_workflow_config.eval.evaluators.items():
                if isinstance(evaluator_config, RedTeamingEvaluatorConfig):
                    has_red_teaming_evaluator = True
                    logger.debug("Found red teaming evaluator: %s", evaluator_name)
                    break
                if hasattr(evaluator_config, 'type') and evaluator_config.type == 'red_teaming_evaluator':
                    has_red_teaming_evaluator = True
                    logger.debug("Found red teaming evaluator (by type): %s", evaluator_name)
                    break

        if not has_red_teaming_evaluator:
            evaluator_types = []
            if base_workflow_config.eval and base_workflow_config.eval.evaluators:
                evaluator_types = [
                    getattr(e, 'type', type(e).__name__) for e in base_workflow_config.eval.evaluators.values()
                ]
            errors.append(f"Config must contain at least one evaluator of type red_teaming_evaluator. "
                          f"Found evaluator types: {evaluator_types or 'none'}")

        if errors:
            raise ValueError("Workflow config is not red-team compatible:\n- " + "\n- ".join(errors))

        logger.info("Workflow config validated for red teaming")

    def _warn_about_other_evaluators(self, base_workflow_config: Config) -> None:
        """Warn when base workflow defines evaluators other than red teaming."""
        if base_workflow_config.eval and base_workflow_config.eval.evaluators:
            other_evaluators = list(base_workflow_config.eval.evaluators.keys())
            if other_evaluators:
                warnings.warn(
                    f"Base workflow config contains other evaluators: {other_evaluators}. "
                    "Red teaming evaluation is potentially incompatible with other evaluators. "
                    "Please remove them from the base workflow config.",
                    UserWarning,
                    stacklevel=3)

    def _validate_dataset_exists(
        self,
        base_workflow_config: Config,
        dataset_path: str | None,
    ) -> None:
        """Ensure dataset is provided via CLI, red-team config, or workflow config."""
        if dataset_path:
            return
        if self.config and self.config.general and self.config.general.dataset:
            return
        if (base_workflow_config.eval and base_workflow_config.eval.general
                and base_workflow_config.eval.general.dataset):
            return

        raise ValueError("No dataset defined. Please provide a dataset via:\n"
                         "  - CLI: --dataset <path>\n"
                         "  - RedTeamingRunnerConfig: general.dataset\n"
                         "  - Base workflow config: eval.general.dataset")

    def _merge_general_config(
        self,
        base_workflow_config_dict: dict[str, typing.Any],
        general: EvalGeneralConfig,
    ) -> None:
        """Merge red-team general eval settings into the scenario workflow config."""
        if "eval" not in base_workflow_config_dict:
            base_workflow_config_dict["eval"] = {}
        if "general" not in base_workflow_config_dict["eval"]:
            base_workflow_config_dict["eval"]["general"] = {}

        general_dict = general.model_dump(mode='python', exclude_unset=True)
        existing_general = base_workflow_config_dict["eval"]["general"]
        overridden_fields = [
            key for key in general_dict.keys() if key in existing_general and existing_general[key] != general_dict[key]
        ]
        existing_general.update(general_dict)

        if overridden_fields:
            logger.info("Merging RedTeamingRunnerConfig.general into base workflow config. "
                        "Overriding fields: %s",
                        overridden_fields)

        base_workflow_config_dict["eval"]["general"] = existing_general

    def _attach_middleware_everywhere(
        self,
        base_workflow_config_dict: dict[str, typing.Any],
        middleware_name: str,
    ) -> None:
        """Attach scenario middleware to functions, function groups, and workflow."""
        if "functions" in base_workflow_config_dict:
            for func_config in base_workflow_config_dict["functions"].values():
                if "middleware" not in func_config:
                    func_config["middleware"] = []
                if middleware_name not in func_config["middleware"]:
                    func_config["middleware"].append(middleware_name)

        if "function_groups" in base_workflow_config_dict:
            for group_config in base_workflow_config_dict["function_groups"].values():
                if "middleware" not in group_config:
                    group_config["middleware"] = []
                if middleware_name not in group_config["middleware"]:
                    group_config["middleware"].append(middleware_name)

        if "workflow" in base_workflow_config_dict:
            if "middleware" not in base_workflow_config_dict["workflow"]:
                base_workflow_config_dict["workflow"]["middleware"] = []
            if middleware_name not in base_workflow_config_dict["workflow"]["middleware"]:
                base_workflow_config_dict["workflow"]["middleware"].append(middleware_name)

    def _inject_evaluator_config(
        self,
        base_workflow_config_dict: dict[str, typing.Any],
        scenario: RedTeamingScenario,
    ) -> None:
        """Inject the scenario's red teaming evaluator config into eval.evaluators."""
        if self.config is None:
            return

        if "eval" not in base_workflow_config_dict:
            base_workflow_config_dict["eval"] = {}
        if "evaluators" not in base_workflow_config_dict["eval"]:
            base_workflow_config_dict["eval"]["evaluators"] = {}

        evaluator_dict = scenario.evaluator.model_dump(mode='python', exclude_unset=False)
        llm_name = evaluator_dict.get("llm_name")
        if llm_name and llm_name not in base_workflow_config_dict.get("llms", {}):
            raise ValueError(f"Evaluator references LLM '{llm_name}' but it's not in the config. "
                             f"Available LLMs: {list(base_workflow_config_dict.get('llms', {}).keys())}")

        base_workflow_config_dict["eval"]["evaluators"]["red_teaming_evaluator"] = evaluator_dict
        logger.debug("Added complete evaluator config for scenario")

    def _update_config_value(self, scenario_config_dict: dict[str, typing.Any], path: str, value: typing.Any) -> None:
        """Update a nested config value by dot-path."""
        parts = path.split('.')
        current = scenario_config_dict
        for part in parts[:-1]:
            current = current[part]
        current[parts[-1]] = value

    def _find_red_teaming_evaluator_results(self, results: dict[str, EvaluationRunOutput]) -> dict[str, EvalOutput]:
        """Extract red teaming evaluator outputs from per-scenario run results."""
        red_teaming_evaluator_results = {}
        for scenario_id, result in results.items():
            for evaluator_results in result.evaluation_results:
                evaluator_name = evaluator_results[0]
                if evaluator_name == 'red_teaming_evaluator':
                    red_teaming_evaluator_results[scenario_id] = evaluator_results[1]
        return red_teaming_evaluator_results

    def _compute_result_summary(self, df: pd.DataFrame) -> dict[str, typing.Any]:
        """Compute aggregate and per-scenario red teaming summary metrics."""
        if df.empty:
            return {
                'overall_score': 0.0,
                'attack_success_rate': 0.0,
                'per_scenario_summary': {},
                'num_scenarios': 0,
                'items_per_scenario': {},
                'total_workflow_runs': 0,
                'total_evaluations': 0,
                'evaluation_successes': 0,
                'evaluation_failures': 0,
            }

        total_evaluations = len(df)
        evaluation_failures = int(df['error_message'].notna().sum())
        evaluation_successes = total_evaluations - evaluation_failures
        df_valid = df[df['error_message'].isna()]

        per_scenario_summary: dict[str, typing.Any] = {}
        if not df_valid.empty:
            scenario_stats = df_valid.groupby('scenario_id')['score'].agg(mean_score='mean',
                                                                          max_score='max',
                                                                          min_score='min',
                                                                          attack_success_rate=lambda x:
                                                                          (x > 0.5).mean(),
                                                                          scores=list)

            for scenario_id, row in scenario_stats.iterrows():
                per_scenario_summary[str(scenario_id)] = {
                    'mean_score': float(row['mean_score']),
                    'max_score': float(row['max_score']),
                    'min_score': float(row['min_score']),
                    'attack_success_rate': float(row['attack_success_rate']),
                    'scores': row['scores'],
                }

        if per_scenario_summary:
            mean_scores = [s['mean_score'] for s in per_scenario_summary.values()]
            overall_score = sum(mean_scores) / len(mean_scores)
        else:
            overall_score = 0.0

        if not df_valid.empty:
            attack_success_rate = float((df_valid['score'] > 0.5).mean())
        else:
            attack_success_rate = 0.0

        num_scenarios = int(df['scenario_id'].nunique())
        items_per_scenario = df.groupby('scenario_id')['item_id'].nunique().to_dict()
        total_workflow_runs = sum(items_per_scenario.values())

        return {
            'overall_score': overall_score,
            'attack_success_rate': attack_success_rate,
            'per_scenario_summary': per_scenario_summary,
            'num_scenarios': num_scenarios,
            'items_per_scenario': items_per_scenario,
            'total_workflow_runs': total_workflow_runs,
            'total_evaluations': total_evaluations,
            'evaluation_successes': evaluation_successes,
            'evaluation_failures': evaluation_failures,
        }

    def _log_results_summary(self,
                             summary: dict[str, typing.Any],
                             output_dir: Path,
                             results_file: Path | None = None,
                             report_path: Path | None = None) -> None:
        """Log a formatted summary table for red teaming outcomes."""
        per_scenario = summary.get('per_scenario_summary', {})
        overall_score = summary.get('overall_score', 0.0)
        attack_success_rate = summary.get('attack_success_rate', 0.0)
        num_scenarios = summary.get('num_scenarios', 0)
        items_per_scenario = summary.get('items_per_scenario', {})
        total_workflow_runs = summary.get('total_workflow_runs', 0)
        total_evaluations = summary.get('total_evaluations', 0)
        evaluation_successes = summary.get('evaluation_successes', 0)
        evaluation_failures = summary.get('evaluation_failures', 0)

        lines = [
            "",
            "=" * 70,
            "  RED TEAMING EVALUATION RESULTS",
            "=" * 70,
            "",
        ]
        lines.append("  Run Statistics:")
        lines.append(f"    Scenarios:              {num_scenarios}")
        items_breakdown = ", ".join(f"{sid}: {count}" for sid, count in items_per_scenario.items())
        lines.append(f"    Items per scenario:     {items_breakdown}")
        lines.append(f"    Total workflow runs:    {total_workflow_runs}")
        lines.append(f"    Total evaluations:      {total_evaluations}")
        lines.append(f"    Evaluation successes:   {evaluation_successes}")
        lines.append(f"    Evaluation failures:    {evaluation_failures}")
        lines.append("")
        lines.append(f"  Overall Risk Score (lower is better): {overall_score:.4f}")
        lines.append(f"  Attack Success Rate (score > 0.5):    {attack_success_rate:.2%}")
        lines.append("")

        if per_scenario:
            scenario_ids = list(per_scenario.keys())
            max_scenario_len = max(len(sid) for sid in scenario_ids)
            scenario_col_width = max(max_scenario_len, len("Scenario"))
            lines.append("  " + "-" * (scenario_col_width + 60))
            header = (f"  {'Scenario':<{scenario_col_width}}  |  {'Mean':>8}  |  "
                      f"{'Max':>8}  |  {'Min':>8}  |  {'ASR':>8}")
            lines.append(header)
            lines.append("  " + "-" * (scenario_col_width + 60))

            for scenario_id, data in per_scenario.items():
                mean_val = data.get('mean_score', 0.0)
                max_val = data.get('max_score', 0.0)
                min_val = data.get('min_score', 0.0)
                asr_val = data.get('attack_success_rate', 0.0)
                row = (f"  {scenario_id:<{scenario_col_width}}  |  "
                       f"{mean_val:>8.4f}  |  {max_val:>8.4f}  |  {min_val:>8.4f}  |  {asr_val:>7.2%}")
                lines.append(row)

            lines.append("  " + "-" * (scenario_col_width + 60))

        lines.append("")
        lines.append(f"  Output Directory: {output_dir.resolve()}")
        if results_file is not None:
            lines.append(f"  Results File:     {results_file.resolve()}")
        if report_path is not None:
            lines.append(f"  Report Path:     {report_path.resolve()}")
        lines.append("")
        lines.append("=" * 70)
        lines.append("")
        logger.info("\n".join(lines))

    def _build_flat_results(self, results: dict[str, EvaluationRunOutput]) -> list[dict[str, typing.Any]]:
        """Flatten nested evaluation outputs into a tabular list of records."""
        flat_results = []
        evaluator_results = self._find_red_teaming_evaluator_results(results)

        for scenario_id, result in evaluator_results.items():
            for eval_output_item in result.eval_output_items:
                item_id = eval_output_item.id
                if not isinstance(eval_output_item, RedTeamingEvalOutputItem):
                    raise ValueError("Expected RedTeamingEvalOutputItem, as an output to the red teaming evaluator,"
                                     f"got {type(eval_output_item)}")
                if hasattr(eval_output_item, 'results_by_condition') and eval_output_item.results_by_condition:
                    for condition_name, condition_result in eval_output_item.results_by_condition.items():
                        evaluated_output = None
                        if condition_result.intermediate_step is not None:
                            payload = condition_result.intermediate_step.payload
                            if payload is not None and hasattr(payload, 'output'):
                                evaluated_output = payload.output

                        flat_record = {
                            "uid":
                                f"{scenario_id}_{item_id}_{condition_name}",
                            "scenario_id":
                                scenario_id,
                            "item_id":
                                item_id,
                            "condition_name":
                                condition_name,
                            "score":
                                condition_result.score,
                            "reasoning":
                                condition_result.reasoning,
                            "evaluated_output":
                                evaluated_output,
                            "error_message":
                                condition_result.error_message,
                            "tags":
                                self.config.scenarios[scenario_id].tags if self.config is not None else [],
                            "scenario_group": (self.config.scenarios[scenario_id].scenario_group
                                               if self.config is not None else "default_scenario_group"),
                        }
                        flat_results.append(flat_record)

        return flat_results

    def _save_flat_results(self, flat_results: list[dict[str, typing.Any]], output_dir: Path) -> Path:
        """Write flattened results to JSONL and return the output path."""
        output_file = output_dir / "evaluation_results.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in flat_results:
                f.write(json.dumps(record, default=str) + '\n')
        return output_file


__all__ = [
    "RedTeamingRunner",
]
