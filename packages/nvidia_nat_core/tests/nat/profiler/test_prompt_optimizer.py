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
import json
import typing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.config import Config
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.optimizable import SearchSpace
from nat.data_models.optimizer import OptimizerConfig
from nat.data_models.optimizer import OptimizerMetric
from nat.data_models.optimizer import OptimizerRunConfig
from nat.profiler.parameter_optimization.prompt_optimizer import PromptOptimizerInputSchema
from nat.profiler.parameter_optimization.prompt_optimizer import optimize_prompts

# Module-level tracking for oracle feedback verification in tests
oracle_feedback_received: dict[str, typing.Any] = {"count": 0, "values": []}


def _make_optimizer_config(tmp_path: Path) -> OptimizerConfig:
    cfg = OptimizerConfig(
        output_path=tmp_path,
        eval_metrics={"acc": OptimizerMetric(evaluator_name="Accuracy", direction="maximize", weight=1.0)},
        reps_per_param_set=2,
    )
    # Keep GA small/fast for tests
    cfg.prompt.ga_population_size = 3
    cfg.prompt.ga_generations = 1
    cfg.prompt.ga_elitism = 0
    cfg.prompt.ga_parallel_evaluations = 2
    cfg.prompt.ga_crossover_rate = 0.0
    cfg.prompt.ga_mutation_rate = 0.0
    # Functions to be provided by the builder in tests
    cfg.prompt.prompt_population_init_function = "init_fn"
    cfg.prompt.prompt_recombination_function = "recombine_fn"
    return cfg


def _make_run_config(cfg: Config) -> OptimizerRunConfig:
    return OptimizerRunConfig(
        config_file=cfg,
        dataset=None,
        result_json_path="$",
        endpoint=None,
        endpoint_timeout=5,
    )


async def test_optimize_prompts_no_prompt_space(tmp_path: Path):
    base_cfg = Config()
    optimizer_config = _make_optimizer_config(tmp_path)
    run_cfg = _make_run_config(base_cfg)

    # No prompt params in the space -> early return, no errors
    await optimize_prompts(base_cfg=base_cfg, full_space={}, optimizer_config=optimizer_config, opt_run_config=run_cfg)


async def test_optimize_prompts_requires_eval_metrics(tmp_path: Path):
    base_cfg = Config()
    optimizer_config = _make_optimizer_config(tmp_path)
    # Required to pass the prompt-space check
    full_space = {"prompt_param": SearchSpace(is_prompt=True, prompt="Hello", prompt_purpose="Greet")}
    optimizer_config.eval_metrics = None
    run_cfg = _make_run_config(base_cfg)

    with pytest.raises(ValueError):
        await optimize_prompts(base_cfg=base_cfg,
                               full_space=full_space,
                               optimizer_config=optimizer_config,
                               opt_run_config=run_cfg)


class InitFunctionConfig(FunctionBaseConfig, name="ga_init_test"):
    pass


class RecombineFunctionConfig(FunctionBaseConfig, name="ga_recombine_test"):
    pass


async def _register_prompt_optimizer_functions():

    @register_function(config_type=InitFunctionConfig)
    async def _register_init(_config: InitFunctionConfig, _b: Builder):  # noqa: ARG001

        async def _init_fn(value: PromptOptimizerInputSchema) -> str:
            # Track oracle feedback for test verification
            if value.oracle_feedback:
                oracle_feedback_received["count"] += 1
                oracle_feedback_received["values"].append(value.oracle_feedback)

            return f"mut({value.original_prompt})"

        yield FunctionInfo.from_fn(_init_fn)

    @register_function(config_type=RecombineFunctionConfig)
    async def _register_recombine(_config: RecombineFunctionConfig, _b: Builder):  # noqa: ARG001

        async def _recombine_fn(value: typing.Any) -> str:  # noqa: ANN001
            if isinstance(value, dict):
                a = value.get("original_prompt", "")
                bprompt = value.get("parent_b", "")
                return f"rec({a}|{bprompt})"
            return "rec(UNKNOWN)"

        yield FunctionInfo.from_fn(_recombine_fn)


async def test_optimize_prompts_happy_path_with_recombine(tmp_path: Path):
    await _register_prompt_optimizer_functions()
    base_cfg = Config()
    optimizer_config = _make_optimizer_config(tmp_path)

    # Provide one prompt param
    full_space = {"prompt_param": SearchSpace(is_prompt=True, prompt="Base", prompt_purpose="Greet")}

    run_cfg = _make_run_config(base_cfg)
    # Add real functions to builder via config; names match optimizer_config
    base_cfg.functions = {
        "init_fn": InitFunctionConfig(),
        "recombine_fn": RecombineFunctionConfig(),
    }
    base_cfg.workflow = InitFunctionConfig()

    # Counters to validate evaluation repetitions
    eval_calls = {"count": 0}

    class _EvalRun:

        def __init__(self, config):  # noqa: ANN001
            self.config = config

        async def run_and_evaluate(self):
            eval_calls["count"] += 1
            return SimpleNamespace(evaluation_results=[("Accuracy", SimpleNamespace(average_score=0.9))])

    def fake_apply_suggestions(cfg, prompts):  # noqa: ANN001
        # Return a new Config to simulate applied prompts
        _ = (cfg, prompts)
        return Config()

    with patch("nat.profiler.parameter_optimization.prompt_optimizer.EvaluationRun",
               _EvalRun), \
         patch("nat.profiler.parameter_optimization.prompt_optimizer.apply_suggestions",
               side_effect=fake_apply_suggestions):

        await optimize_prompts(base_cfg=base_cfg,
                               full_space=full_space,
                               optimizer_config=optimizer_config,
                               opt_run_config=run_cfg)

    # Files should be produced
    final_path = optimizer_config.output_path / "optimized_prompts.json"
    hist_path = optimizer_config.output_path / "ga_history_prompts.csv"
    ckpt_path = optimizer_config.output_path / "optimized_prompts_gen1.json"
    assert final_path.exists()
    assert hist_path.exists()
    assert ckpt_path.exists()

    # Final JSON structure contains our prompt param with [prompt, purpose]
    with open(final_path, encoding="utf-8") as f:
        best_prompts = json.load(f)
    assert "prompt_param" in best_prompts
    val = best_prompts["prompt_param"]
    assert isinstance(val, list) and len(val) == 2

    # We ran at least once; lower bound: population size * reps (approximate)
    assert eval_calls["count"] >= optimizer_config.prompt.ga_population_size * optimizer_config.reps_per_param_set


async def test_optimize_prompts_happy_path_without_recombine(tmp_path: Path):
    await _register_prompt_optimizer_functions()
    base_cfg = Config()
    optimizer_config = _make_optimizer_config(tmp_path)
    # Remove recombination function to exercise fallback path
    optimizer_config.prompt.prompt_recombination_function = None

    full_space = {"p": SearchSpace(is_prompt=True, prompt="X", prompt_purpose="Y")}
    run_cfg = _make_run_config(base_cfg)
    base_cfg.functions = {
        "init_fn": InitFunctionConfig(),
    }
    base_cfg.workflow = InitFunctionConfig()

    class _EvalRun:

        def __init__(self, config):  # noqa: ANN001
            self.config = config

        async def run_and_evaluate(self):
            return SimpleNamespace(evaluation_results=[("Accuracy", SimpleNamespace(average_score=0.5))])

    def fake_apply_suggestions(cfg, prompts):  # noqa: ANN001
        _ = (cfg, prompts)
        return Config()

    with patch("nat.profiler.parameter_optimization.prompt_optimizer.EvaluationRun",
               _EvalRun), \
         patch("nat.profiler.parameter_optimization.prompt_optimizer.apply_suggestions",
               side_effect=fake_apply_suggestions):

        await optimize_prompts(base_cfg=base_cfg,
                               full_space=full_space,
                               optimizer_config=optimizer_config,
                               opt_run_config=run_cfg)

    # Outputs exist
    assert (optimizer_config.output_path / "optimized_prompts.json").exists()
    assert (optimizer_config.output_path / "ga_history_prompts.csv").exists()
    assert (optimizer_config.output_path / "optimized_prompts_gen1.json").exists()


async def test_optimize_prompts_with_oracle_feedback(tmp_path: Path):
    """Test that oracle feedback is extracted and passed to mutations."""
    # Reset the oracle feedback tracker
    oracle_feedback_received["count"] = 0
    oracle_feedback_received["values"] = []

    await _register_prompt_optimizer_functions()
    base_cfg = Config()
    optimizer_config = _make_optimizer_config(tmp_path)

    # Enable oracle feedback
    optimizer_config.prompt.oracle_feedback_mode = "always"
    optimizer_config.prompt.oracle_feedback_worst_n = 2
    optimizer_config.prompt.oracle_feedback_max_chars = 1000

    # Enable mutations so feedback gets passed (default config has mutation_rate=0.0)
    optimizer_config.prompt.ga_mutation_rate = 1.0  # Always mutate
    optimizer_config.prompt.ga_generations = 2  # Need 2+ generations for offspring with feedback

    full_space = {"prompt_param": SearchSpace(is_prompt=True, prompt="Base", prompt_purpose="Greet")}
    run_cfg = _make_run_config(base_cfg)
    base_cfg.functions = {
        "init_fn": InitFunctionConfig(),
        "recombine_fn": RecombineFunctionConfig(),
    }
    base_cfg.workflow = InitFunctionConfig()

    class _EvalRun:

        def __init__(self, config):  # noqa: ANN001
            self.config = config

        async def run_and_evaluate(self):
            from nat.eval.evaluator.evaluator_model import EvalOutput
            from nat.eval.evaluator.evaluator_model import EvalOutputItem

            items = [
                EvalOutputItem(id=1, score=0.2, reasoning="Failed to greet properly"),
                EvalOutputItem(id=2, score=0.8, reasoning="Good greeting"),
            ]
            eval_output = EvalOutput(average_score=0.5, eval_output_items=items)
            return SimpleNamespace(evaluation_results=[("Accuracy", eval_output)])

    def fake_apply_suggestions(cfg, prompts):  # noqa: ANN001
        _ = (cfg, prompts)
        return Config()

    with patch("nat.profiler.parameter_optimization.prompt_optimizer.EvaluationRun", _EvalRun), \
         patch("nat.profiler.parameter_optimization.prompt_optimizer.apply_suggestions",
               side_effect=fake_apply_suggestions):

        await optimize_prompts(
            base_cfg=base_cfg,
            full_space=full_space,
            optimizer_config=optimizer_config,
            opt_run_config=run_cfg,
        )

    # Verify output files created
    assert (optimizer_config.output_path / "optimized_prompts.json").exists()

    # Verify oracle feedback was passed to at least one mutation
    # With mutation_rate=1.0 and generations=2, feedback should be passed to offspring
    assert oracle_feedback_received["count"] > 0, "Oracle feedback should have been passed to at least one mutation"
    # Verify the feedback content contains the expected reasoning from worst-scoring items
    assert any("Failed to greet properly" in fb for fb in oracle_feedback_received["values"]), \
        "Feedback should contain reasoning from worst-scoring evaluation items"
