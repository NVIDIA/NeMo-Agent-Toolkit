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

from typing import cast

import pytest

import nat.middleware.register  # noqa: F401  # Import register module to trigger registration
from nat.builder.builder import Builder
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.config import Config
from nat.data_models.dataset_handler import EvalDatasetJsonConfig
from nat.data_models.evaluate import EvalConfig
from nat.data_models.evaluate import EvalGeneralConfig
from nat.data_models.function import EmptyFunctionConfig
from nat.data_models.function import FunctionGroupBaseConfig
from nat.eval.red_teaming_evaluator.filter_conditions import IntermediateStepsFilterCondition
from nat.eval.red_teaming_evaluator.register import RedTeamingEvaluatorConfig
from nat.eval.runners.red_teaming_runner.config import RedTeamingRunnerConfig
from nat.eval.runners.red_teaming_runner.config import RedTeamingScenario
from nat.eval.runners.red_teaming_runner.config import _RedTeamingScenarioRaw
from nat.eval.runners.red_teaming_runner.runner import RedTeamingRunner
from nat.llm.nim_llm import NIMModelConfig
from nat.middleware.red_teaming.red_teaming_middleware_config import RedTeamingMiddlewareConfig


class SimpleFunctionGroupConfig(FunctionGroupBaseConfig, name="simple_function_group"):
    """Simple function group config for testing."""
    pass


@pytest.fixture(scope="module", autouse=True)
async def register_test_types():
    """Register test types with GlobalTypeRegistry."""

    @register_function_group(config_type=SimpleFunctionGroupConfig)
    async def simple_function_group(config: SimpleFunctionGroupConfig, _builder: Builder):
        """Test function group for scenario tests."""
        group = FunctionGroup(config=config)
        yield group


@pytest.fixture(name="base_config")
def fixture_base_config() -> Config:
    """Minimal base workflow config with functions, function_groups, and workflow."""
    return Config(
        llms={"workflow_llm": NIMModelConfig(model_name="test-model")},
        functions={"my_func": EmptyFunctionConfig()},
        function_groups={"my_group": SimpleFunctionGroupConfig()},
        workflow=EmptyFunctionConfig(),
        eval=EvalConfig(general=EvalGeneralConfig(
            dataset=EvalDatasetJsonConfig(file_path="base_dataset.json"),
            max_concurrency=2,
        )),
    )


@pytest.fixture(name="red_teaming_config")
def fixture_red_teaming_config() -> RedTeamingRunnerConfig:
    """Red teaming config with one attack scenario and one baseline."""

    evaluator_config = RedTeamingEvaluatorConfig(
        llm_name="test_judge_llm",  # type: ignore[arg-type]
        judge_llm_prompt="Evaluate attack success",
        filter_conditions=[IntermediateStepsFilterCondition(name="default")],
        scenario_specific_instructions="Check for 42",
    )

    return RedTeamingRunnerConfig(
        llms={"test_judge_llm": NIMModelConfig(model_name="test-judge-model")},
        general=EvalGeneralConfig(max_concurrency=2),
        scenarios={
            "attack_42":
                RedTeamingScenario(
                    middleware=RedTeamingMiddlewareConfig(attack_payload="42"),
                    evaluator=evaluator_config,
                ),
            "baseline":
                RedTeamingScenario(middleware=None, evaluator=evaluator_config),
        },
    )


@pytest.fixture(name="red_teaming_config_with_extends")
def fixture_red_teaming_config_with_extends() -> RedTeamingRunnerConfig:
    """Red teaming config with one attack scenario and one baseline."""

    evaluator_config = {"_extends": "test"}

    return RedTeamingRunnerConfig(
        llms={"test_judge_llm": NIMModelConfig(model_name="test-judge-llm")},
        evaluator_defaults={
            "test":
                RedTeamingEvaluatorConfig(
                    llm_name="test_judge_llm",  # type: ignore[arg-type]
                    judge_llm_prompt="Evaluate attack success",
                    filter_conditions=[IntermediateStepsFilterCondition(name="default")],
                )
        },
        general=EvalGeneralConfig(max_concurrency=2),
        scenarios={
            "attack_42":
                _RedTeamingScenarioRaw(
                    middleware=RedTeamingMiddlewareConfig(attack_payload="42"),
                    evaluator=evaluator_config,
                ),
            "baseline":
                _RedTeamingScenarioRaw(middleware=None, evaluator=evaluator_config),
        },
    )


def test_middleware_attached_everywhere(base_config: Config, red_teaming_config: RedTeamingRunnerConfig):
    """Middleware should be attached to all functions, function_groups, and workflow."""
    runner = RedTeamingRunner(config=red_teaming_config, base_workflow_config=base_config)
    configs = runner.generate_workflow_configs()
    attack_config = configs["attack_42"]

    middleware_name = "red_teaming_attack_42"
    assert middleware_name in attack_config.middleware
    assert middleware_name in attack_config.functions["my_func"].middleware
    assert middleware_name in attack_config.function_groups["my_group"].middleware
    assert middleware_name in attack_config.workflow.middleware


def test_evaluator_injected_with_scenario_overrides(base_config: Config, red_teaming_config: RedTeamingRunnerConfig):
    """Evaluator config should be injected with fixed LLM name and scenario overrides."""
    runner = RedTeamingRunner(config=red_teaming_config, base_workflow_config=base_config)
    configs = runner.generate_workflow_configs()
    workflow_config = configs["attack_42"]

    # Evaluator LLM added with fixed name
    assert "test_judge_llm" in workflow_config.llms

    # Evaluator present in eval section
    assert "red_teaming_evaluator" in workflow_config.eval.evaluators
    evaluator = cast(RedTeamingEvaluatorConfig, workflow_config.eval.evaluators["red_teaming_evaluator"])

    # Fixed LLM name and scenario override applied
    assert evaluator.llm_name == "test_judge_llm"
    assert evaluator.scenario_specific_instructions == "Check for 42"


def test_baseline_scenario_no_middleware(base_config: Config, red_teaming_config: RedTeamingRunnerConfig):
    """Baseline scenario should not add any red teaming middleware."""
    runner = RedTeamingRunner(config=red_teaming_config, base_workflow_config=base_config)
    configs = runner.generate_workflow_configs()
    baseline_config = configs["baseline"]

    # No red_teaming middleware should exist
    red_team_middlewares = [k for k in baseline_config.middleware if k.startswith("red_teaming")]
    assert len(red_team_middlewares) == 0


def test_general_config_merged(base_config: Config):
    """RedTeamingRunnerConfig.general should merge with base config, overriding specified fields only."""
    evaluator_config = RedTeamingEvaluatorConfig(
        llm_name="test_judge_llm",  # type: ignore[arg-type]
        judge_llm_prompt="prompt",
        filter_conditions=[IntermediateStepsFilterCondition(name="default")],
    )
    rt_config = RedTeamingRunnerConfig(
        llms={"test_judge_llm": NIMModelConfig(model_name="test-judge-llm")},
        general=EvalGeneralConfig(max_concurrency=10),  # Override only this
        scenarios={"test": RedTeamingScenario(middleware=None, evaluator=evaluator_config)},
    )

    runner = RedTeamingRunner(config=rt_config, base_workflow_config=base_config)
    configs = runner.generate_workflow_configs()
    result = configs["test"]

    # max_concurrency overridden, dataset preserved from base
    assert result.eval.general.max_concurrency == 10
    assert result.eval.general.dataset is not None
    assert str(result.eval.general.dataset.file_path) == "base_dataset.json"


def test_dataset_validation_error(red_teaming_config: RedTeamingRunnerConfig):
    """Should raise ValueError when no dataset is defined anywhere."""

    base_config = Config(workflow=EmptyFunctionConfig())  # No dataset anywhere

    runner = RedTeamingRunner(config=red_teaming_config, base_workflow_config=base_config)
    with pytest.raises(ValueError, match="No dataset defined"):
        runner.generate_workflow_configs()


def test_direct_config_validation_requires_middleware_and_evaluator():
    """When no RedTeamingRunnerConfig provided, base_config must have middleware and evaluator."""
    base_config = Config(workflow=EmptyFunctionConfig())

    runner = RedTeamingRunner(config=None, base_workflow_config=base_config)
    with pytest.raises(ValueError, match="not red-team compatible"):
        runner.generate_workflow_configs()
