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
from typing import cast

import pytest

from nat.data_models.component_ref import LLMRef
from nat.data_models.config import Config
from nat.data_models.evaluate import EvalConfig
from nat.data_models.function import EmptyFunctionConfig
from nat.data_models.function import FunctionGroupBaseConfig
from nat.eval.red_teaming_evaluator.filter_conditions import IntermediateStepsFilterCondition
from nat.eval.red_teaming_evaluator.register import RedTeamingEvaluatorConfig
from nat.eval.runners.red_team_eval_runner.red_team_eval_config import RedTeamScenarioEntry
from nat.middleware.cache_middleware import CacheMiddlewareConfig


class SimpleFunctionGroupConfig(FunctionGroupBaseConfig, name="simple_function_group"):
    """Simple function group config for testing."""
    pass


@pytest.fixture
def base_config():
    """Create a base Config with middleware, functions, function groups, and evaluators."""
    config = Config()
    # TODO(kyriacos): Add a RedTeamingMiddleware implementation when implemented.
    config.middleware["cache_middleware"] = CacheMiddlewareConfig()

    func1_config = EmptyFunctionConfig()
    func1_config.middleware = ["cache_middleware"]
    config.functions["func1"] = func1_config

    func2_config = EmptyFunctionConfig()
    func2_config.middleware = ["cache_middleware"]
    config.functions["func2"] = func2_config

    group1_config = SimpleFunctionGroupConfig()
    group1_config.middleware = ["cache_middleware"]
    config.function_groups["group1"] = group1_config

    eval_config = EvalConfig()
    red_team_eval_config = RedTeamingEvaluatorConfig(
        llm_name=LLMRef("test_llm"),
        judge_llm_prompt="Test prompt",
        filter_conditions=[],
        scenario_specific_instructions="Original instructions",
    )
    eval_config.evaluators["red_team_eval"] = red_team_eval_config
    config.eval = eval_config

    return config


@pytest.fixture
def base_config_with_filter_conditions(base_config):
    """Extend base_config with filter conditions."""
    config = copy.deepcopy(base_config)
    filter_condition = IntermediateStepsFilterCondition(name="original_filter", event_type="TOOL_END")
    config.eval.evaluators["red_team_eval"].filter_conditions = [filter_condition]
    return config


def test_valid_scenario_with_middleware():
    """Test valid scenario with middleware."""
    scenario = RedTeamScenarioEntry(
        scenario_id="test_scenario",
        middleware_name="cache_middleware",
        target_function_or_group="func1",
    )
    assert scenario.scenario_id == "test_scenario"
    assert scenario.middleware_name == "cache_middleware"


def test_valid_baseline_scenario():
    """Test valid baseline scenario."""
    scenario = RedTeamScenarioEntry(scenario_id="baseline", middleware_name=None, target_function_or_group=None)
    assert scenario.middleware_name is None


def test_invalid_baseline_with_target():
    """Test baseline scenario with target raises error."""
    with pytest.raises(ValueError, match="When middleware_name is null"):
        RedTeamScenarioEntry(scenario_id="invalid", middleware_name=None, target_function_or_group="func1")


def test_invalid_middleware_without_target():
    """Test middleware without target raises error."""
    with pytest.raises(ValueError, match="target_function_or_group must be specified"):
        RedTeamScenarioEntry(scenario_id="invalid", middleware_name="cache_middleware", target_function_or_group=None)


def test_middleware_removed_from_functions_and_placed_on_target_function(base_config):
    """Test middleware removed from all functions and placed only on target function."""
    scenario = RedTeamScenarioEntry(
        scenario_id="test", middleware_name="cache_middleware", target_function_or_group="func1"
    )
    modified_config = scenario.apply_to_config(copy.deepcopy(base_config))

    assert "cache_middleware" not in modified_config.functions["func2"].middleware
    assert "cache_middleware" not in modified_config.function_groups["group1"].middleware
    assert "cache_middleware" in modified_config.functions["func1"].middleware


def test_middleware_removed_from_functions_and_placed_on_target_function_group(base_config):
    """Test middleware removed from all functions/groups and placed only on target function group."""
    scenario = RedTeamScenarioEntry(
        scenario_id="test", middleware_name="cache_middleware", target_function_or_group="group1"
    )
    modified_config = scenario.apply_to_config(copy.deepcopy(base_config))

    assert "cache_middleware" not in modified_config.functions["func1"].middleware
    assert "cache_middleware" not in modified_config.functions["func2"].middleware
    assert "cache_middleware" in modified_config.function_groups["group1"].middleware


def test_evaluation_instructions_replaced(base_config):
    """Test scenario-specific instructions replace base configuration instructions."""
    scenario = RedTeamScenarioEntry(
        scenario_id="test",
        middleware_name="cache_middleware",
        target_function_or_group="func1",
        evaluation_instructions="New instructions",
    )
    modified_config = scenario.apply_to_config(copy.deepcopy(base_config))
    evaluator_config = cast(RedTeamingEvaluatorConfig, modified_config.eval.evaluators["red_team_eval"])
    assert evaluator_config.scenario_specific_instructions == "New instructions"


def test_filter_conditions_replaced(base_config_with_filter_conditions):
    """Test scenario filter conditions replace base configuration filter conditions."""
    new_filter = IntermediateStepsFilterCondition(name="new_filter", event_type="LLM_END")
    scenario = RedTeamScenarioEntry(
        scenario_id="test",
        middleware_name="cache_middleware",
        target_function_or_group="func1",
        filter_conditions=[new_filter],
    )
    modified_config = scenario.apply_to_config(copy.deepcopy(base_config_with_filter_conditions))
    evaluator_config = cast(RedTeamingEvaluatorConfig, modified_config.eval.evaluators["red_team_eval"])
    assert evaluator_config.filter_conditions is not None
    assert len(evaluator_config.filter_conditions) == 1
    assert evaluator_config.filter_conditions[0].name == "new_filter"

