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
from nat.middleware.red_teaming_middleware import RedTeamingMiddlewareConfig


class SimpleFunctionGroupConfig(FunctionGroupBaseConfig, name="simple_function_group"):
    """Simple function group config for testing."""
    pass


@pytest.fixture
def base_config():
    """Create a base Config with middleware, functions, function groups, and evaluators."""
    config = Config()
    # TODO(kyriacos): Add a RedTeamingMiddleware implementation when implemented.
    config.middleware["red_teaming_middleware"] = RedTeamingMiddlewareConfig(attack_payload="test",
                                                                             target_function_or_group="func1")

    func1_config = EmptyFunctionConfig()
    func1_config.middleware = ["red_teaming_middleware"]
    config.functions["func1"] = func1_config

    func2_config = EmptyFunctionConfig()
    func2_config.middleware = ["red_teaming_middleware"]
    config.functions["func2"] = func2_config

    group1_config = SimpleFunctionGroupConfig()
    group1_config.middleware = ["red_teaming_middleware"]
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


def test_middleware_removed_from_functions_and_placed_on_target_function(base_config):
    """Test middleware removed from all functions and placed only on target function."""
    scenario = RedTeamScenarioEntry(
        scenario_id="test", middleware_name="red_teaming_middleware"
    )
    modified_config = scenario.apply_to_config(copy.deepcopy(base_config))

    assert "red_teaming_middleware" not in modified_config.functions["func2"].middleware
    assert "red_teaming_middleware" not in modified_config.function_groups["group1"].middleware
    assert "red_teaming_middleware" in modified_config.functions["func1"].middleware


def test_middleware_removed_from_functions_and_placed_on_target_function_group(base_config):
    """Test middleware removed from all functions/groups and placed only on target function group."""
    config = copy.deepcopy(base_config)
    config.middleware["red_teaming_middleware"].target_function_or_group = "group1"
    scenario = RedTeamScenarioEntry(
        scenario_id="test", middleware_name="red_teaming_middleware",
    )
    modified_config = scenario.apply_to_config(config)

    assert "red_teaming_middleware" not in modified_config.functions["func1"].middleware
    assert "red_teaming_middleware" not in modified_config.functions["func2"].middleware
    assert "red_teaming_middleware" in modified_config.function_groups["group1"].middleware


def test_evaluation_instructions_replaced(base_config):
    """Test scenario-specific instructions replace base configuration instructions."""
    scenario = RedTeamScenarioEntry(
        scenario_id="test",
        middleware_name="red_teaming_middleware",
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
        middleware_name="red_teaming_middleware",
        filter_conditions=[new_filter],
    )
    modified_config = scenario.apply_to_config(copy.deepcopy(base_config_with_filter_conditions))
    evaluator_config = cast(RedTeamingEvaluatorConfig, modified_config.eval.evaluators["red_team_eval"])
    assert evaluator_config.filter_conditions is not None
    assert len(evaluator_config.filter_conditions) == 1
    assert evaluator_config.filter_conditions[0].name == "new_filter"


def test_full_scenario_applied(base_config_with_filter_conditions):
    """Test that a fully populated scenario with all fields is correctly applied to config."""
    # Create a comprehensive scenario with all fields populated
    new_filter = IntermediateStepsFilterCondition(name="scenario_filter", event_type="TOOL_START")
    scenario = RedTeamScenarioEntry(
        scenario_id="comprehensive_test",
        middleware_name="red_teaming_middleware",
        middleware_config_override={
            "attack_payload": "new_attack_payload",
            "target_function_or_group": "func2"
        },
        evaluation_instructions="Comprehensive scenario instructions",
        filter_conditions=[new_filter],
    )

    # Apply scenario to config
    modified_config = scenario.apply_to_config(copy.deepcopy(base_config_with_filter_conditions))

    # Verify middleware is removed from all functions/groups except the target
    assert "red_teaming_middleware" not in modified_config.functions["func1"].middleware
    assert "red_teaming_middleware" not in modified_config.function_groups["group1"].middleware

    # Verify middleware is on the target function (func2)
    assert "red_teaming_middleware" in modified_config.functions["func2"].middleware

    # Verify middleware config was updated with overrides
    middleware_config = cast(RedTeamingMiddlewareConfig, modified_config.middleware["red_teaming_middleware"])
    assert middleware_config.attack_payload == "new_attack_payload"
    assert middleware_config.target_function_or_group == "func2"

    # Verify evaluation instructions were updated
    evaluator_config = cast(RedTeamingEvaluatorConfig, modified_config.eval.evaluators["red_team_eval"])
    assert evaluator_config.scenario_specific_instructions == "Comprehensive scenario instructions"

    # Verify filter conditions were updated
    assert evaluator_config.filter_conditions is not None
    assert len(evaluator_config.filter_conditions) == 1
    assert evaluator_config.filter_conditions[0].name == "scenario_filter"
    assert evaluator_config.filter_conditions[0].event_type == "TOOL_START"


def test_error_when_no_target_function_or_group_specified(base_config):
    """Test that ValueError is raised when no target_function_or_group is specified."""
    # Create config with middleware that has no target specified
    config = copy.deepcopy(base_config)
    config.middleware["red_teaming_middleware"].target_function_or_group = None

    scenario = RedTeamScenarioEntry(
        scenario_id="test",
        middleware_name="red_teaming_middleware",
    )

    with pytest.raises(ValueError, match="No target function or group specified"):
        scenario.apply_to_config(config)


def test_error_when_middleware_not_found(base_config):
    """Test that ValueError is raised when specified middleware doesn't exist in config."""
    scenario = RedTeamScenarioEntry(
        scenario_id="test",
        middleware_name="nonexistent_middleware",
    )

    with pytest.raises(ValueError, match="Middleware 'nonexistent_middleware' not found in middleware"):
        scenario.apply_to_config(copy.deepcopy(base_config))


def test_error_when_target_not_found(base_config):
    """Test that ValueError is raised when target function/group doesn't exist."""
    scenario = RedTeamScenarioEntry(
        scenario_id="test",
        middleware_name="red_teaming_middleware",
        middleware_config_override={"target_function_or_group": "nonexistent_target"}
    )

    with pytest.raises(ValueError, match="Target 'nonexistent_target' not found in config"):
        scenario.apply_to_config(copy.deepcopy(base_config))


def test_error_when_middleware_config_override_has_invalid_key(base_config):
    """Test that KeyError is raised when middleware_config_override contains invalid key."""
    scenario = RedTeamScenarioEntry(
        scenario_id="test",
        middleware_name="red_teaming_middleware",
        middleware_config_override={
            "invalid_key": "some_value",
            "target_function_or_group": "func1"
        }
    )

    with pytest.raises(KeyError, match="middleware_config_override contains key"):
        scenario.apply_to_config(copy.deepcopy(base_config))

