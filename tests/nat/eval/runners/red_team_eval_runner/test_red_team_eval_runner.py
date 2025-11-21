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

import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.data_models.component_ref import LLMRef
from nat.data_models.config import Config
from nat.data_models.evaluate import EvalConfig
from nat.data_models.function import EmptyFunctionConfig
from nat.eval.config import EvaluationRunConfig
from nat.eval.red_teaming_evaluator.register import RedTeamingEvaluatorConfig
from nat.eval.runners.red_team_eval_runner.red_team_eval_config import RedTeamingEvaluationConfig
from nat.eval.runners.red_team_eval_runner.red_team_eval_runner import RedTeamingEvaluationRunner
from nat.middleware.cache_middleware import CacheMiddlewareConfig


@pytest.fixture
def valid_config():
    """Create a valid base Config with middleware and red_teaming_evaluator."""
    config = Config()
    # TODO(kyriacos): Add a RedTeamingMiddleware implementation when implemented.
    config.middleware["cache_middleware"] = CacheMiddlewareConfig()

    func1_config = EmptyFunctionConfig()
    func1_config.middleware = ["cache_middleware"]
    config.functions["func1"] = func1_config

    eval_config = EvalConfig()
    red_team_eval_config = RedTeamingEvaluatorConfig(
        llm_name=LLMRef("test_llm"),
        judge_llm_prompt="Test prompt",
        filter_conditions=[],
        scenario_specific_instructions="Test instructions",
    )
    eval_config.evaluators["red_team_eval"] = red_team_eval_config
    config.eval = eval_config

    return config


@pytest.fixture
def scenarios_data():
    """Sample scenarios data."""
    return [
        {
            "scenario_id": "baseline",
            "middleware_name": None,
            "target_function_or_group": None,
        },
        {
            "scenario_id": "test_scenario",
            "middleware_name": "cache_middleware",
            "target_function_or_group": "func1",
        },
    ]


@pytest.fixture
def mock_scenarios_file(tmp_path, scenarios_data):
    """Create a temporary scenarios file."""
    scenarios_file = tmp_path / "scenarios.json"
    scenarios_file.write_text(json.dumps(scenarios_data))
    return scenarios_file


@pytest.fixture
def eval_run_config(mock_scenarios_file, valid_config):
    """Create an EvaluationRunConfig."""
    return EvaluationRunConfig(
        config_file=valid_config,
    )


@pytest.fixture
def red_team_config(eval_run_config, mock_scenarios_file):
    """Create a RedTeamingEvaluationConfig."""
    return RedTeamingEvaluationConfig(
        base_evaluation_config=eval_run_config,
        red_team_scenarios_file=mock_scenarios_file,
    )


def test_init_loads_scenarios(red_team_config):
    """Test that initialization loads scenarios correctly."""
    runner = RedTeamingEvaluationRunner(red_team_config)
    assert len(runner._scenarios) == 2
    assert runner._scenarios[0].scenario_id == "baseline"
    assert runner._scenarios[1].scenario_id == "test_scenario"


def test_init_validates_base_config(eval_run_config, mock_scenarios_file):
    """Test that initialization validates the base config."""
    # Config without middleware should fail
    invalid_config = Config()
    eval_run_config.config_file = invalid_config

    config = RedTeamingEvaluationConfig(
        base_evaluation_config=eval_run_config,
        red_team_scenarios_file=mock_scenarios_file,
    )

    with pytest.raises(ValueError, match="must contain at least one middleware"):
        RedTeamingEvaluationRunner(config)


def test_init_requires_red_teaming_evaluator(eval_run_config, mock_scenarios_file, valid_config):
    """Test that initialization requires a red_teaming_evaluator."""
    # Remove the red_teaming_evaluator
    valid_config.eval.evaluators = {}
    eval_run_config.config_file = valid_config

    config = RedTeamingEvaluationConfig(
        base_evaluation_config=eval_run_config,
        red_team_scenarios_file=mock_scenarios_file,
    )

    with pytest.raises(ValueError, match="must contain at least one evaluator"):
        RedTeamingEvaluationRunner(config)


def test_load_scenarios_invalid_json(eval_run_config, tmp_path):
    """Test that loading invalid JSON raises an error."""
    scenarios_file = tmp_path / "invalid.json"
    scenarios_file.write_text("not a json array")

    config = RedTeamingEvaluationConfig(
        base_evaluation_config=eval_run_config,
        red_team_scenarios_file=scenarios_file,
    )

    with pytest.raises(ValueError):
        RedTeamingEvaluationRunner(config)


def test_load_scenarios_not_array(eval_run_config, tmp_path):
    """Test that non-array JSON raises an error."""
    scenarios_file = tmp_path / "not_array.json"
    scenarios_file.write_text(json.dumps({"not": "array"}))

    config = RedTeamingEvaluationConfig(
        base_evaluation_config=eval_run_config,
        red_team_scenarios_file=scenarios_file,
    )

    with pytest.raises(ValueError, match="must contain a JSON array"):
        RedTeamingEvaluationRunner(config)


def test_load_base_config_from_path(eval_run_config, tmp_path, valid_config, mock_scenarios_file):
    """Test loading config from a Path."""
    config_file = tmp_path / "config.yml"

    # Mock load_config to return valid_config (patch at import location)
    with patch("nat.runtime.loader.load_config", return_value=valid_config):
        eval_run_config.config_file = config_file
        config = RedTeamingEvaluationConfig(
            base_evaluation_config=eval_run_config,
            red_team_scenarios_file=mock_scenarios_file,
        )
        runner = RedTeamingEvaluationRunner(config)
        assert isinstance(runner.config.base_evaluation_config.config_file, Config)


@pytest.mark.asyncio
async def test_run_single_scenario(red_team_config):
    """Test running a single scenario."""
    runner = RedTeamingEvaluationRunner(red_team_config)
    scenario = runner._scenarios[0]

    # Mock EvaluationRun
    with patch("nat.eval.runners.red_team_eval_runner.red_team_eval_runner.EvaluationRun") as mock_eval_run:
        mock_output = MagicMock()
        mock_eval_run.return_value.run_and_evaluate = AsyncMock(return_value=mock_output)

        output = await runner.run_single_scenario(scenario)

        assert output == mock_output
        mock_eval_run.assert_called_once()
        mock_eval_run.return_value.run_and_evaluate.assert_called_once()


@pytest.mark.asyncio
async def test_run_all_scenarios(red_team_config):
    """Test running all scenarios."""
    runner = RedTeamingEvaluationRunner(red_team_config)

    # Mock EvaluationRun
    with patch("nat.eval.runners.red_team_eval_runner.red_team_eval_runner.EvaluationRun") as mock_eval_run:
        mock_output = MagicMock()
        mock_eval_run.return_value.run_and_evaluate = AsyncMock(return_value=mock_output)

        outputs = await runner.run_all()

        assert len(outputs) == 2
        assert "baseline" in outputs
        assert "test_scenario" in outputs
        assert mock_eval_run.return_value.run_and_evaluate.call_count == 2


def test_validate_base_config_no_middleware(valid_config):
    """Test validation fails without middleware."""
    valid_config.middleware = {}

    runner = RedTeamingEvaluationRunner.__new__(RedTeamingEvaluationRunner)
    with pytest.raises(ValueError, match="must contain at least one middleware"):
        runner._validate_base_config(valid_config)


def test_validate_base_config_wrong_evaluator_type(valid_config):
    """Test validation fails without red_teaming_evaluator."""
    # Create a different evaluator type (not red_teaming_evaluator)
    eval_config = EvalConfig()
    # Create a mock evaluator without the type attribute
    mock_evaluator = MagicMock()
    mock_evaluator.type = "some_other_evaluator"
    eval_config.evaluators["other_eval"] = mock_evaluator
    valid_config.eval = eval_config

    runner = RedTeamingEvaluationRunner.__new__(RedTeamingEvaluationRunner)
    with pytest.raises(ValueError, match="must contain at least one evaluator of type 'red_teaming_evaluator'"):
        runner._validate_base_config(valid_config)


@pytest.mark.asyncio
async def test_run_single_scenario_applies_scenario_config(red_team_config):
    """Test that running a scenario applies the scenario configuration."""
    runner = RedTeamingEvaluationRunner(red_team_config)
    scenario = runner._scenarios[1]  # test_scenario with middleware

    with patch("nat.eval.runners.red_team_eval_runner.red_team_eval_runner.EvaluationRun") as mock_eval_run:
        mock_output = MagicMock()
        mock_eval_run.return_value.run_and_evaluate = AsyncMock(return_value=mock_output)

        await runner.run_single_scenario(scenario)

        # Verify that EvaluationRun was called with a modified config
        # Access keyword arguments using call_args.kwargs
        called_config = mock_eval_run.call_args.kwargs["config"].config_file
        assert isinstance(called_config, Config)
        # Verify middleware is applied to func1
        assert "cache_middleware" in called_config.functions["func1"].middleware

