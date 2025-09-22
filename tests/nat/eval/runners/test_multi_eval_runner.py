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
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from nat.eval.config import EvaluationRunConfig
from nat.eval.evaluator.evaluator_model import EvalInput
from nat.eval.evaluator.evaluator_model import EvalInputItem
from nat.eval.evaluator.evaluator_model import EvalOutput
from nat.eval.evaluator.evaluator_model import EvalOutputItem
from nat.eval.runners.config import MultiEvaluationRunConfig
from nat.eval.runners.multi_eval_runner import MultiEvaluationRunner
from nat.profiler.data_models import ProfilerResults


@pytest.fixture
def base_eval_run_config():
    """Fixture for base evaluation run configuration."""
    return EvaluationRunConfig(config_file=Path("config.yml"),
                               endpoint=None,
                               endpoint_timeout=300,
                               adjust_dataset_size=True,
                               num_passes=1)


@pytest.fixture
def multi_eval_config(base_eval_run_config):
    """Fixture for multi-evaluation run configuration."""
    configs = {}
    for i, concurrency in enumerate([1, 2, 4]):
        config = copy.deepcopy(base_eval_run_config)
        config.override = (("eval.general.max_concurrency", str(concurrency)), )
        configs[f"concurrency_{concurrency}"] = config

    return MultiEvaluationRunConfig(configs=configs)


@pytest.fixture
def mock_evaluation_run_output():
    """Fixture for mock evaluation run output."""
    from nat.eval.config import EvaluationRunOutput

    # Create simple mock objects for testing
    eval_item = EvalInputItem(id=1,
                              input_obj="Test input",
                              expected_output_obj="Expected output",
                              output_obj="Generated output",
                              expected_trajectory=[],
                              trajectory=[],
                              full_dataset_entry={
                                  "id": 1, "question": "Test input", "answer": "Expected output"
                              })
    eval_input = EvalInput(eval_input_items=[eval_item])

    eval_output = EvalOutput(average_score=0.9,
                             eval_output_items=[EvalOutputItem(id=1, score=0.9, reasoning="Test evaluation")])

    return EvaluationRunOutput(workflow_output_file=Path("workflow_output.json"),
                               evaluator_output_files=[Path("evaluator_output.json")],
                               workflow_interrupted=False,
                               eval_input=eval_input,
                               evaluation_results=[("MockEvaluator", eval_output)],
                               usage_stats=None,
                               profiler_results=ProfilerResults())


async def test_run_all_with_overrides(base_eval_run_config, mock_evaluation_run_output):
    """Test run_all with overrides."""
    configs = {}

    # Create config with multiple overrides
    config1 = copy.deepcopy(base_eval_run_config)
    config1.override = (("eval.general.max_concurrency", "1"), ("eval.general.output_dir", "./.tmp/test1"))
    configs["complex_1"] = config1

    # Create config with different overrides
    config2 = copy.deepcopy(base_eval_run_config)
    config2.override = (("eval.general.max_concurrency", "2"), ("eval.general.workflow_alias", "alias_complex_2"))
    configs["complex_2"] = config2

    config = MultiEvaluationRunConfig(configs=configs)
    runner = MultiEvaluationRunner(config)

    with patch.object(runner, "run_single_evaluation", new_callable=AsyncMock) as mock_run_single:
        mock_run_single.return_value = mock_evaluation_run_output

        result = await runner.run_all()

        # Verify both complex configs were processed
        assert mock_run_single.call_count == 2

        # Verify the calls were made with correct configs
        expected_keys = ["complex_1", "complex_2"]
        actual_keys = [call[0][0] for call in mock_run_single.call_args_list]
        assert set(actual_keys) == set(expected_keys)

        # Verify results were stored and returned
        assert len(runner.evaluation_run_outputs) == 2
        assert result == runner.evaluation_run_outputs


async def test_run_all_partial_failure(multi_eval_config, mock_evaluation_run_output):
    """Test run_all with partial failures."""
    runner = MultiEvaluationRunner(multi_eval_config)

    with patch.object(runner, "run_single_evaluation", new_callable=AsyncMock) as mock_run_single:
        # First call succeeds, second fails, third succeeds
        mock_run_single.side_effect = [
            mock_evaluation_run_output, Exception("Second evaluation failed"), mock_evaluation_run_output
        ]

        with pytest.raises(Exception, match="Second evaluation failed"):
            await runner.run_all()

        # Verify only the first result was stored before the exception
        assert len(runner.evaluation_run_outputs) == 1
        assert "concurrency_1" in runner.evaluation_run_outputs
