# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest.mock import AsyncMock
from unittest.mock import patch

from nat.data_models.evaluator import EvalOutput
from nat.data_models.evaluator import EvalOutputItem
from nat.plugins.eval.runtime.eval_harness import EvaluationHarness


async def test_evaluate_returns_per_evaluator_outputs():
    """Harness returns per-evaluator outputs for successful evaluators."""
    harness = EvaluationHarness()
    samples = [object()]

    output_a = EvalOutput(average_score=1.0, eval_output_items=[EvalOutputItem(id=1, score=1.0, reasoning={})])
    output_b = EvalOutput(average_score=0.5, eval_output_items=[EvalOutputItem(id=1, score=0.5, reasoning={})])

    evaluator_a = AsyncMock()
    evaluator_a.evaluate_atif_fn = AsyncMock(return_value=output_a)
    evaluator_b = AsyncMock()
    evaluator_b.evaluate_atif_fn = AsyncMock(return_value=output_b)

    results = await harness.evaluate({"A": evaluator_a, "B": evaluator_b}, samples)

    assert list(results.keys()) == ["A", "B"]
    assert results["A"] == output_a
    assert results["B"] == output_b
    evaluator_a.evaluate_atif_fn.assert_awaited_once_with(samples)
    evaluator_b.evaluate_atif_fn.assert_awaited_once_with(samples)


async def test_evaluate_best_effort_when_one_evaluator_fails():
    """Harness continues and returns successful outputs when one evaluator fails."""
    harness = EvaluationHarness()
    samples = [object()]

    output = EvalOutput(average_score=0.7, eval_output_items=[EvalOutputItem(id=1, score=0.7, reasoning={})])
    good_evaluator = AsyncMock()
    good_evaluator.evaluate_atif_fn = AsyncMock(return_value=output)
    bad_evaluator = AsyncMock()
    bad_evaluator.evaluate_atif_fn = AsyncMock(side_effect=RuntimeError("boom"))

    with patch("nat.plugins.eval.runtime.eval_harness.logger.exception") as mock_log_exception:
        results = await harness.evaluate({"good": good_evaluator, "bad": bad_evaluator}, samples)

    assert results == {"good": output}
    mock_log_exception.assert_called_once()
    good_evaluator.evaluate_atif_fn.assert_awaited_once_with(samples)
    bad_evaluator.evaluate_atif_fn.assert_awaited_once_with(samples)


async def test_evaluate_skips_none_evaluator_entry():
    """Harness skips falsy evaluator entries."""
    harness = EvaluationHarness()
    samples = [object()]

    with patch("nat.plugins.eval.runtime.eval_harness.logger.warning") as mock_log_warning:
        results = await harness.evaluate({"missing": None}, samples)

    assert results == {}
    mock_log_warning.assert_not_called()
