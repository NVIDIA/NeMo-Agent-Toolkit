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

import pytest

from aiq.eval.evaluator.base_evaluator import BaseEvaluator
from aiq.eval.evaluator.evaluator_model import EvalInput
from aiq.eval.evaluator.evaluator_model import EvalInputItem
from aiq.eval.evaluator.evaluator_model import EvalOutputItem


class MockSimilarityEvaluator(BaseEvaluator):
    """Mock evaluator subclass to simulate similarity evaluation logic."""

    def __init__(self):
        super().__init__(max_concurrency=2, tqdm_desc="Mock Evaluator")

    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        # Fakescore based on input length for determinism
        score = round(len(item.output_obj) / max(len(item.expected_output_obj), 1), 2)
        reasoning = {
            "input": item.input_obj,
            "expected": item.expected_output_obj,
            "generated": item.output_obj,
            "similarity_score": score
        }
        return EvalOutputItem(id=item.id, score=score, reasoning=reasoning)


class FailingEvaluator(BaseEvaluator):

    def __init__(self):
        super().__init__(max_concurrency=2, tqdm_desc="Failing Evaluator")

    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        raise RuntimeError(f"Intentional failure for item {item.id}")


@pytest.fixture
def mock_input_items():
    return EvalInput(eval_input_items=[
        EvalInputItem(
            id="1",
            input_obj="Q1",
            expected_output_obj="This is the expected answer.",
            output_obj="This is the output.",
            trajectory=[],
            expected_trajectory=[],
            full_dataset_entry={
                "question": "Q1", "expected_answer": "This is the expected answer.", "output": "This is the output."
            }),
        EvalInputItem(id="2",
                      input_obj="Q2",
                      expected_output_obj="Short",
                      output_obj="Shorter",
                      trajectory=[],
                      expected_trajectory=[],
                      full_dataset_entry={
                          "question": "Q2", "expected_answer": "Short", "output": "Shorter"
                      })
    ])


async def test_similarity_evaluator_returns_valid_scores(mock_input_items):
    evaluator = MockSimilarityEvaluator()
    output = await evaluator.evaluate(mock_input_items)

    assert len(output.eval_output_items) == 2
    for item in output.eval_output_items:
        assert isinstance(item, EvalOutputItem)
        assert 0.0 <= item.score <= 2.0  # depending on string length ratio
        assert isinstance(item.reasoning, dict)
        assert "similarity_score" in item.reasoning

    assert output.average_score is not None
    assert isinstance(output.average_score, float)


async def test_similarity_evaluator_handles_empty_input():
    evaluator = MockSimilarityEvaluator()
    empty_input = EvalInput(eval_input_items=[])

    output = await evaluator.evaluate(empty_input)
    assert output.eval_output_items == []
    assert output.average_score is None


async def test_evaluator_handles_item_failure(mock_input_items):
    """Ensure BaseEvaluator returns EvalOutputItem with error info when evaluate_item fails."""
    # Use only the first item from the fixture
    single_item_input = mock_input_items.model_copy()
    single_item_input.eval_input_items = [mock_input_items.eval_input_items[0]]

    evaluator = FailingEvaluator()
    output = await evaluator.evaluate(single_item_input)

    assert len(output.eval_output_items) == 1

    failed_item = output.eval_output_items[0]
    assert isinstance(failed_item, EvalOutputItem)
    assert failed_item.score == 0.0
    assert isinstance(failed_item.reasoning, dict)
    assert "Evaluator error" in failed_item.reasoning["error"]
    assert "Intentional failure" in failed_item.reasoning["error"]
    assert output.average_score == 0.0
