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
"""Shared test fixtures for LangSmith/openevals evaluator tests."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from nat.eval.evaluator.evaluator_model import EvalInput
from nat.eval.evaluator.evaluator_model import EvalInputItem


def make_mock_builder(mock_llm=None):
    """Create a mock EvalBuilder with configurable get_llm.

    Args:
        mock_llm: Optional mock LLM to return from ``get_llm``.
            When ``None``, a default ``MagicMock`` is used.
    """
    builder = MagicMock(spec=["get_llm", "get_max_concurrency"])
    builder.get_llm = AsyncMock(return_value=mock_llm or MagicMock(name="mock_judge_llm"))
    builder.get_max_concurrency.return_value = 2
    return builder


@pytest.fixture
def eval_input_matching():
    """EvalInput where output matches expected output (for exact_match = True)."""
    return EvalInput(eval_input_items=[
        EvalInputItem(
            id="match_1",
            input_obj="What is 2 + 2?",
            expected_output_obj="4",
            output_obj="4",
            trajectory=[],
            expected_trajectory=[],
            full_dataset_entry={
                "question": "What is 2 + 2?",
                "expected_answer": "4",
                "output": "4",
            },
        ),
    ])


@pytest.fixture
def eval_input_non_matching():
    """EvalInput where output does NOT match expected output."""
    return EvalInput(eval_input_items=[
        EvalInputItem(
            id="mismatch_1",
            input_obj="What is 2 + 2?",
            expected_output_obj="4",
            output_obj="5",
            trajectory=[],
            expected_trajectory=[],
            full_dataset_entry={
                "question": "What is 2 + 2?",
                "expected_answer": "4",
                "output": "5",
            },
        ),
    ])


@pytest.fixture
def eval_input_multi_item():
    """EvalInput with multiple items (mix of matching and non-matching)."""
    return EvalInput(eval_input_items=[
        EvalInputItem(
            id="multi_1",
            input_obj="Capital of France?",
            expected_output_obj="Paris",
            output_obj="Paris",
            trajectory=[],
            expected_trajectory=[],
            full_dataset_entry={},
        ),
        EvalInputItem(
            id="multi_2",
            input_obj="Capital of Germany?",
            expected_output_obj="Berlin",
            output_obj="Munich",
            trajectory=[],
            expected_trajectory=[],
            full_dataset_entry={},
        ),
        EvalInputItem(
            id="multi_3",
            input_obj="Capital of Japan?",
            expected_output_obj="Tokyo",
            output_obj="Tokyo",
            trajectory=[],
            expected_trajectory=[],
            full_dataset_entry={},
        ),
    ])
