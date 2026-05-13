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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from nat.data_models.evaluator import EvalInput
from nat.data_models.evaluator import EvalInputItem


def make_mock_builder(mock_llm=None):
    """Create a mock EvalBuilder with a configurable get_llm.

    Args:
        mock_llm: Optional mock LLM to return from ``get_llm``.
            When ``None``, a default ``MagicMock`` is used.
    """
    builder = MagicMock(spec=["get_llm", "get_max_concurrency"])
    builder.get_llm = AsyncMock(return_value=mock_llm or MagicMock(name="mock_judge_llm"))
    builder.get_max_concurrency.return_value = 2
    return builder


async def register_evaluator_ctx(register_fn, config, builder=None):
    """Drive the async context manager returned by a ``@register_evaluator`` function.

    Convenience helper that enters the async context manager and returns
    the yielded ``EvaluatorInfo``.

    Args:
        register_fn: The decorated registration function (e.g.,
            ``register_langsmith_evaluator``, ``register_langsmith_judge``).
        config: The evaluator config to pass.
        builder: An ``EvalBuilder`` (or mock).  When ``None``, a default
            mock builder is created via :func:`make_mock_builder`.
    """
    if builder is None:
        builder = make_mock_builder()
    async with register_fn(config, builder) as info:
        return info


@pytest.fixture(name="eval_input_matching")
def fixture_eval_input_matching():
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


@pytest.fixture(name="eval_input_non_matching")
def fixture_eval_input_non_matching():
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


@pytest.fixture(name="eval_input_multi_item")
def fixture_eval_input_multi_item():
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


@pytest.fixture(name="item_with_context")
def fixture_item_with_context():
    """EvalInputItem whose full_dataset_entry has a 'retrieved_context' field."""
    return EvalInputItem(
        id="ctx_1",
        input_obj="What is a doodad?",
        expected_output_obj="A small gadget",
        output_obj="A doodad is a kitten",
        trajectory=[],
        expected_trajectory=[],
        full_dataset_entry={
            "question": "What is a doodad?",
            "answer": "A small gadget",
            "retrieved_context": "Doodads are small mechanical gadgets used in workshops.",
            "agent_plan": "Step 1: look it up. Step 2: summarize.",
        },
    )


@pytest.fixture(name="eval_input_with_context")
def fixture_eval_input_with_context(item_with_context):
    """EvalInput wrapping a single item with context fields."""
    return EvalInput(eval_input_items=[item_with_context])
