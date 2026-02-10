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
"""Shared conversion utilities for LangSmith/openevals evaluator integration.

Provides functions to convert between NAT evaluation data models and
LangSmith/openevals data models.
"""

import uuid
from datetime import UTC
from datetime import datetime
from typing import Any

from langsmith.evaluation.evaluator import EvaluationResult

from nat.eval.evaluator.evaluator_model import EvalInputItem
from nat.eval.evaluator.evaluator_model import EvalOutputItem


def eval_input_item_to_openevals_kwargs(item: EvalInputItem) -> dict[str, Any]:
    """Convert a NAT EvalInputItem to openevals keyword arguments.

    Maps NAT evaluation data to the (inputs, outputs, reference_outputs)
    convention used by openevals evaluators.

    Args:
        item: NAT evaluation input item.

    Returns:
        Dictionary with 'inputs', 'outputs', and 'reference_outputs' keys.
    """
    return {
        "inputs": item.input_obj,
        "outputs": item.output_obj,
        "reference_outputs": item.expected_output_obj,
    }


def eval_input_item_to_run_and_example(item: EvalInputItem) -> tuple[Any, Any]:
    """Convert a NAT EvalInputItem to synthetic LangSmith Run and Example objects.

    Creates minimal Run and Example instances with the data that most
    LangSmith evaluators need (inputs, outputs, expected outputs).

    Args:
        item: NAT evaluation input item.

    Returns:
        Tuple of (Run, Example) instances.
    """
    from langsmith.schemas import Example
    from langsmith.schemas import Run

    run = Run(
        id=uuid.uuid4(),
        name="nat_eval_run",
        start_time=datetime.now(UTC),
        end_time=datetime.now(UTC),
        run_type="chain",
        inputs={"input": item.input_obj},
        outputs={"output": item.output_obj},
        trace_id=uuid.uuid4(),
    )

    example = Example(
        id=uuid.uuid4(),
        inputs={"input": item.input_obj},
        outputs={"output": item.expected_output_obj},
        dataset_id=uuid.uuid4(),
        created_at=datetime.now(UTC),
    )

    return run, example


def langsmith_result_to_eval_output_item(
    item_id: Any,
    result: dict | Any,
) -> EvalOutputItem:
    """Convert a LangSmith/openevals evaluation result to a NAT EvalOutputItem.

    Handles both dict results (from openevals/function evaluators) and
    EvaluationResult objects (from RunEvaluator classes).

    Args:
        item_id: The id from the corresponding EvalInputItem.
        result: The evaluation result -- either a dict with 'score'/'key'/'comment'
            fields, an EvaluationResult, or an EvaluationResults containing
            multiple results.

    Returns:
        NAT EvalOutputItem with score and reasoning.
    """
    # Handle EvaluationResults (batch of results) -- take the first one.
    # EvaluationResults is a TypedDict so isinstance() is not available;
    # check for a dict with a "results" key instead.
    if isinstance(result, dict) and "results" in result:
        results_list = result["results"]
        if results_list:
            result = results_list[0]
        else:
            return EvalOutputItem(
                id=item_id,
                score=0.0,
                reasoning={"error": "Empty EvaluationResults returned"},
            )

    # Handle EvaluationResult objects (from RunEvaluator classes)
    if isinstance(result, EvaluationResult):
        score = result.score if result.score is not None else result.value
        reasoning = {
            "key": result.key,
            "comment": result.comment,
        }
        if result.metadata:
            reasoning["metadata"] = result.metadata
        return EvalOutputItem(id=item_id, score=score, reasoning=reasoning)

    # Handle plain dict results (from openevals and function evaluators)
    if isinstance(result, dict):
        score = result.get("score")
        reasoning = {
            "key": result.get("key", "unknown"),
            "comment": result.get("comment"),
        }
        if result.get("metadata"):
            reasoning["metadata"] = result["metadata"]
        return EvalOutputItem(id=item_id, score=score, reasoning=reasoning)

    # Fallback for unexpected result types
    return EvalOutputItem(
        id=item_id,
        score=0.0,
        reasoning={
            "error": f"Unexpected result type: {type(result).__name__}", "raw": str(result)
        },
    )
