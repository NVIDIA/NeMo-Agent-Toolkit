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

import importlib
import uuid
from datetime import UTC
from datetime import datetime
from typing import Any

from langsmith.evaluation.evaluator import EvaluationResult

from nat.data_models.evaluator import EvalInputItem
from nat.plugins.eval.data_models.evaluator_io import EvalOutputItem

_MISSING = object()


def _import_from_dotted_path(dotted_path: str, *, label: str = "object") -> Any:
    """Import an attribute from a Python dotted path.

    Resolves ``'module.path.attribute'`` into the corresponding Python object
    but does **not** instantiate classes.  Used by
    ``langsmith_custom_evaluator._import_evaluator`` and
    ``langsmith_judge._build_create_kwargs`` (for ``output_schema``).

    Args:
        dotted_path: Full Python dotted path (e.g., ``'my_pkg.module.MyClass'``).
        label: Human-readable label for error messages (e.g., ``'evaluator'``, ``'output_schema'``).

    Returns:
        The imported attribute.

    Raises:
        ValueError: If the path does not contain a module/attribute separator.
        ImportError: If the module cannot be imported.
        AttributeError: If the attribute cannot be found in the module.
    """
    parts = dotted_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid {label} path '{dotted_path}'. Expected format: 'module.attribute'")

    module_path, attr_name = parts

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_path}' for {label} '{dotted_path}'. "
                          f"Make sure the package is installed and the path is correct.") from e

    obj = getattr(module, attr_name, _MISSING)
    if obj is _MISSING:
        raise AttributeError(f"Module '{module_path}' has no attribute '{attr_name}'. "
                             f"Available attributes: {[a for a in dir(module) if not a.startswith('_')]}")

    return obj


def eval_input_item_to_openevals_kwargs(
    item: EvalInputItem,
    extra_fields: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Convert a NAT EvalInputItem to openevals keyword arguments.

    Maps NAT evaluation data to the (inputs, outputs, reference_outputs)
    convention used by openevals evaluators.  When *extra_fields* is
    provided, additional values are pulled from ``item.full_dataset_entry``
    and included as extra keyword arguments (e.g., ``context``, ``plan``).

    Args:
        item: NAT evaluation input item.
        extra_fields: Mapping of kwarg names to dataset field names, looked up in ``item.full_dataset_entry``.

    Returns:
        Dictionary with at least ``inputs``, ``outputs``, and ``reference_outputs`` keys, plus any extra fields.

    Raises:
        ValueError: If an extra_fields key conflicts with ``inputs``, ``outputs``, or ``reference_outputs``.
        KeyError: If a requested extra field is not present in the dataset entry.
    """
    kwargs: dict[str, Any] = {
        "inputs": item.input_obj,
        "outputs": item.output_obj,
        "reference_outputs": item.expected_output_obj,
    }

    if extra_fields:
        dataset_entry = item.full_dataset_entry if isinstance(item.full_dataset_entry, dict) else {}
        for kwarg_name, dataset_key in extra_fields.items():
            if kwarg_name in ("inputs", "outputs", "reference_outputs"):
                raise ValueError(f"extra_fields key '{kwarg_name}' conflicts with a standard evaluator "
                                 f"parameter.  Use a different kwarg name.")
            if dataset_key not in dataset_entry:
                raise KeyError(f"extra_fields maps '{kwarg_name}' to dataset field '{dataset_key}', "
                               f"but '{dataset_key}' was not found in the dataset entry.  "
                               f"Available keys: {sorted(dataset_entry.keys())}")
            kwargs[kwarg_name] = dataset_entry[dataset_key]

    return kwargs


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


def _extract_field(data: dict, field_path: str) -> Any:
    """Extract a value from a nested dict using dot-notation.

    Args:
        data: The dictionary to extract from.
        field_path: Dot-separated path (e.g., ``'analysis.score'``).

    Returns:
        The extracted value.

    Raises:
        KeyError: If any segment of the path is missing.
        TypeError: If an intermediate value is not a dict.
    """
    current: Any = data
    for part in field_path.split("."):
        if not isinstance(current, dict):
            raise TypeError(f"Cannot traverse into non-dict value at '{part}' "
                            f"in field path '{field_path}'. Got {type(current).__name__}.")
        if part not in current:
            raise KeyError(f"Field '{part}' not found in result while resolving "
                           f"score_field '{field_path}'.  Available keys: {sorted(current.keys())}")
        current = current[part]
    return current


def _handle_custom_schema_result(
    item_id: Any,
    result: dict,
    score_field: str,
) -> EvalOutputItem:
    """Handle a raw dict from a custom ``output_schema`` evaluator.

    The score is extracted using :func:`_extract_field` with dot-notation.
    """
    try:
        score = _extract_field(result, score_field)
    except (KeyError, TypeError) as exc:
        error_msg = f"Failed to extract score_field '{score_field}': {exc}"
        return EvalOutputItem(
            id=item_id,
            score=0.0,
            reasoning={"raw": str(result)},
            error=error_msg,
        )
    return EvalOutputItem(
        id=item_id,
        score=score,
        reasoning={"raw_output": result},
    )


def _handle_list_result(item_id: Any, result: list) -> EvalOutputItem:
    """Handle a bare list of results (e.g., from ``create_json_match_evaluator``).

    Scores are averaged; per-item details are preserved in reasoning.
    """
    if not result:
        return EvalOutputItem(
            id=item_id,
            score=0.0,
            reasoning={},
            error="Empty list of results returned",
        )

    scores: list[float] = []
    per_item: list[dict] = []
    for i, item_result in enumerate(result):
        converted = langsmith_result_to_eval_output_item(f"{item_id}_sub_{i}", item_result)
        if converted.score is not None:
            numeric = (float(converted.score) if not isinstance(converted.score, bool) else
                       (1.0 if converted.score else 0.0))
            scores.append(numeric)
        per_item.append({
            "id": converted.id,
            "score": converted.score,
            "reasoning": converted.reasoning,
        })

    avg_score = sum(scores) / len(scores) if scores else 0.0
    return EvalOutputItem(
        id=item_id,
        score=avg_score,
        reasoning={
            "aggregated_from": len(result), "per_item": per_item
        },
    )


def _handle_evaluation_result(item_id: Any, result: EvaluationResult) -> EvalOutputItem:
    """Handle an ``EvaluationResult`` object (from RunEvaluator classes)."""
    score = result.score if result.score is not None else result.value
    reasoning: dict[str, Any] = {
        "key": result.key,
        "comment": result.comment,
    }
    if result.metadata:
        reasoning["metadata"] = result.metadata
    return EvalOutputItem(id=item_id, score=score, reasoning=reasoning)


def _handle_dict_result(item_id: Any, result: dict) -> EvalOutputItem:
    """Handle a plain dict result (from openevals / function evaluators)."""
    score = result.get("score")
    reasoning: dict[str, Any] = {
        "key": result.get("key", "unknown"),
        "comment": result.get("comment"),
    }
    if result.get("metadata"):
        reasoning["metadata"] = result["metadata"]
    return EvalOutputItem(id=item_id, score=score, reasoning=reasoning)


def langsmith_result_to_eval_output_item(
    item_id: Any,
    result: dict | list | Any,
    score_field: str | None = None,
) -> EvalOutputItem:
    """Convert a LangSmith/openevals evaluation result to a NAT EvalOutputItem.

    Dispatches to specialised handlers based on the result type:

    - Custom ``output_schema`` dict (when *score_field* is set)
    - Bare list (e.g., ``create_json_match_evaluator``)
    - ``EvaluationResults`` batch (dict with ``"results"`` key)
    - ``EvaluationResult`` object (from RunEvaluator classes)
    - Plain dict (from openevals / function evaluators)
    - Fallback for unexpected types

    Args:
        item_id: The id from the corresponding EvalInputItem.
        result: The evaluation result.
        score_field: Dot-notation path to the score in custom ``output_schema`` results (e.g., ``'analysis.score'``).

    Returns:
        NAT EvalOutputItem with score and reasoning.
    """
    # Custom output_schema path
    if score_field is not None and isinstance(result, dict):
        return _handle_custom_schema_result(item_id, result, score_field)

    # Bare list
    if isinstance(result, list):
        return _handle_list_result(item_id, result)

    # EvaluationResults batch -- unwrap then fall through
    if isinstance(result, dict) and "results" in result:
        results_list = result["results"]
        if results_list:
            result = results_list[0]
        else:
            return EvalOutputItem(
                id=item_id,
                score=0.0,
                reasoning={},
                error="Empty EvaluationResults returned",
            )

    # EvaluationResult object
    if isinstance(result, EvaluationResult):
        return _handle_evaluation_result(item_id, result)

    # Plain dict
    if isinstance(result, dict):
        return _handle_dict_result(item_id, result)

    # Fallback for unexpected result types
    error_msg = f"Unexpected result type: {type(result).__name__}"
    return EvalOutputItem(
        id=item_id,
        score=0.0,
        reasoning={"raw": str(result)},
        error=error_msg,
    )
