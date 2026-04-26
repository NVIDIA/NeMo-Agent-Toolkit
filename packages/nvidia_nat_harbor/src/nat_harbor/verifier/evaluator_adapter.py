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
"""ATIF evaluator loading and execution for Harbor verifier bridge."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
from collections.abc import Awaitable
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nat.atif import ATIFTrajectory
from nat.plugins.eval.data_models.evaluator_io import EvalOutput
from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSample
from nat.runtime.loader import load_config


class BridgeEvaluatorError(RuntimeError):
    """Raised when evaluator bridge dispatch or normalization fails."""


def _load_callable(ref: str) -> Callable[..., Any]:
    if ":" not in ref:
        raise BridgeEvaluatorError(f"Invalid evaluator ref '{ref}'. Expected format 'module:function'.")
    module_name, attr_name = ref.split(":", 1)
    if not module_name or not attr_name:
        raise BridgeEvaluatorError(f"Invalid evaluator ref '{ref}'. Expected non-empty module and function names.")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise BridgeEvaluatorError(f"Failed to import evaluator module '{module_name}'.") from exc
    resolved = getattr(module, attr_name, None)
    if resolved is None:
        raise BridgeEvaluatorError(f"Evaluator '{ref}' was not found.")
    if not callable(resolved):
        raise BridgeEvaluatorError(f"Evaluator '{ref}' is not callable.")
    return resolved


def _normalize_eval_output(result: Any) -> tuple[float, dict[str, Any]]:
    if isinstance(result, EvalOutput):
        reward = result.average_score
        if reward is None:
            numeric_scores = [
                float(item.score) for item in result.eval_output_items if isinstance(item.score, int | float)
            ]
            reward = (sum(numeric_scores) / len(numeric_scores)) if numeric_scores else 0.0
        if not isinstance(reward, int | float):
            raise BridgeEvaluatorError("Evaluator returned non-numeric average score.")
        return float(reward), {"eval_output": result.model_dump(mode="json")}

    if isinstance(result, int | float):
        return float(result), {}

    if isinstance(result, dict):
        reward = result.get("reward")
        details = result.get("details", {})
        if not isinstance(reward, int | float):
            raise BridgeEvaluatorError("Evaluator dict output must include numeric 'reward'.")
        if not isinstance(details, dict):
            raise BridgeEvaluatorError("Evaluator dict output 'details' must be an object.")
        return float(reward), details

    raise BridgeEvaluatorError("Evaluator must return `EvalOutput`, numeric reward, or dict payload.")


def _coerce_sample(sample_like: Any, fallback_item_id: str) -> AtifEvalSample:
    if isinstance(sample_like, AtifEvalSample):
        return sample_like

    if isinstance(sample_like, dict) and "trajectory" in sample_like:
        trajectory = ATIFTrajectory.model_validate(sample_like["trajectory"])
        return AtifEvalSample(
            item_id=sample_like.get("item_id", fallback_item_id),
            trajectory=trajectory,
            expected_output_obj=sample_like.get("expected_output_obj"),
            output_obj=sample_like.get("output_obj"),
            metadata=sample_like.get("metadata", {}),
        )

    trajectory = ATIFTrajectory.model_validate(sample_like)
    return AtifEvalSample(item_id=fallback_item_id, trajectory=trajectory)


def load_atif_samples(artifact_path: Path) -> list[AtifEvalSample]:
    """Load one-or-many ATIF trajectory samples from artifact JSON."""
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [_coerce_sample(item, fallback_item_id=f"item-{idx}") for idx, item in enumerate(payload, start=1)]
    return [_coerce_sample(payload, fallback_item_id="item-1")]


async def _run_custom_evaluator(
    *,
    evaluator_ref: str,
    atif_samples: list[AtifEvalSample],
    artifact_path: Path,
) -> tuple[float, dict[str, Any]]:
    evaluator_callable = _load_callable(evaluator_ref)
    try:
        raw_result = evaluator_callable(atif_samples=atif_samples, artifact_path=str(artifact_path))
    except TypeError:
        # Support legacy custom signatures that only accept atif samples.
        raw_result = evaluator_callable(atif_samples)
    if inspect.isawaitable(raw_result):
        raw_result = await raw_result
    reward, details = _normalize_eval_output(raw_result)
    details["evaluator_ref"] = evaluator_ref
    details["evaluator_mode"] = "custom"
    return reward, details


async def _run_builtin_evaluator(
    *,
    evaluator_kind: str,
    atif_samples: list[AtifEvalSample],
    config_file: str,
    evaluator_name: str | None,
) -> tuple[float, dict[str, Any]]:
    configured_name = evaluator_name or evaluator_kind
    config = load_config(Path(config_file))
    from nat.plugins.eval.runtime.builder import WorkflowEvalBuilder

    async with WorkflowEvalBuilder.from_config(config) as builder:
        evaluator = builder.get_evaluator(configured_name)
        evaluate_atif_fn = getattr(evaluator, "evaluate_atif_fn", None)
        if not callable(evaluate_atif_fn):
            raise BridgeEvaluatorError(
                f"Configured evaluator '{configured_name}' does not expose `evaluate_atif_fn`."
            )
        eval_output = evaluate_atif_fn(atif_samples)
        if isinstance(eval_output, Awaitable):
            eval_output = await eval_output
        reward, details = _normalize_eval_output(eval_output)
        details["evaluator_name"] = configured_name
        details["evaluator_mode"] = "builtin"
        details["evaluator_kind"] = evaluator_kind
        return reward, details


async def evaluate_artifact(
    *,
    artifact_path: Path,
    evaluator_kind: str,
    evaluator_ref: str | None = None,
    config_file: str | None = None,
    evaluator_name: str | None = None,
) -> tuple[float, dict[str, Any]]:
    """Evaluate ATIF artifacts with builtin or custom evaluator dispatch."""
    atif_samples = load_atif_samples(artifact_path)
    if evaluator_kind == "custom":
        if not evaluator_ref:
            raise BridgeEvaluatorError("`--evaluator-ref` is required when `--evaluator-kind custom` is used.")
        return await _run_custom_evaluator(
            evaluator_ref=evaluator_ref,
            atif_samples=atif_samples,
            artifact_path=artifact_path,
        )

    if evaluator_ref:
        reward, details = await _run_custom_evaluator(
            evaluator_ref=evaluator_ref,
            atif_samples=atif_samples,
            artifact_path=artifact_path,
        )
        details["evaluator_mode"] = "builtin_ref_override"
        details["evaluator_kind"] = evaluator_kind
        return reward, details

    if not config_file:
        raise BridgeEvaluatorError(
            f"`--config-file` is required for builtin evaluator kind '{evaluator_kind}'."
        )
    return await _run_builtin_evaluator(
        evaluator_kind=evaluator_kind,
        atif_samples=atif_samples,
        config_file=config_file,
        evaluator_name=evaluator_name,
    )


def evaluate_artifact_sync(
    *,
    artifact_path: Path,
    evaluator_kind: str,
    evaluator_ref: str | None = None,
    config_file: str | None = None,
    evaluator_name: str | None = None,
) -> tuple[float, dict[str, Any]]:
    """Synchronous wrapper around bridge evaluation."""
    return asyncio.run(
        evaluate_artifact(
            artifact_path=artifact_path,
            evaluator_kind=evaluator_kind,
            evaluator_ref=evaluator_ref,
            config_file=config_file,
            evaluator_name=evaluator_name,
        )
    )
