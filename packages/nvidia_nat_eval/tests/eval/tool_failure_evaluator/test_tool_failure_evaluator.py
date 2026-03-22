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
"""Unit tests for ToolFailureEvaluator."""

from __future__ import annotations

from typing import Any

import pytest

from nat.data_models.evaluator import EvalInputItem
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.intermediate_step import ToolErrorData
from nat.data_models.invocation_node import InvocationNode
from nat.plugins.eval.tool_failure_evaluator.evaluator import ToolFailureEvaluator
from nat.plugins.eval.tool_failure_evaluator.models import _ToolFailureReasoning

_DUMMY_ANCESTRY: InvocationNode = InvocationNode(function_id="f-0", function_name="test_fn")


def _wrap(payload: IntermediateStepPayload) -> IntermediateStep:
    """Wrap a payload into a full IntermediateStep."""
    return IntermediateStep(parent_id="root", function_ancestry=_DUMMY_ANCESTRY, payload=payload)


def _tool_end_step(
    name: str,
    output: Any | None = None,
    tool_input: dict[str, Any] | str | None = None,
) -> IntermediateStep:
    """Build a TOOL_END IntermediateStep."""
    return _wrap(
        IntermediateStepPayload(
            event_type=IntermediateStepType.TOOL_END,
            name=name,
            data=StreamEventData(input=tool_input, output=output),
        ))


def _eval_input(item_id: str, trajectory: list[IntermediateStep]) -> EvalInputItem:
    """Build an EvalInputItem wrapping the given trajectory."""
    return EvalInputItem(
        id=item_id,
        input_obj="question",
        expected_output_obj="answer",
        trajectory=trajectory,
        full_dataset_entry={},
    )


@pytest.fixture(name="evaluator")
def evaluator_fixture() -> ToolFailureEvaluator:
    """Provide a fresh ToolFailureEvaluator instance."""
    return ToolFailureEvaluator()


class TestEvaluateIntermediateStepTrajectory:
    """Tests for evaluating IntermediateStep trajectories."""

    async def test_empty_trajectory_produces_default_reasoning(self, evaluator: ToolFailureEvaluator):
        """An empty trajectory should yield default ToolFailureReasoning with
        zero counts, no failed tools, and a perfect score.
        """
        result = await evaluator.evaluate_item(_eval_input("empty", []))

        reasoning: _ToolFailureReasoning = result.reasoning
        assert reasoning.total_tool_calls == 0
        assert reasoning.failed_tool_calls == 0
        assert reasoning.failed_tools is None
        assert reasoning.per_tool_summary is None
        assert reasoning.score == 1.0

    async def test_all_failed_calls_populate_summary_with_error_details(self, evaluator: ToolFailureEvaluator):
        """When every call errors, ToolSummary should capture each failure and
        every _ToolCall.error should contain the error string while output is None.
        """
        error_output: ToolErrorData = ToolErrorData(
            content="ValueError: bad input",
            error_type="ValueError",
            error_message="bad input",
        )
        trajectory = [
            _tool_end_step("lookup", output=error_output, tool_input={"query": "q1"}),
            _tool_end_step("lookup", output=error_output, tool_input={"query": "q2"}),
        ]
        result = await evaluator.evaluate_item(_eval_input("fail", trajectory))

        reasoning: _ToolFailureReasoning = result.reasoning
        assert reasoning.total_tool_calls == 2
        assert reasoning.failed_tool_calls == 2
        assert reasoning.failed_tools == ["lookup"]
        assert reasoning.score == 0.0

        summary = reasoning.per_tool_summary[0]
        assert summary.tool_name == "lookup"
        assert summary.total_calls == 2
        assert summary.failed_calls == 2
        assert len(summary.failed_attempts) == 2
        for attempt in summary.failed_attempts:
            assert attempt.error == "ValueError: bad input"
            assert attempt.output is None

    async def test_mixed_results_split_correctly_across_models(self, evaluator: ToolFailureEvaluator):
        """When one tool succeeds and another fails, only the failing tool
        should appear in per_tool_summary and failed_tools.
        """
        error_output: ToolErrorData = ToolErrorData(
            content="KeyError: missing",
            error_type="KeyError",
            error_message="missing",
        )
        trajectory = [
            _tool_end_step("search", output={"content": "ok"}, tool_input={"q": "a"}),
            _tool_end_step("lookup", output=error_output, tool_input={"k": "x"}),
        ]
        result = await evaluator.evaluate_item(_eval_input("mixed", trajectory))

        reasoning: _ToolFailureReasoning = result.reasoning
        assert reasoning.total_tool_calls == 2
        assert reasoning.failed_tool_calls == 1
        assert reasoning.failed_tools == ["lookup"]
        assert reasoning.score == 0.5

        assert len(reasoning.per_tool_summary) == 1
        assert reasoning.per_tool_summary[0].tool_name == "lookup"

    async def test_same_tool_mixed_results_filters_attempts_to_failures_only(self, evaluator: ToolFailureEvaluator):
        """When a single tool has both successes and failures, ToolSummary.failed_attempts
        should contain only the failed _ToolCall entries while total_calls reflects all.
        """
        error_output: ToolErrorData = ToolErrorData(
            content="RuntimeError: boom",
            error_type="RuntimeError",
            error_message="boom",
        )
        trajectory = [
            _tool_end_step("tool_a", output={"content": "ok"}, tool_input={"q": "good"}),
            _tool_end_step("tool_a", output=error_output, tool_input={"q": "bad"}),
        ]
        result = await evaluator.evaluate_item(_eval_input("filter", trajectory))

        reasoning: _ToolFailureReasoning = result.reasoning
        summary = reasoning.per_tool_summary[0]
        assert summary.total_calls == 2
        assert summary.failed_calls == 1
        assert len(summary.failed_attempts) == 1
        assert summary.failed_attempts[0].error == "RuntimeError: boom"
        assert summary.failed_attempts[0].input == {"q": "bad"}

    async def test_none_data_on_step_is_not_treated_as_error(self, evaluator: ToolFailureEvaluator):
        """A TOOL_END step with data=None should count as a call but not a failure."""
        step: IntermediateStep = _wrap(
            IntermediateStepPayload(event_type=IntermediateStepType.TOOL_END, name="tool_x", data=None))
        result = await evaluator.evaluate_item(_eval_input("nodata", [step]))

        assert result.reasoning.total_tool_calls == 1
        assert result.reasoning.failed_tool_calls == 0
        assert result.score == 1.0

    async def test_missing_tool_name_recorded_as_unknown(self, evaluator: ToolFailureEvaluator):
        """When step.name is None, the evaluator should use 'unknown' as the tool
        name in both failed_tools and per_tool_summary.
        """
        error_output: ToolErrorData = ToolErrorData(
            content="RuntimeError: err",
            error_type="RuntimeError",
            error_message="err",
        )
        step: IntermediateStep = _wrap(
            IntermediateStepPayload(
                event_type=IntermediateStepType.TOOL_END,
                name=None,
                data=StreamEventData(input=None, output=error_output),
            ))
        result = await evaluator.evaluate_item(_eval_input("noname", [step]))

        assert result.reasoning.failed_tools == ["unknown"]
        assert result.reasoning.per_tool_summary[0].tool_name == "unknown"

    async def test_serialized_tool_error_dict_detected_as_error(self, evaluator: ToolFailureEvaluator):
        """After JSON serialization/deserialization, ToolErrorData becomes a dict.
        The evaluator must detect dicts with 'error_type' key as tool errors.
        """
        error_as_dict: dict[str, str] = {
            "content": "ValueError: Column not found",
            "error_type": "ValueError",
            "error_message": "Column not found",
        }
        trajectory = [_tool_end_step("lookup", output=error_as_dict, tool_input={"query": "bad"})]
        result = await evaluator.evaluate_item(_eval_input("serialized", trajectory))

        reasoning: _ToolFailureReasoning = result.reasoning
        assert reasoning.total_tool_calls == 1
        assert reasoning.failed_tool_calls == 1
        assert reasoning.failed_tools == ["lookup"]
        assert reasoning.score == 0.0
        assert reasoning.per_tool_summary[0].failed_attempts[0].error == "ValueError: Column not found"

    async def test_dict_without_error_type_not_treated_as_error(self, evaluator: ToolFailureEvaluator):
        """A dict output without 'error_type' key should not be treated as an error."""
        normal_dict_output: dict[str, str] = {"content": "some result", "status": "ok"}
        trajectory = [_tool_end_step("tool", output=normal_dict_output)]
        result = await evaluator.evaluate_item(_eval_input("normal_dict", trajectory))

        assert result.reasoning.total_tool_calls == 1
        assert result.reasoning.failed_tool_calls == 0
        assert result.reasoning.failed_tools is None
        assert result.score == 1.0
