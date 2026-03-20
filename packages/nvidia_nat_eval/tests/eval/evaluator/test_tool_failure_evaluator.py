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
"""Unit tests for ToolFailureEvaluator model population.

Validates that ToolFailureReasoning, ToolSummary, and _ToolCall are correctly
populated from both the legacy IntermediateStep lane and the ATIF lane, and
that error detection correctly distinguishes failures from successes.
"""

from __future__ import annotations

from typing import Any

import pytest

from nat.data_models.atif import ATIFAgentConfig
from nat.data_models.atif import ATIFTrajectory
from nat.data_models.atif.observation import Observation
from nat.data_models.atif.observation_result import ObservationResult
from nat.data_models.atif.step import Step
from nat.data_models.atif.tool_call import ToolCall as ATIFToolCall
from nat.data_models.evaluator import EvalInputItem
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.invocation_node import InvocationNode
from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSample
from nat.plugins.eval.tool_failure_evaluator.evaluator import ToolFailureEvaluator
from nat.plugins.eval.tool_failure_evaluator.models import ToolFailureReasoning

_DUMMY_ANCESTRY: InvocationNode = InvocationNode(function_id="f-0", function_name="test_fn")


def _wrap(payload: IntermediateStepPayload) -> IntermediateStep:
    """Wrap a payload into a full IntermediateStep."""
    return IntermediateStep(parent_id="root", function_ancestry=_DUMMY_ANCESTRY, payload=payload)


def _tool_end_step(
    name: str,
    output: dict[str, Any] | None = None,
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


def _atif_step(
    step_id: int,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    observation_content: str = "",
    extra: dict[str, Any] | None = None,
) -> Step:
    """Build an ATIF Step with a single tool call and observation."""
    return Step(
        step_id=step_id,
        source="agent",
        tool_calls=[ATIFToolCall(tool_call_id=f"tc-{step_id}", function_name=tool_name, arguments=arguments or {})],
        observation=Observation(results=[ObservationResult(content=observation_content)]),
        extra=extra,
    )


def _atif_sample(item_id: str, steps: list[Step]) -> AtifEvalSample:
    """Build an AtifEvalSample wrapping the given steps."""
    trajectory: ATIFTrajectory = ATIFTrajectory(
        session_id=f"session-{item_id}",
        agent=ATIFAgentConfig(name="test-agent", version="0.0.0"),
        steps=steps,
    )
    return AtifEvalSample(item_id=item_id, trajectory=trajectory)


@pytest.fixture(name="evaluator")
def evaluator_fixture() -> ToolFailureEvaluator:
    """Provide a fresh ToolFailureEvaluator instance."""
    return ToolFailureEvaluator()


class TestLegacyLaneModelPopulation:
    """Verify ToolFailureReasoning, ToolSummary, and _ToolCall are correctly
    populated from legacy IntermediateStep trajectories.
    """

    async def test_empty_trajectory_produces_default_reasoning(self, evaluator: ToolFailureEvaluator):
        """An empty trajectory should yield default ToolFailureReasoning with
        zero counts, no failed tools, and a perfect score.
        """
        result = await evaluator.evaluate_item(_eval_input("empty", []))

        reasoning: ToolFailureReasoning = result.reasoning
        assert reasoning.total_tool_calls == 0
        assert reasoning.failed_tool_calls == 0
        assert reasoning.failed_tools == []
        assert reasoning.per_tool_summary == []
        assert reasoning.score == 1.0

    async def test_all_failed_calls_populate_summary_with_error_details(self, evaluator: ToolFailureEvaluator):
        """When every call errors, ToolSummary should capture each failure and
        every _ToolCall.error should contain the error string while output is None.
        """
        error_output: dict[str, Any] = {"status": "error", "content": "ValueError: bad input"}
        trajectory = [
            _tool_end_step("lookup", output=error_output, tool_input={"query": "q1"}),
            _tool_end_step("lookup", output=error_output, tool_input={"query": "q2"}),
        ]
        result = await evaluator.evaluate_item(_eval_input("fail", trajectory))

        reasoning: ToolFailureReasoning = result.reasoning
        assert reasoning.total_tool_calls == 2
        assert reasoning.failed_tool_calls == 2
        assert reasoning.failed_tools == ["lookup"]
        assert reasoning.score == 0.0

        summary = reasoning.per_tool_summary[0]
        assert summary.tool_name == "lookup"
        assert summary.total_calls == 2
        assert summary.failed_calls == 2
        assert len(summary.attempts) == 2
        for attempt in summary.attempts:
            assert attempt.error == "ValueError: bad input"
            assert attempt.output is None

    async def test_mixed_results_split_correctly_across_models(self, evaluator: ToolFailureEvaluator):
        """When one tool succeeds and another fails, only the failing tool
        should appear in per_tool_summary and failed_tools.
        """
        trajectory = [
            _tool_end_step("search", output={"content": "ok"}, tool_input={"q": "a"}),
            _tool_end_step("lookup", output={
                "status": "error", "content": "KeyError"
            }, tool_input={"k": "x"}),
        ]
        result = await evaluator.evaluate_item(_eval_input("mixed", trajectory))

        reasoning: ToolFailureReasoning = result.reasoning
        assert reasoning.total_tool_calls == 2
        assert reasoning.failed_tool_calls == 1
        assert reasoning.failed_tools == ["lookup"]
        assert reasoning.score == 0.5

        assert len(reasoning.per_tool_summary) == 1
        assert reasoning.per_tool_summary[0].tool_name == "lookup"

    async def test_same_tool_mixed_results_filters_attempts_to_failures_only(self, evaluator: ToolFailureEvaluator):
        """When a single tool has both successes and failures, ToolSummary.attempts
        should contain only the failed _ToolCall entries while total_calls reflects all.
        """
        trajectory = [
            _tool_end_step("tool_a", output={"content": "ok"}, tool_input={"q": "good"}),
            _tool_end_step("tool_a", output={
                "status": "error", "content": "boom"
            }, tool_input={"q": "bad"}),
        ]
        result = await evaluator.evaluate_item(_eval_input("filter", trajectory))

        reasoning: ToolFailureReasoning = result.reasoning
        summary = reasoning.per_tool_summary[0]
        assert summary.total_calls == 2
        assert summary.failed_calls == 1
        assert len(summary.attempts) == 1
        assert summary.attempts[0].error == "boom"
        assert summary.attempts[0].input == {"q": "bad"}

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
        step: IntermediateStep = _wrap(
            IntermediateStepPayload(
                event_type=IntermediateStepType.TOOL_END,
                name=None,
                data=StreamEventData(input=None, output={
                    "status": "error", "content": "err"
                }),
            ))
        result = await evaluator.evaluate_item(_eval_input("noname", [step]))

        assert result.reasoning.failed_tools == ["unknown"]
        assert result.reasoning.per_tool_summary[0].tool_name == "unknown"


class TestAtifLaneModelPopulation:
    """Verify ToolFailureReasoning, ToolSummary, and _ToolCall are correctly
    populated from ATIF trajectories using each error detection path.
    """

    async def test_error_detected_via_extra_tool_errors(self, evaluator: ToolFailureEvaluator):
        """Structured error metadata in step.extra['tool_errors'] should populate
        _ToolCall.error with the error string from the extra entry.
        """
        steps = [
            _atif_step(
                1,
                "lookup",
                arguments={"query": "q1"},
                observation_content="Column not found",
                extra={"tool_errors": [{
                    "tool": "lookup", "error": "ValueError: Column not found"
                }]},
            ),
        ]
        result = await evaluator.evaluate_atif_item(_atif_sample("extra", steps))

        reasoning: ToolFailureReasoning = result.reasoning
        assert reasoning.failed_tool_calls == 1
        assert reasoning.failed_tools == ["lookup"]
        assert reasoning.per_tool_summary[0].attempts[0].error == "ValueError: Column not found"
        assert reasoning.per_tool_summary[0].attempts[0].input == {"query": "q1"}

    async def test_error_detected_via_stringified_tool_message_dict(self, evaluator: ToolFailureEvaluator):
        """A Python dict literal with status='error' in the observation content
        should be parsed and the content field used as _ToolCall.error.
        """
        steps = [
            _atif_step(
                1,
                "api_call",
                observation_content="{'status': 'error', 'content': 'TimeoutError: timed out'}",
            ),
        ]
        result = await evaluator.evaluate_atif_item(_atif_sample("parsed", steps))

        assert result.reasoning.failed_tool_calls == 1
        assert result.reasoning.per_tool_summary[0].attempts[0].error == "TimeoutError: timed out"

    async def test_error_detected_via_raw_error_pattern(self, evaluator: ToolFailureEvaluator):
        """Observation content matching 'XyzError: ...' should be detected as a
        failure and used directly as the _ToolCall.error string.
        """
        steps = [
            _atif_step(1, "processor", observation_content="RuntimeError: internal failure"),
        ]
        result = await evaluator.evaluate_atif_item(_atif_sample("pattern", steps))

        assert result.reasoning.failed_tool_calls == 1
        assert result.reasoning.per_tool_summary[0].attempts[0].error == "RuntimeError: internal failure"

    async def test_extra_tool_errors_takes_priority_over_observation_pattern(self, evaluator: ToolFailureEvaluator):
        """When both extra['tool_errors'] and a raw error pattern match, the
        error string should come from extra, not the observation content.
        """
        steps = [
            _atif_step(
                1,
                "tool",
                observation_content="ValueError: from observation",
                extra={"tool_errors": [{
                    "tool": "tool", "error": "ValueError: from extra"
                }]},
            ),
        ]
        result = await evaluator.evaluate_atif_item(_atif_sample("priority", steps))

        assert result.reasoning.per_tool_summary[0].attempts[0].error == "ValueError: from extra"

    async def test_mixed_success_and_failure_populates_only_failing_tool(self, evaluator: ToolFailureEvaluator):
        """With one successful and one failing tool, only the failing tool
        should appear in per_tool_summary and failed_tools.
        """
        steps = [
            _atif_step(1, "good_tool", observation_content="success"),
            _atif_step(
                2,
                "bad_tool",
                observation_content="err",
                extra={"tool_errors": [{
                    "tool": "bad_tool", "error": "KeyError: not found"
                }]},
            ),
        ]
        result = await evaluator.evaluate_atif_item(_atif_sample("mixed", steps))

        assert result.reasoning.total_tool_calls == 2
        assert result.reasoning.failed_tool_calls == 1
        assert result.reasoning.failed_tools == ["bad_tool"]
        assert result.score == 0.5

    async def test_observation_with_none_content_is_not_treated_as_error(self, evaluator: ToolFailureEvaluator):
        """An observation result with content=None should not trigger error detection."""
        step = Step(
            step_id=1,
            source="agent",
            tool_calls=[ATIFToolCall(tool_call_id="tc-1", function_name="tool", arguments={})],
            observation=Observation(results=[ObservationResult(content=None)]),
        )
        result = await evaluator.evaluate_atif_item(_atif_sample("no-content", [step]))

        assert result.reasoning.total_tool_calls == 1
        assert result.reasoning.failed_tool_calls == 0

    async def test_non_error_string_not_misclassified(self, evaluator: ToolFailureEvaluator):
        """Observation content that contains a colon but doesn't match the
        'XyzError: ...' pattern should not be treated as a failure.
        """
        steps = [_atif_step(1, "tool", observation_content="Status: all clear, no issues")]
        result = await evaluator.evaluate_atif_item(_atif_sample("benign", steps))

        assert result.reasoning.failed_tool_calls == 0
        assert result.score == 1.0
