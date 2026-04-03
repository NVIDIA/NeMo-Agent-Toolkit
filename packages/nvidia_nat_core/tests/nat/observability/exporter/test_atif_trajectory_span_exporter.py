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
"""Tests for ATIFTrajectorySpanExporter."""

import pytest

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.atif import ATIFObservation
from nat.data_models.atif import ATIFObservationResult
from nat.data_models.atif import ATIFStep
from nat.data_models.atif import ATIFStepMetrics
from nat.data_models.atif import ATIFToolCall
from nat.data_models.atif import ATIFTrajectory
from nat.data_models.atif import ATIFAgentConfig
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.intermediate_step import UsageInfo
from nat.data_models.invocation_node import InvocationNode
from nat.data_models.span import Span
from nat.data_models.span import SpanAttributes
from nat.data_models.token_usage import TokenUsageBaseModel
from nat.observability.exporter.atif_trajectory_span_exporter import ATIFTrajectorySpanExporter
from nat.observability.utils.time_utils import ns_timestamp

_BASE_TIME = 1700000000.0


class ConcreteATIFSpanExporter(ATIFTrajectorySpanExporter[Span]):
    """Concrete implementation for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exported_spans: list[Span] = []

    async def export_processed(self, item: Span) -> None:
        self.exported_spans.append(item)


def _make_step(
    event_type: IntermediateStepType,
    *,
    name: str = "test",
    input_data: str | dict | None = None,
    output_data: str | dict | None = None,
    timestamp_offset: float = 0.0,
    parent_id: str = "root",
    function_name: str = "my_workflow",
    function_id: str = "func-id-1",
    function_parent_id: str | None = None,
    function_parent_name: str | None = None,
    usage: UsageInfo | None = None,
    step_uuid: str | None = None,
    framework: LLMFrameworkEnum | None = None,
) -> IntermediateStep:
    """Create a minimal IntermediateStep for testing."""
    payload_kwargs: dict = {
        "event_type": event_type,
        "event_timestamp": _BASE_TIME + timestamp_offset,
        "name": name,
        "data": StreamEventData(input=input_data, output=output_data),
    }
    if usage is not None:
        payload_kwargs["usage_info"] = usage
    if step_uuid is not None:
        payload_kwargs["UUID"] = step_uuid
    if framework is not None:
        payload_kwargs["framework"] = framework
    if event_type.endswith("_END") and event_type != "LLM_NEW_TOKEN":
        payload_kwargs["span_event_timestamp"] = _BASE_TIME + timestamp_offset - 0.5
    return IntermediateStep(
        parent_id=parent_id,
        function_ancestry=InvocationNode(
            function_name=function_name,
            function_id=function_id,
            parent_id=function_parent_id,
            parent_name=function_parent_name,
        ),
        payload=IntermediateStepPayload(**payload_kwargs),
    )


def _make_usage(prompt: int = 100, completion: int = 50) -> UsageInfo:
    return UsageInfo(
        token_usage=TokenUsageBaseModel(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        ),
    )


def _build_simple_trajectory() -> ATIFTrajectory:
    """Build a simple trajectory: user → agent (LLM + 1 tool) → agent (final answer)."""
    return ATIFTrajectory(
        session_id="test-session",
        agent=ATIFAgentConfig(name="test_agent", version="0.0.0"),
        steps=[
            ATIFStep(
                step_id=1,
                source="user",
                message="What is 2 * 3?",
                timestamp="2023-11-14T22:13:20+00:00",
                extra={
                    "ancestry": {
                        "function_id": "fn_react_agent",
                        "function_name": "react_agent",
                        "parent_id": None,
                        "parent_name": None,
                    },
                    "invocation": {
                        "start_timestamp": _BASE_TIME,
                        "end_timestamp": _BASE_TIME + 5.0,
                    },
                },
            ),
            ATIFStep(
                step_id=2,
                source="agent",
                message="I'll use the multiply tool.",
                timestamp="2023-11-14T22:13:21+00:00",
                model_name="llama-3.1-70b",
                metrics=ATIFStepMetrics(prompt_tokens=100, completion_tokens=50),
                tool_calls=[
                    ATIFToolCall(
                        tool_call_id="call_abc",
                        function_name="calculator__multiply",
                        arguments={"a": 2, "b": 3},
                    ),
                ],
                observation=ATIFObservation(
                    results=[
                        ATIFObservationResult(source_call_id="call_abc", content="6"),
                    ],
                ),
                extra={
                    "ancestry": {
                        "function_id": "fn_llm",
                        "function_name": "llama-3.1-70b",
                        "parent_id": "fn_react_agent",
                        "parent_name": "react_agent",
                    },
                    "invocation": {
                        "start_timestamp": _BASE_TIME + 0.5,
                        "end_timestamp": _BASE_TIME + 1.5,
                    },
                    "tool_ancestry": [
                        {
                            "function_id": "fn_multiply",
                            "function_name": "calculator__multiply",
                            "parent_id": "fn_react_agent",
                            "parent_name": "react_agent",
                        },
                    ],
                    "tool_invocations": [
                        {
                            "start_timestamp": _BASE_TIME + 1.5,
                            "end_timestamp": _BASE_TIME + 2.0,
                            "invocation_id": "call_abc",
                            "status": "completed",
                        },
                    ],
                },
            ),
            # Terminal step: no model_name, no tool_calls → merged into workflow span
            ATIFStep(
                step_id=3,
                source="agent",
                message="The result is 6.",
                timestamp="2023-11-14T22:13:25+00:00",
                extra={
                    "ancestry": {
                        "function_id": "fn_react_agent",
                        "function_name": "react_agent",
                        "parent_id": None,
                        "parent_name": None,
                    },
                    "invocation": {
                        "start_timestamp": _BASE_TIME + 4.0,
                        "end_timestamp": _BASE_TIME + 5.0,
                    },
                },
            ),
        ],
    )


def _build_nested_trajectory() -> ATIFTrajectory:
    """Build a nested trajectory: react_agent → LLM → power_of_two → calculator__multiply."""
    return ATIFTrajectory(
        session_id="test-nested-session",
        agent=ATIFAgentConfig(name="power_of_two_agent", version="0.0.0"),
        steps=[
            ATIFStep(
                step_id=1,
                source="user",
                message="What is the power of two of 5?",
                timestamp="2023-11-14T22:13:20+00:00",
                extra={
                    "ancestry": {
                        "function_id": "fn_react_agent",
                        "function_name": "react_agent",
                        "parent_id": None,
                        "parent_name": None,
                    },
                    "invocation": {
                        "start_timestamp": _BASE_TIME,
                        "end_timestamp": _BASE_TIME + 10.0,
                    },
                },
            ),
            ATIFStep(
                step_id=2,
                source="agent",
                message="I'll compute 5^2 using power_of_two.",
                timestamp="2023-11-14T22:13:21+00:00",
                model_name="llama-3.1-70b",
                metrics=ATIFStepMetrics(prompt_tokens=200, completion_tokens=80),
                tool_calls=[
                    ATIFToolCall(
                        tool_call_id="call_pow2",
                        function_name="power_of_two",
                        arguments={"x": 5},
                    ),
                    ATIFToolCall(
                        tool_call_id="call_mul",
                        function_name="calculator__multiply",
                        arguments={"a": 5, "b": 5},
                    ),
                ],
                observation=ATIFObservation(
                    results=[
                        ATIFObservationResult(source_call_id="call_pow2", content="25"),
                        ATIFObservationResult(source_call_id="call_mul", content="25"),
                    ],
                ),
                extra={
                    "ancestry": {
                        "function_id": "fn_llm",
                        "function_name": "llama-3.1-70b",
                        "parent_id": "fn_react_agent",
                        "parent_name": "react_agent",
                    },
                    "invocation": {
                        "start_timestamp": _BASE_TIME + 1.0,
                        "end_timestamp": _BASE_TIME + 2.0,
                    },
                    "tool_ancestry": [
                        {
                            "function_id": "fn_power_of_two",
                            "function_name": "power_of_two",
                            "parent_id": "fn_react_agent",
                            "parent_name": "react_agent",
                        },
                        {
                            "function_id": "fn_multiply",
                            "function_name": "calculator__multiply",
                            "parent_id": "fn_power_of_two",
                            "parent_name": "power_of_two",
                        },
                    ],
                    "tool_invocations": [
                        {
                            "start_timestamp": _BASE_TIME + 2.0,
                            "end_timestamp": _BASE_TIME + 4.0,
                            "invocation_id": "call_pow2",
                            "status": "completed",
                        },
                        {
                            "start_timestamp": _BASE_TIME + 2.5,
                            "end_timestamp": _BASE_TIME + 3.5,
                            "invocation_id": "call_mul",
                            "status": "completed",
                        },
                    ],
                },
            ),
            # Terminal step: merged into workflow span
            ATIFStep(
                step_id=3,
                source="agent",
                message="The power of two of 5 is 25.",
                timestamp="2023-11-14T22:13:30+00:00",
                extra={
                    "ancestry": {
                        "function_id": "fn_react_agent",
                        "function_name": "react_agent",
                        "parent_id": None,
                        "parent_name": None,
                    },
                    "invocation": {
                        "start_timestamp": _BASE_TIME + 8.0,
                        "end_timestamp": _BASE_TIME + 10.0,
                    },
                },
            ),
        ],
    )


def _build_orphan_function_trajectory() -> ATIFTrajectory:
    """Build a trajectory with orphan function steps (no LLM events).

    This mirrors the real-world case where a react_agent workflow
    only emits FUNCTION events (no LLM_START/LLM_END).
    """
    return ATIFTrajectory(
        session_id="test-orphan-session",
        agent=ATIFAgentConfig(name="power_of_two_agent", version="0.0.0"),
        steps=[
            ATIFStep(
                step_id=1,
                source="user",
                message="What is the power of two of 5?",
                timestamp="2023-11-14T22:13:20+00:00",
                extra={
                    "ancestry": {
                        "function_id": "root",
                        "function_name": "root",
                    },
                    "invocation": {
                        "status": "completed",
                    },
                },
            ),
            # Orphan function step: calculator__multiply (child of power_of_two)
            ATIFStep(
                step_id=2,
                source="agent",
                message="",
                timestamp="2023-11-14T22:13:22+00:00",
                tool_calls=[
                    ATIFToolCall(
                        tool_call_id="call_mul",
                        function_name="calculator__multiply",
                        arguments={"a": 5, "b": 5},
                    ),
                ],
                observation=ATIFObservation(
                    results=[
                        ATIFObservationResult(source_call_id="call_mul", content="25"),
                    ],
                ),
                extra={
                    "ancestry": {
                        "function_id": "uuid-multiply",
                        "function_name": "calculator__multiply",
                        "parent_id": "uuid-pow2",
                        "parent_name": "power_of_two",
                    },
                    "invocation": {"status": "completed"},
                    "tool_ancestry": [
                        {
                            "function_id": "uuid-multiply",
                            "function_name": "calculator__multiply",
                            "parent_id": "uuid-pow2",
                            "parent_name": "power_of_two",
                        },
                    ],
                    "tool_invocations": [
                        {
                            "invocation_id": "call_uuid-multiply",
                            "status": "completed",
                        },
                    ],
                },
            ),
            # Orphan function step: power_of_two (child of <workflow>, which is suppressed)
            ATIFStep(
                step_id=3,
                source="agent",
                message="",
                timestamp="2023-11-14T22:13:22.001+00:00",
                tool_calls=[
                    ATIFToolCall(
                        tool_call_id="call_pow2",
                        function_name="power_of_two",
                        arguments={"x": 5},
                    ),
                ],
                observation=ATIFObservation(
                    results=[
                        ATIFObservationResult(source_call_id="call_pow2", content="25"),
                    ],
                ),
                extra={
                    "ancestry": {
                        "function_id": "uuid-pow2",
                        "function_name": "power_of_two",
                        "parent_id": "uuid-workflow-wrapper",
                        "parent_name": "<workflow>",
                    },
                    "invocation": {"status": "completed"},
                    "tool_ancestry": [
                        {
                            "function_id": "uuid-pow2",
                            "function_name": "power_of_two",
                            "parent_id": "uuid-workflow-wrapper",
                            "parent_name": "<workflow>",
                        },
                    ],
                    "tool_invocations": [
                        {
                            "invocation_id": "call_uuid-pow2",
                            "status": "completed",
                        },
                    ],
                },
            ),
            # Terminal step
            ATIFStep(
                step_id=4,
                source="agent",
                message="The power of two of 5 is 25.",
                timestamp="2023-11-14T22:13:30+00:00",
                extra={
                    "ancestry": {
                        "function_id": "root",
                        "function_name": "root",
                    },
                    "invocation": {"status": "completed"},
                },
            ),
        ],
    )


class TestTrajectoryToSpans:
    """Test _trajectory_to_spans conversion."""

    @pytest.fixture
    def exporter(self):
        return ConcreteATIFSpanExporter()

    def test_simple_trajectory_span_count(self, exporter):
        """Simple trajectory produces expected number of spans.

        Terminal step (no model_name, no tools) is merged into workflow span.
        """
        trajectory = _build_simple_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        # 1 workflow + 1 LLM + 1 tool = 3 spans (terminal merged into workflow)
        assert len(spans) == 3

    def test_simple_trajectory_workflow_span(self, exporter):
        """Workflow span has correct name and no parent."""
        trajectory = _build_simple_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        workflow_span = spans[0]
        assert workflow_span.name == "react_agent"
        assert workflow_span.parent is None
        assert workflow_span.attributes["nat.event_type"] == "WORKFLOW_START"
        assert workflow_span.attributes["nat.function.id"] == "fn_react_agent"

    def test_simple_trajectory_workflow_output_from_terminal(self, exporter):
        """Workflow span gets output.value from the terminal step."""
        trajectory = _build_simple_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        workflow_span = spans[0]
        assert SpanAttributes.OUTPUT_VALUE.value in workflow_span.attributes

    def test_simple_trajectory_llm_span_parent(self, exporter):
        """LLM span is a child of the workflow span."""
        trajectory = _build_simple_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        workflow_span = spans[0]
        llm_span = spans[1]
        assert llm_span.name == "llama-3.1-70b"
        assert llm_span.parent is not None
        assert llm_span.parent.context.span_id == workflow_span.context.span_id
        assert llm_span.attributes["nat.event_type"] == "LLM_START"

    def test_simple_trajectory_tool_span_parent(self, exporter):
        """Tool span parent is determined by tool_ancestry.parent_id."""
        trajectory = _build_simple_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        workflow_span = spans[0]
        tool_span = spans[2]
        # tool_ancestry.parent_id = "fn_react_agent" → parent is workflow span
        assert tool_span.name == "calculator__multiply"
        assert tool_span.parent is not None
        assert tool_span.parent.context.span_id == workflow_span.context.span_id

    def test_simple_trajectory_shared_trace_id(self, exporter):
        """All spans share the same trace_id."""
        trajectory = _build_simple_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        trace_ids = {s.context.trace_id for s in spans}
        assert len(trace_ids) == 1

    def test_timing_from_invocation(self, exporter):
        """Span timing comes from invocation.start/end_timestamp."""
        trajectory = _build_simple_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        llm_span = spans[1]
        assert llm_span.start_time == ns_timestamp(_BASE_TIME + 0.5)
        assert llm_span.end_time == ns_timestamp(_BASE_TIME + 1.5)

    def test_tool_timing(self, exporter):
        """Tool span timing comes from tool_invocations."""
        trajectory = _build_simple_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        tool_span = spans[2]
        assert tool_span.start_time == ns_timestamp(_BASE_TIME + 1.5)
        assert tool_span.end_time == ns_timestamp(_BASE_TIME + 2.0)

    def test_token_metrics(self, exporter):
        """LLM span includes token count attributes from step.metrics."""
        trajectory = _build_simple_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        llm_span = spans[1]
        assert llm_span.attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT.value] == 100
        assert llm_span.attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION.value] == 50
        assert llm_span.attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL.value] == 150

    def test_tool_input_output(self, exporter):
        """Tool span has input/output from tool_call.arguments and observation."""
        trajectory = _build_simple_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        tool_span = spans[2]
        assert SpanAttributes.INPUT_VALUE.value in tool_span.attributes
        assert SpanAttributes.OUTPUT_VALUE.value in tool_span.attributes

    def test_user_input_on_workflow_span(self, exporter):
        """Workflow span has user message as input.value."""
        trajectory = _build_simple_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        workflow_span = spans[0]
        assert workflow_span.attributes[SpanAttributes.INPUT_VALUE.value] == "What is 2 * 3?"

    def test_agent_output_on_llm_span(self, exporter):
        """Agent step message appears as output.value on LLM span."""
        trajectory = _build_simple_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        llm_span = spans[1]
        assert SpanAttributes.OUTPUT_VALUE.value in llm_span.attributes

    def test_missing_timing_fallback(self, exporter):
        """When invocation timing is missing, falls back to step timestamp."""
        trajectory = ATIFTrajectory(
            session_id="test",
            agent=ATIFAgentConfig(name="agent", version="0.0.0"),
            steps=[
                ATIFStep(
                    step_id=1,
                    source="user",
                    message="hi",
                    timestamp="2023-11-14T22:13:20+00:00",
                    extra={
                        "ancestry": {
                            "function_id": "fn_wf",
                            "function_name": "workflow",
                            "parent_id": None,
                            "parent_name": None,
                        },
                    },
                ),
            ],
        )
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        assert len(spans) == 1
        # start_time == end_time (zero duration fallback)
        assert spans[0].start_time == spans[0].end_time

    def test_timing_from_timing_map(self, exporter):
        """When invocation has no timestamps, falls back to timing_map."""
        timing_map = {
            "fn_wf": (_BASE_TIME, _BASE_TIME + 3.0),
        }
        trajectory = ATIFTrajectory(
            session_id="test",
            agent=ATIFAgentConfig(name="agent", version="0.0.0"),
            steps=[
                ATIFStep(
                    step_id=1,
                    source="user",
                    message="hi",
                    timestamp="2023-11-14T22:13:20+00:00",
                    extra={
                        "ancestry": {
                            "function_id": "fn_wf",
                            "function_name": "workflow",
                            "parent_id": None,
                            "parent_name": None,
                        },
                        "invocation": {"status": "completed"},
                    },
                ),
            ],
        )
        spans = exporter._trajectory_to_spans(trajectory, timing_map=timing_map)
        assert spans[0].start_time == ns_timestamp(_BASE_TIME)
        assert spans[0].end_time == ns_timestamp(_BASE_TIME + 3.0)

    def test_span_kind_attributes(self, exporter):
        """Spans have correct nat.span.kind values."""
        trajectory = _build_simple_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        assert spans[0].attributes["nat.span.kind"] == "WORKFLOW"
        assert spans[1].attributes["nat.span.kind"] == "LLM"
        assert spans[2].attributes["nat.span.kind"] == "TOOL"


class TestNestedTrajectory:
    """Test nested tool call hierarchy: react_agent → LLM → power_of_two → calculator__multiply."""

    @pytest.fixture
    def exporter(self):
        return ConcreteATIFSpanExporter()

    def test_nested_span_count(self, exporter):
        """Nested trajectory produces expected number of spans."""
        trajectory = _build_nested_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        # 1 workflow + 1 LLM + 2 tools = 4 (terminal merged into workflow)
        assert len(spans) == 4

    def test_nested_power_of_two_parent(self, exporter):
        """power_of_two span parent is the workflow span (parent_id=fn_react_agent)."""
        trajectory = _build_nested_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        workflow_span = spans[0]
        pow2_span = spans[2]  # first tool
        assert pow2_span.name == "power_of_two"
        assert pow2_span.parent is not None
        assert pow2_span.parent.context.span_id == workflow_span.context.span_id

    def test_nested_multiply_parent(self, exporter):
        """calculator__multiply span parent is power_of_two (parent_id=fn_power_of_two)."""
        trajectory = _build_nested_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        pow2_span = spans[2]
        mul_span = spans[3]  # second tool
        assert mul_span.name == "calculator__multiply"
        assert mul_span.parent is not None
        assert mul_span.parent.context.span_id == pow2_span.context.span_id

    def test_nested_timing(self, exporter):
        """Nested tool spans have correct timing from tool_invocations."""
        trajectory = _build_nested_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        pow2_span = spans[2]
        mul_span = spans[3]
        assert pow2_span.start_time == ns_timestamp(_BASE_TIME + 2.0)
        assert pow2_span.end_time == ns_timestamp(_BASE_TIME + 4.0)
        assert mul_span.start_time == ns_timestamp(_BASE_TIME + 2.5)
        assert mul_span.end_time == ns_timestamp(_BASE_TIME + 3.5)

    def test_nested_token_metrics(self, exporter):
        """LLM span in nested trajectory has correct token counts."""
        trajectory = _build_nested_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        llm_span = spans[1]
        assert llm_span.attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT.value] == 200
        assert llm_span.attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION.value] == 80
        assert llm_span.attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL.value] == 280

    def test_nested_workflow_output(self, exporter):
        """Workflow span gets output from terminal step."""
        trajectory = _build_nested_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        workflow_span = spans[0]
        assert SpanAttributes.OUTPUT_VALUE.value in workflow_span.attributes


class TestOrphanFunctionSteps:
    """Test orphan function steps (no LLM events, only FUNCTION_END events).

    This mirrors the real-world case where a react_agent workflow
    only emits FUNCTION events, producing orphan ATIF steps.
    """

    @pytest.fixture
    def exporter(self):
        return ConcreteATIFSpanExporter()

    def test_orphan_span_count(self, exporter):
        """Orphan function trajectory creates correct number of spans."""
        trajectory = _build_orphan_function_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        # 1 workflow + 1 calculator__multiply + 1 power_of_two = 3
        # (terminal merged, no duplicate LLM+tool pairs)
        assert len(spans) == 3

    def test_orphan_no_duplicate_spans(self, exporter):
        """Each orphan function produces ONE span, not an LLM+tool pair."""
        trajectory = _build_orphan_function_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        span_names = [s.name for s in spans]
        assert span_names.count("calculator__multiply") == 1
        assert span_names.count("power_of_two") == 1

    def test_orphan_function_span_kind(self, exporter):
        """Orphan function spans have FUNCTION span kind, not LLM."""
        trajectory = _build_orphan_function_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        for span in spans[1:]:  # skip workflow
            assert span.attributes["nat.span.kind"] == "FUNCTION"

    def test_orphan_parent_fallback_to_workflow(self, exporter):
        """When parent_id is not in fn_span_map, falls back to workflow span."""
        trajectory = _build_orphan_function_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        workflow_span = spans[0]
        # power_of_two has parent_id="uuid-workflow-wrapper" (suppressed <workflow>)
        # → not found → falls back to workflow span
        pow2_span = next(s for s in spans if s.name == "power_of_two")
        assert pow2_span.parent is not None
        assert pow2_span.parent.context.span_id == workflow_span.context.span_id

    def test_orphan_nested_parent(self, exporter):
        """calculator__multiply finds power_of_two as its parent via deferred resolution."""
        trajectory = _build_orphan_function_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        pow2_span = next(s for s in spans if s.name == "power_of_two")
        mul_span = next(s for s in spans if s.name == "calculator__multiply")
        # calculator__multiply is processed BEFORE power_of_two (inner END first),
        # but the deferred parent resolution pass fixes this after all spans are registered.
        assert mul_span.parent is not None
        assert mul_span.parent.context.span_id == pow2_span.context.span_id

    def test_orphan_timing_from_timing_map(self, exporter):
        """Orphan function spans get timing from timing_map."""
        timing_map = {
            "uuid-multiply": (_BASE_TIME + 1.0, _BASE_TIME + 1.5),
            "uuid-pow2": (_BASE_TIME + 0.5, _BASE_TIME + 2.0),
        }
        trajectory = _build_orphan_function_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map=timing_map)
        mul_span = next(s for s in spans if s.name == "calculator__multiply")
        pow2_span = next(s for s in spans if s.name == "power_of_two")
        assert mul_span.start_time == ns_timestamp(_BASE_TIME + 1.0)
        assert mul_span.end_time == ns_timestamp(_BASE_TIME + 1.5)
        assert pow2_span.start_time == ns_timestamp(_BASE_TIME + 0.5)
        assert pow2_span.end_time == ns_timestamp(_BASE_TIME + 2.0)

    def test_orphan_workflow_output(self, exporter):
        """Terminal step output is merged into workflow span."""
        trajectory = _build_orphan_function_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        workflow_span = spans[0]
        assert SpanAttributes.OUTPUT_VALUE.value in workflow_span.attributes


def _build_full_parity_trajectory() -> ATIFTrajectory:
    """Build a trajectory matching the full Path A structure.

    Path A produces: workflow → workflow → LLM → tool → LLM
    This requires:
    - Two source="user" steps (outer + inner WORKFLOW_START)
    - One agent step with model_name + tools (first LLM call)
    - One agent step with model_name but no tools (final LLM answer)
    - One terminal step (no model_name, no tools)
    """
    return ATIFTrajectory(
        session_id="test-parity-session",
        agent=ATIFAgentConfig(name="power_of_two_agent", version="0.0.0"),
        steps=[
            # Outer WORKFLOW_START
            ATIFStep(
                step_id=1,
                source="user",
                message="What is 2 to the power of 5?",
                timestamp="2023-11-14T22:13:20+00:00",
                extra={
                    "ancestry": {
                        "function_id": "fn_outer_workflow",
                        "function_name": "power_of_two_agent",
                        "parent_id": None,
                        "parent_name": None,
                    },
                    "invocation": {
                        "start_timestamp": _BASE_TIME,
                        "end_timestamp": _BASE_TIME + 10.0,
                    },
                },
            ),
            # Inner WORKFLOW_START (react_agent loop)
            ATIFStep(
                step_id=2,
                source="user",
                message="What is 2 to the power of 5?",
                timestamp="2023-11-14T22:13:20.100+00:00",
                extra={
                    "ancestry": {
                        "function_id": "fn_react_agent",
                        "function_name": "react_agent",
                        "parent_id": "fn_outer_workflow",
                        "parent_name": "power_of_two_agent",
                    },
                    "invocation": {
                        "start_timestamp": _BASE_TIME + 0.1,
                        "end_timestamp": _BASE_TIME + 9.0,
                    },
                },
            ),
            # First LLM call with tool calls
            ATIFStep(
                step_id=3,
                source="agent",
                message="I'll compute power of two using the tool.",
                timestamp="2023-11-14T22:13:21+00:00",
                model_name="nvidia/nemotron-3-nano-30b-a3b",
                metrics=ATIFStepMetrics(prompt_tokens=200, completion_tokens=80),
                tool_calls=[
                    ATIFToolCall(
                        tool_call_id="call_pow2",
                        function_name="power_of_two",
                        arguments={"x": 5},
                    ),
                ],
                observation=ATIFObservation(
                    results=[
                        ATIFObservationResult(source_call_id="call_pow2", content="32"),
                    ],
                ),
                extra={
                    "ancestry": {
                        "function_id": "fn_llm_1",
                        "function_name": "nvidia/nemotron-3-nano-30b-a3b",
                        "parent_id": "fn_react_agent",
                        "parent_name": "react_agent",
                    },
                    "invocation": {
                        "start_timestamp": _BASE_TIME + 1.0,
                        "end_timestamp": _BASE_TIME + 3.0,
                    },
                    "tool_ancestry": [
                        {
                            "function_id": "fn_power_of_two",
                            "function_name": "power_of_two",
                            "parent_id": "fn_llm_1",
                            "parent_name": "nvidia/nemotron-3-nano-30b-a3b",
                        },
                    ],
                    "tool_invocations": [
                        {
                            "start_timestamp": _BASE_TIME + 3.0,
                            "end_timestamp": _BASE_TIME + 5.0,
                            "invocation_id": "call_pow2",
                            "status": "completed",
                        },
                    ],
                },
            ),
            # Final LLM response (model_name set, no tools) — Case 3a
            ATIFStep(
                step_id=4,
                source="agent",
                message="2 to the power of 5 is 32.",
                timestamp="2023-11-14T22:13:26+00:00",
                model_name="nvidia/nemotron-3-nano-30b-a3b",
                metrics=ATIFStepMetrics(prompt_tokens=150, completion_tokens=30),
                extra={
                    "ancestry": {
                        "function_id": "fn_llm_2",
                        "function_name": "nvidia/nemotron-3-nano-30b-a3b",
                        "parent_id": "fn_react_agent",
                        "parent_name": "react_agent",
                    },
                    "invocation": {
                        "start_timestamp": _BASE_TIME + 6.0,
                        "end_timestamp": _BASE_TIME + 8.0,
                    },
                },
            ),
            # Terminal step (no model_name, no tools) — Case 3b
            ATIFStep(
                step_id=5,
                source="agent",
                message="2 to the power of 5 is 32.",
                timestamp="2023-11-14T22:13:29+00:00",
                extra={
                    "ancestry": {
                        "function_id": "fn_react_agent",
                        "function_name": "react_agent",
                        "parent_id": "fn_outer_workflow",
                        "parent_name": "power_of_two_agent",
                    },
                    "invocation": {
                        "start_timestamp": _BASE_TIME + 8.5,
                        "end_timestamp": _BASE_TIME + 9.0,
                    },
                },
            ),
        ],
    )


class TestFullParityTrajectory:
    """Test the full Path A parity structure: workflow → workflow → LLM → tool → LLM."""

    @pytest.fixture
    def exporter(self):
        return ConcreteATIFSpanExporter()

    def test_parity_span_count(self, exporter):
        """Full parity trajectory produces 5 spans: 2 workflow + 2 LLM + 1 tool."""
        trajectory = _build_full_parity_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        assert len(spans) == 5

    def test_parity_span_names(self, exporter):
        """Span names match expected Path A structure."""
        trajectory = _build_full_parity_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        names = [s.name for s in spans]
        assert names == [
            "power_of_two_agent",            # outer workflow
            "react_agent",                   # inner workflow
            "nvidia/nemotron-3-nano-30b-a3b",  # first LLM
            "power_of_two",                  # tool
            "nvidia/nemotron-3-nano-30b-a3b",  # final LLM
        ]

    def test_parity_span_kinds(self, exporter):
        """Span kinds match expected types."""
        trajectory = _build_full_parity_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        kinds = [s.attributes["nat.span.kind"] for s in spans]
        assert kinds == ["WORKFLOW", "WORKFLOW", "LLM", "TOOL", "LLM"]

    def test_inner_workflow_parent(self, exporter):
        """Inner workflow span is a child of the outer workflow span."""
        trajectory = _build_full_parity_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        outer_wf = spans[0]
        inner_wf = spans[1]
        assert outer_wf.parent is None
        assert inner_wf.parent is not None
        assert inner_wf.parent.context.span_id == outer_wf.context.span_id

    def test_first_llm_parent(self, exporter):
        """First LLM span is a child of the inner workflow."""
        trajectory = _build_full_parity_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        inner_wf = spans[1]
        first_llm = spans[2]
        assert first_llm.parent is not None
        assert first_llm.parent.context.span_id == inner_wf.context.span_id

    def test_tool_parent(self, exporter):
        """Tool span is a child of the first LLM span."""
        trajectory = _build_full_parity_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        first_llm = spans[2]
        tool_span = spans[3]
        assert tool_span.parent is not None
        assert tool_span.parent.context.span_id == first_llm.context.span_id

    def test_final_llm_parent(self, exporter):
        """Final LLM span is a child of the inner workflow."""
        trajectory = _build_full_parity_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        inner_wf = spans[1]
        final_llm = spans[4]
        assert final_llm.parent is not None
        assert final_llm.parent.context.span_id == inner_wf.context.span_id

    def test_final_llm_has_output(self, exporter):
        """Final LLM span has the agent's final answer as output."""
        trajectory = _build_full_parity_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        final_llm = spans[4]
        assert SpanAttributes.OUTPUT_VALUE.value in final_llm.attributes

    def test_final_llm_has_token_metrics(self, exporter):
        """Final LLM span has token count attributes."""
        trajectory = _build_full_parity_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        final_llm = spans[4]
        assert final_llm.attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT.value] == 150
        assert final_llm.attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION.value] == 30
        assert final_llm.attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL.value] == 180

    def test_workflow_output_set_from_final_llm(self, exporter):
        """Workflow span output is also set from the final LLM answer."""
        trajectory = _build_full_parity_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        # The inner workflow is the active workflow_span when Case 3a fires,
        # but the terminal step (Case 3b) also sets output on the workflow.
        inner_wf = spans[1]
        assert SpanAttributes.OUTPUT_VALUE.value in inner_wf.attributes

    def test_shared_trace_id(self, exporter):
        """All 5 spans share the same trace_id."""
        trajectory = _build_full_parity_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        trace_ids = {s.context.trace_id for s in spans}
        assert len(trace_ids) == 1

    def test_inner_workflow_end_time_covers_all(self, exporter):
        """Inner workflow end_time is updated to cover the full trajectory."""
        trajectory = _build_full_parity_trajectory()
        spans = exporter._trajectory_to_spans(trajectory, timing_map={})
        inner_wf = spans[1]
        # The latest span end is from the terminal step: _BASE_TIME + 9.0
        assert inner_wf.end_time >= ns_timestamp(_BASE_TIME + 8.0)


class TestEventCollection:
    """Test that export() collects steps and triggers on WORKFLOW_END."""

    @pytest.fixture
    def exporter(self):
        return ConcreteATIFSpanExporter()

    def test_events_collected(self, exporter):
        """Events are collected in _collected_steps."""
        step = _make_step(IntermediateStepType.LLM_START, timestamp_offset=0.0)
        exporter.export(step)
        assert len(exporter._collected_steps) == 1

    def test_non_intermediate_step_ignored(self, exporter):
        """Non-IntermediateStep objects are ignored."""
        exporter.export("not an event")  # type: ignore
        assert len(exporter._collected_steps) == 0

    def test_workflow_end_clears_collected(self, exporter):
        """WORKFLOW_END triggers processing and clears collected steps."""
        wf_start = _make_step(
            IntermediateStepType.WORKFLOW_START,
            function_name="agent",
            function_id="fn_agent",
            input_data="hello",
            timestamp_offset=0.0,
        )
        wf_end = _make_step(
            IntermediateStepType.WORKFLOW_END,
            function_name="agent",
            function_id="fn_agent",
            output_data="goodbye",
            timestamp_offset=5.0,
        )
        exporter.export(wf_start)
        exporter.export(wf_end)
        # After WORKFLOW_END, collected_steps should be cleared
        assert len(exporter._collected_steps) == 0
