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
"""Tests for the ATIF converter."""

import datetime

import pytest

from nat.atif import ATIFTrajectory
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.intermediate_step import UsageInfo
from nat.data_models.invocation_node import InvocationNode
from nat.data_models.token_usage import TokenUsageBaseModel
from nat.utils import atif_converter as atif_converter_module
from nat.utils.atif_converter import ATIFStreamConverter
from nat.utils.atif_converter import IntermediateStepToATIFConverter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TIME = 1700000000.0


def _epoch_to_iso(epoch: float) -> str:
    """Convert Unix epoch to ISO 8601 string for assertions."""
    return datetime.datetime.fromtimestamp(epoch, tz=datetime.UTC).isoformat()


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
        payload_kwargs["span_event_timestamp"] = (_BASE_TIME + timestamp_offset - 0.5)
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


def _make_usage(
    prompt: int = 100,
    completion: int = 50,
    cached: int = 0,
) -> UsageInfo:
    """Create a UsageInfo with token counts."""
    return UsageInfo(
        token_usage=TokenUsageBaseModel(
            prompt_tokens=prompt,
            completion_tokens=completion,
            cached_tokens=cached,
            total_tokens=prompt + completion,
        ),
        num_llm_calls=1,
    )


@pytest.mark.parametrize(
    ("raw_input", "expected"),
    [
        ({
            "input_message": "hello from input_message"
        }, "hello from input_message"),
        ({
            "messages": [
                {
                    "role": "system", "content": "system"
                },
                {
                    "role": "user", "content": "first user"
                },
                {
                    "role": "assistant", "content": "assistant reply"
                },
                {
                    "role": "user", "content": "last user"
                },
            ]
        },
         "last user"),
    ],
)
def test_extract_user_input_chat_variants(raw_input: str | dict, expected: str):
    """User input extraction handles both `input_message` and chat `messages` payloads."""
    assert atif_converter_module._extract_user_input(raw_input) == expected


@pytest.mark.parametrize(
    ("raw_input", "expected"),
    [
        ('{"expression": "2+2"}', {
            "expression": "2+2"
        }),
        ("{'expression': '2+2'}", {
            "expression": "2+2"
        }),
        ("2+2", {
            "input": "2+2"
        }),
        (42, {
            "input": "42"
        }),
    ],
)
def test_parse_tool_arguments_variants(raw_input: str | int, expected: dict[str, str]):
    """Tool argument parsing supports JSON/literal/plain/scalar payload variants."""
    assert atif_converter_module._parse_tool_arguments(raw_input) == expected


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(name="simple_trajectory")
def fixture_simple_trajectory() -> list[IntermediateStep]:
    """A simple trajectory: user query → LLM → tool → LLM → final answer."""
    return [
        _make_step(
            IntermediateStepType.WORKFLOW_START,
            input_data="What is 2+2?",
            timestamp_offset=0.0,
        ),
        _make_step(
            IntermediateStepType.LLM_END,
            name="gpt-4",
            output_data="I need to calculate 2+2",
            timestamp_offset=1.0,
            usage=_make_usage(100, 20),
        ),
        _make_step(
            IntermediateStepType.TOOL_END,
            name="calculator",
            input_data={"expression": "2+2"},
            output_data="4",
            timestamp_offset=2.0,
            step_uuid="tool-uuid-1",
        ),
        _make_step(
            IntermediateStepType.LLM_END,
            name="gpt-4",
            output_data="The answer is 4",
            timestamp_offset=3.0,
            usage=_make_usage(150, 30),
        ),
        _make_step(
            IntermediateStepType.WORKFLOW_END,
            output_data="The answer is 4",
            timestamp_offset=4.0,
        ),
    ]


@pytest.fixture(name="no_tool_trajectory")
def fixture_no_tool_trajectory() -> list[IntermediateStep]:
    """A trajectory with no tool calls."""
    return [
        _make_step(
            IntermediateStepType.WORKFLOW_START,
            input_data="Say hello",
            timestamp_offset=0.0,
        ),
        _make_step(
            IntermediateStepType.LLM_END,
            name="gpt-4",
            output_data="Hello!",
            timestamp_offset=1.0,
            usage=_make_usage(50, 10),
        ),
        _make_step(
            IntermediateStepType.WORKFLOW_END,
            output_data="Hello!",
            timestamp_offset=2.0,
        ),
    ]


@pytest.fixture(name="multi_tool_trajectory")
def fixture_multi_tool_trajectory() -> list[IntermediateStep]:
    """A trajectory where one LLM turn triggers multiple tool calls."""
    return [
        _make_step(
            IntermediateStepType.WORKFLOW_START,
            input_data="Compare GOOG and AAPL prices",
            timestamp_offset=0.0,
        ),
        _make_step(
            IntermediateStepType.LLM_END,
            name="gpt-4",
            output_data="I'll look up both stocks",
            timestamp_offset=1.0,
            usage=_make_usage(100, 25),
        ),
        _make_step(
            IntermediateStepType.TOOL_END,
            name="stock_lookup",
            input_data={"ticker": "GOOG"},
            output_data="GOOG: $185",
            timestamp_offset=2.0,
            step_uuid="tool-goog",
        ),
        _make_step(
            IntermediateStepType.TOOL_END,
            name="stock_lookup",
            input_data={"ticker": "AAPL"},
            output_data="AAPL: $220",
            timestamp_offset=3.0,
            step_uuid="tool-aapl",
        ),
        _make_step(
            IntermediateStepType.LLM_END,
            name="gpt-4",
            output_data="GOOG is $185, AAPL is $220",
            timestamp_offset=4.0,
            usage=_make_usage(200, 40),
        ),
        _make_step(
            IntermediateStepType.WORKFLOW_END,
            output_data="GOOG is $185, AAPL is $220",
            timestamp_offset=5.0,
        ),
    ]


@pytest.fixture(name="batch_converter")
def fixture_batch_converter() -> IntermediateStepToATIFConverter:
    """Create a batch converter instance."""
    return IntermediateStepToATIFConverter()


# ---------------------------------------------------------------------------
# Batch converter tests
# ---------------------------------------------------------------------------


class TestBatchConverter:
    """Tests for IntermediateStepToATIFConverter."""

    def test_empty_steps(self, batch_converter: IntermediateStepToATIFConverter):
        """Empty input produces a trajectory with no steps."""
        result = batch_converter.convert([])
        assert isinstance(result, ATIFTrajectory)
        assert result.steps == []
        assert result.schema_version == "ATIF-v1.6"

    def test_simple_trajectory(
        self,
        batch_converter: IntermediateStepToATIFConverter,
        simple_trajectory: list[IntermediateStep],
    ):
        """Basic workflow with one tool call produces correct ATIF steps."""
        result = batch_converter.convert(simple_trajectory)

        # Step 1: user message
        assert result.steps[0].source == "user"
        assert result.steps[0].message == "What is 2+2?"
        assert result.steps[0].step_id == 1

        # Step 2: agent turn with tool call
        agent_step = result.steps[1]
        assert agent_step.source == "agent"
        assert agent_step.message == "I need to calculate 2+2"
        assert agent_step.tool_calls is not None
        assert len(agent_step.tool_calls) == 1
        assert agent_step.tool_calls[0].function_name == "calculator"
        assert agent_step.tool_calls[0].arguments == {"expression": "2+2"}
        assert agent_step.observation is not None
        assert agent_step.observation.results[0].content == "4"

        # Step 3: final agent response
        assert result.steps[2].source == "agent"
        assert result.steps[2].message == "The answer is 4"
        assert result.steps[2].tool_calls is None

        # Step 4: terminal workflow marker preserving WORKFLOW_END timestamp
        assert result.steps[3].source == "agent"
        assert result.steps[3].message == "The answer is 4"
        assert result.steps[3].tool_calls is None
        assert result.steps[3].timestamp == _epoch_to_iso(_BASE_TIME + 4.0)

        assert len(result.steps) == 4

    def test_no_tool_trajectory(
        self,
        batch_converter: IntermediateStepToATIFConverter,
        no_tool_trajectory: list[IntermediateStep],
    ):
        """Trajectory without tools preserves a terminal workflow marker."""
        result = batch_converter.convert(no_tool_trajectory)

        assert len(result.steps) == 3
        assert result.steps[0].source == "user"
        assert result.steps[0].message == "Say hello"
        assert result.steps[1].source == "agent"
        assert result.steps[1].message == "Hello!"
        assert result.steps[1].tool_calls is None
        assert result.steps[2].source == "agent"
        assert result.steps[2].message == "Hello!"
        assert result.steps[2].timestamp == _epoch_to_iso(_BASE_TIME + 2.0)

    def test_multi_tool_single_turn(
        self,
        batch_converter: IntermediateStepToATIFConverter,
        multi_tool_trajectory: list[IntermediateStep],
    ):
        """Multiple tool calls in one LLM turn are grouped correctly."""
        result = batch_converter.convert(multi_tool_trajectory)

        # user + agent(with 2 tools) + final agent + terminal marker
        assert len(result.steps) == 4
        agent_with_tools = result.steps[1]
        assert len(agent_with_tools.tool_calls) == 2
        assert agent_with_tools.tool_calls[0].function_name == "stock_lookup"
        assert agent_with_tools.tool_calls[1].function_name == "stock_lookup"
        assert len(agent_with_tools.observation.results) == 2

    def test_agent_config_inferred(
        self,
        batch_converter: IntermediateStepToATIFConverter,
        simple_trajectory: list[IntermediateStep],
    ):
        """Agent name and model are inferred from steps."""
        result = batch_converter.convert(simple_trajectory)

        assert result.agent.name == "my_workflow"
        assert result.agent.model_name == "gpt-4"

    def test_agent_name_override(
        self,
        batch_converter: IntermediateStepToATIFConverter,
        simple_trajectory: list[IntermediateStep],
    ):
        """Explicit agent_name overrides the inferred value."""
        result = batch_converter.convert(simple_trajectory, agent_name="custom-agent")
        assert result.agent.name == "custom-agent"

    def test_session_id_override(
        self,
        batch_converter: IntermediateStepToATIFConverter,
        simple_trajectory: list[IntermediateStep],
    ):
        """Explicit session_id is used in the output."""
        result = batch_converter.convert(simple_trajectory, session_id="my-session-123")
        assert result.session_id == "my-session-123"

    def test_framework_in_extra(
        self,
        batch_converter: IntermediateStepToATIFConverter,
    ):
        """Framework is included in invocation metadata when present."""
        steps = [
            _make_step(
                IntermediateStepType.WORKFLOW_START,
                input_data="Hi",
                timestamp_offset=0.0,
            ),
            _make_step(
                IntermediateStepType.LLM_END,
                name="gpt-4",
                output_data="Hello!",
                timestamp_offset=1.0,
                usage=_make_usage(50, 10),
                framework=LLMFrameworkEnum.LANGCHAIN,
            ),
            _make_step(
                IntermediateStepType.WORKFLOW_END,
                output_data="Hello!",
                timestamp_offset=2.0,
            ),
        ]
        result = batch_converter.convert(steps)
        agent_step = result.steps[1]
        assert agent_step.extra is not None
        assert agent_step.extra["invocation"]["framework"] == "langchain"

    def test_final_metrics(
        self,
        batch_converter: IntermediateStepToATIFConverter,
        simple_trajectory: list[IntermediateStep],
    ):
        """Final metrics aggregate token usage across LLM steps."""
        result = batch_converter.convert(simple_trajectory)

        assert result.final_metrics is not None
        assert result.final_metrics.total_prompt_tokens == 250  # 100 + 150
        assert result.final_metrics.total_completion_tokens == 50  # 20 + 30
        assert result.final_metrics.total_steps == 3  # 2 agent turns + terminal workflow marker

    def test_timestamps_are_iso(
        self,
        batch_converter: IntermediateStepToATIFConverter,
        simple_trajectory: list[IntermediateStep],
    ):
        """All timestamps are valid ISO 8601 strings."""
        result = batch_converter.convert(simple_trajectory)
        for step in result.steps:
            if step.timestamp:
                assert "T" in step.timestamp
                assert "+" in step.timestamp or "Z" in step.timestamp

    def test_step_ids_sequential(
        self,
        batch_converter: IntermediateStepToATIFConverter,
        simple_trajectory: list[IntermediateStep],
    ):
        """Step IDs are sequential starting from 1."""
        result = batch_converter.convert(simple_trajectory)
        ids = [s.step_id for s in result.steps]
        assert ids == list(range(1, len(ids) + 1))

    def test_serialization_roundtrip(
        self,
        batch_converter: IntermediateStepToATIFConverter,
        simple_trajectory: list[IntermediateStep],
    ):
        """Trajectory can be serialized to JSON and back."""
        result = batch_converter.convert(simple_trajectory)
        json_str = result.model_dump_json(exclude_none=True)
        restored = ATIFTrajectory.model_validate_json(json_str)
        assert len(restored.steps) == len(result.steps)
        assert restored.schema_version == "ATIF-v1.6"

    def test_metrics_include_reasoning_tokens(
        self,
        batch_converter: IntermediateStepToATIFConverter,
    ):
        """reasoning_tokens from UsageInfo is mapped to metrics.extra."""
        steps = [
            _make_step(
                IntermediateStepType.WORKFLOW_START,
                input_data="Hi",
                timestamp_offset=0.0,
            ),
            _make_step(
                IntermediateStepType.LLM_END,
                name="gpt-4",
                output_data="Hello",
                timestamp_offset=1.0,
                usage=UsageInfo(token_usage=TokenUsageBaseModel(
                    prompt_tokens=10,
                    completion_tokens=5,
                    reasoning_tokens=100,
                    total_tokens=115,
                ), ),
            ),
            _make_step(
                IntermediateStepType.WORKFLOW_END,
                output_data="Hello",
                timestamp_offset=2.0,
            ),
        ]
        result = batch_converter.convert(steps)
        agent_step = result.steps[1]
        assert agent_step.metrics is not None
        assert agent_step.metrics.extra is not None
        assert agent_step.metrics.extra.get("reasoning_tokens") == 100

    def test_metrics_include_cached_tokens(
        self,
        batch_converter: IntermediateStepToATIFConverter,
    ):
        """cached_tokens from UsageInfo is mapped to metrics."""
        steps = [
            _make_step(
                IntermediateStepType.WORKFLOW_START,
                input_data="Hi",
                timestamp_offset=0.0,
            ),
            _make_step(
                IntermediateStepType.LLM_END,
                name="gpt-4",
                output_data="Hello",
                timestamp_offset=1.0,
                usage=UsageInfo(token_usage=TokenUsageBaseModel(
                    prompt_tokens=100,
                    completion_tokens=20,
                    cached_tokens=50,
                    total_tokens=120,
                ), ),
            ),
            _make_step(
                IntermediateStepType.WORKFLOW_END,
                output_data="Hello",
                timestamp_offset=2.0,
            ),
        ]
        result = batch_converter.convert(steps)
        agent_step = result.steps[1]
        assert agent_step.metrics is not None
        assert agent_step.metrics.cached_tokens == 50
        assert result.final_metrics is not None
        assert result.final_metrics.total_cached_tokens == 50

    def test_tool_call_id_and_observation_source_match(
        self,
        batch_converter: IntermediateStepToATIFConverter,
        simple_trajectory: list[IntermediateStep],
    ):
        """tool_call_id and observation source_call_id are linked correctly."""
        result = batch_converter.convert(simple_trajectory)
        agent_with_tools = result.steps[1]
        assert agent_with_tools.tool_calls is not None
        assert agent_with_tools.observation is not None
        for tc, obs in zip(agent_with_tools.tool_calls, agent_with_tools.observation.results, strict=True):
            assert obs.source_call_id == tc.tool_call_id
            assert tc.tool_call_id.startswith("call_")

    def test_profiling_extra_populated(
        self,
        batch_converter: IntermediateStepToATIFConverter,
        simple_trajectory: list[IntermediateStep],
    ):
        """step.extra contains flat ancestry metadata for profiling."""
        result = batch_converter.convert(simple_trajectory)

        # User step has profiling extra
        user_step = result.steps[0]
        assert user_step.extra is not None
        assert user_step.extra["ancestry"]["function_id"] == "func-id-1"
        assert user_step.extra["ancestry"]["function_name"] == "my_workflow"

        # Agent step with tool call has tool_ancestry (from TOOL_END)
        agent_step = result.steps[1]
        assert agent_step.extra is not None
        assert agent_step.extra["ancestry"]["function_id"] == "func-id-1"
        assert agent_step.extra.get("tool_ancestry") is not None
        assert len(agent_step.extra["tool_ancestry"]) == 1
        assert agent_step.extra["tool_ancestry"][0]["function_id"] == "func-id-1"
        assert agent_step.extra["tool_ancestry"][0]["function_name"] == "my_workflow"

    def test_nested_tool_ancestry_is_populated(self, batch_converter: IntermediateStepToATIFConverter):
        """Nested lineage is represented through canonical `tool_ancestry`."""
        steps = [
            _make_step(
                IntermediateStepType.WORKFLOW_START,
                input_data="What is 2^4?",
                timestamp_offset=0.0,
                function_name="root",
                function_id="root",
            ),
            _make_step(
                IntermediateStepType.LLM_END,
                name="gpt-4",
                output_data="I'll call power_of_two",
                timestamp_offset=1.0,
                usage=_make_usage(10, 5),
                function_name="react_agent",
                function_id="wf-1",
                function_parent_id="root",
                function_parent_name="root",
            ),
            _make_step(
                IntermediateStepType.TOOL_END,
                name="calculator__multiply",
                input_data={
                    "a": 4, "b": 4
                },
                output_data="16",
                timestamp_offset=2.0,
                step_uuid="tool-uuid-rich-path",
                function_name="calculator__multiply",
                function_id="fn-1",
                function_parent_id="wf-1",
                function_parent_name="react_agent",
            ),
            _make_step(
                IntermediateStepType.WORKFLOW_END,
                output_data="16",
                timestamp_offset=3.0,
                function_name="react_agent",
                function_id="wf-1",
                function_parent_id="root",
                function_parent_name="root",
            ),
        ]
        result = batch_converter.convert(steps)
        agent_step = result.steps[1]
        assert agent_step.extra is not None
        assert agent_step.extra["ancestry"]["function_id"] == "wf-1"
        assert agent_step.extra.get("tool_ancestry") is not None
        assert len(agent_step.extra["tool_ancestry"]) == 1
        assert agent_step.extra["tool_ancestry"][0]["function_id"] == "fn-1"
        assert agent_step.extra["tool_ancestry"][0]["parent_id"] == "wf-1"

    def test_tool_ancestry_includes_nested_internal_functions(self, batch_converter: IntermediateStepToATIFConverter):
        """Nested internal function lineage is encoded in canonical `tool_ancestry`."""
        steps = [
            _make_step(
                IntermediateStepType.WORKFLOW_START,
                input_data="What is 3^2?",
                timestamp_offset=0.0,
                function_name="<workflow>",
                function_id="wf-1",
                function_parent_id="root",
                function_parent_name="root",
            ),
            _make_step(
                IntermediateStepType.LLM_END,
                name="gpt-4",
                output_data="Call power_of_two",
                timestamp_offset=1.0,
                usage=_make_usage(10, 5),
                function_name="<workflow>",
                function_id="wf-1",
                function_parent_id="root",
                function_parent_name="root",
            ),
            _make_step(
                IntermediateStepType.TOOL_END,
                name="power_of_two",
                input_data={"number": 3},
                output_data="9",
                timestamp_offset=2.0,
                function_name="<workflow>",
                function_id="wf-1",
                function_parent_id="root",
                function_parent_name="root",
                step_uuid="tool-power-uuid",
            ),
            _make_step(
                IntermediateStepType.FUNCTION_END,
                name="calculator__multiply",
                output_data="9",
                timestamp_offset=2.1,
                function_name="calculator__multiply",
                function_id="fn-mul",
                function_parent_id="fn-power",
                function_parent_name="power_of_two",
            ),
            _make_step(
                IntermediateStepType.WORKFLOW_END,
                output_data="9",
                timestamp_offset=3.0,
                function_name="<workflow>",
                function_id="wf-1",
                function_parent_id="root",
                function_parent_name="root",
            ),
        ]
        result = batch_converter.convert(steps)
        agent_step = result.steps[1]
        assert agent_step.tool_calls is not None
        assert agent_step.tool_calls[0].function_name == "power_of_two"
        assert agent_step.extra is not None
        assert agent_step.extra.get("tool_ancestry") is not None
        tool_fn = next(entry for entry in agent_step.extra["tool_ancestry"]
                       if entry["function_name"] == "calculator__multiply")
        assert tool_fn["function_id"] == "fn-mul"
        assert tool_fn["function_name"] == "calculator__multiply"
        assert tool_fn["parent_id"] == "fn-power"
        assert tool_fn["parent_name"] == "power_of_two"

    def test_observed_invocations_ordered_by_span_start(self, batch_converter: IntermediateStepToATIFConverter):
        """Observed invocations are ordered by span start, not end arrival."""
        steps = [
            _make_step(
                IntermediateStepType.WORKFLOW_START,
                input_data="Run branch calls",
                timestamp_offset=0.0,
                function_name="<workflow>",
                function_id="wf-1",
                function_parent_id="root",
                function_parent_name="root",
            ),
            _make_step(
                IntermediateStepType.LLM_END,
                name="gpt-4",
                output_data="Calling branch",
                timestamp_offset=1.0,
                usage=_make_usage(10, 5),
                function_name="<workflow>",
                function_id="wf-1",
                function_parent_id="root",
                function_parent_name="root",
            ),
            _make_step(
                IntermediateStepType.FUNCTION_END,
                name="branch_b_tool",
                output_data="B",
                timestamp_offset=2.0,  # ends first
                function_name="branch_b_tool",
                function_id="fn-b",
                function_parent_id="wf-1",
                function_parent_name="<workflow>",
            ),
            _make_step(
                IntermediateStepType.FUNCTION_END,
                name="branch_a_tool",
                output_data="A",
                timestamp_offset=2.1,  # ends later
                function_name="branch_a_tool",
                function_id="fn-a",
                function_parent_id="wf-1",
                function_parent_name="<workflow>",
            ),
            _make_step(
                IntermediateStepType.WORKFLOW_END,
                output_data="done",
                timestamp_offset=3.0,
                function_name="<workflow>",
                function_id="wf-1",
                function_parent_id="root",
                function_parent_name="root",
            ),
        ]
        # Force start-time ordering opposite to end-time ordering.
        steps[2].payload.span_event_timestamp = _BASE_TIME + 1.9  # branch_b starts later
        steps[3].payload.span_event_timestamp = _BASE_TIME + 1.2  # branch_a starts earlier

        result = batch_converter.convert(steps)
        agent_step = result.steps[1]
        assert agent_step.tool_calls is not None
        assert [tc.function_name for tc in agent_step.tool_calls] == ["branch_a_tool", "branch_b_tool"]

    def test_synthetic_workflow_function_end_is_suppressed(self, batch_converter: IntermediateStepToATIFConverter):
        """Synthetic `<workflow>` function-end calls are not exported as tool calls."""
        steps = [
            _make_step(
                IntermediateStepType.WORKFLOW_START,
                input_data="What is 2+2?",
                timestamp_offset=0.0,
                function_name="<workflow>",
                function_id="wf-1",
                function_parent_id="root",
                function_parent_name="root",
            ),
            _make_step(
                IntermediateStepType.LLM_END,
                name="gpt-4",
                output_data="Answering",
                timestamp_offset=1.0,
                usage=_make_usage(10, 5),
                function_name="<workflow>",
                function_id="wf-1",
                function_parent_id="root",
                function_parent_name="root",
            ),
            _make_step(
                IntermediateStepType.FUNCTION_END,
                name="<workflow>",
                output_data="4",
                timestamp_offset=1.1,
                function_name="<workflow>",
                function_id="wf-1",
                function_parent_id="root",
                function_parent_name="root",
            ),
            _make_step(
                IntermediateStepType.WORKFLOW_END,
                output_data="4",
                timestamp_offset=2.0,
                function_name="<workflow>",
                function_id="wf-1",
                function_parent_id="root",
                function_parent_name="root",
            ),
        ]

        result = batch_converter.convert(steps)
        agent_step = result.steps[1]
        assert agent_step.tool_calls is None
        assert agent_step.observation is None
        assert agent_step.extra is not None
        assert agent_step.extra.get("tool_ancestry") == []
        assert agent_step.extra.get("tool_invocations") is None

    def test_agent_tool_definitions_populated(
        self,
        batch_converter: IntermediateStepToATIFConverter,
    ):
        """tool_definitions from TraceMetadata.tools_schema is mapped to agent."""
        from nat.data_models.intermediate_step import ToolDetails
        from nat.data_models.intermediate_step import ToolParameters
        from nat.data_models.intermediate_step import ToolSchema
        from nat.data_models.intermediate_step import TraceMetadata

        steps = [
            _make_step(
                IntermediateStepType.WORKFLOW_START,
                input_data="Hi",
                timestamp_offset=0.0,
            ),
            _make_step(
                IntermediateStepType.LLM_END,
                name="gpt-4",
                output_data="I'll use a tool",
                timestamp_offset=1.0,
                usage=_make_usage(50, 10),
            ),
            _make_step(
                IntermediateStepType.WORKFLOW_END,
                output_data="I'll use a tool",
                timestamp_offset=2.0,
            ),
        ]
        tool_schema = ToolSchema(
            type="function",
            function=ToolDetails(
                name="weather",
                description="Get weather",
                parameters=ToolParameters(properties={}),
            ),
        )
        steps[1].payload.metadata = TraceMetadata(tools_schema=[tool_schema])
        result = batch_converter.convert(steps)
        assert result.agent.tool_definitions is not None
        assert len(result.agent.tool_definitions) == 1
        assert result.agent.tool_definitions[0]["function"]["name"] == "weather"

    @pytest.mark.parametrize("event_type", [IntermediateStepType.TOOL_END, IntermediateStepType.FUNCTION_END])
    def test_batch_converter_emits_orphan_invocation_step(
        self,
        batch_converter: IntermediateStepToATIFConverter,
        event_type: IntermediateStepType,
    ):
        """Orphan tool/function end events are emitted as standalone agent tool steps."""
        steps = [
            _make_step(
                IntermediateStepType.WORKFLOW_START,
                input_data="Run one orphan call",
                timestamp_offset=0.0,
            ),
            _make_step(
                event_type,
                name="calculator",
                input_data={"expression": "2+2"},
                output_data="4",
                timestamp_offset=1.0,
                step_uuid="orphan-tool-1",
                function_name="calculator",
                function_id="fn-calc",
                function_parent_id="wf-1",
                function_parent_name="workflow",
            ),
            _make_step(
                IntermediateStepType.WORKFLOW_END,
                output_data="done",
                timestamp_offset=2.0,
            ),
        ]
        result = batch_converter.convert(steps)

        orphan_step = result.steps[1]
        assert orphan_step.source == "agent"
        assert orphan_step.message == ""
        assert orphan_step.tool_calls is not None
        assert len(orphan_step.tool_calls) == 1
        assert orphan_step.observation is not None
        assert len(orphan_step.observation.results) == 1
        assert orphan_step.observation.results[0].source_call_id == orphan_step.tool_calls[0].tool_call_id
        assert orphan_step.extra is not None
        assert len(orphan_step.extra["tool_ancestry"]) == 1
        assert len(orphan_step.extra["tool_invocations"]) == 1
        assert orphan_step.extra["tool_invocations"][0]["invocation_id"] == orphan_step.tool_calls[0].tool_call_id

    def test_converter_ignores_non_exported_events(self, batch_converter: IntermediateStepToATIFConverter):
        """Non-exported start/chunk/unsupported-end events do not produce ATIF steps."""
        steps = [
            _make_step(
                IntermediateStepType.WORKFLOW_START,
                input_data="hello",
                timestamp_offset=0.0,
            ),
            _make_step(
                IntermediateStepType.TOOL_START,
                name="calculator",
                timestamp_offset=0.5,
            ),
            _make_step(
                IntermediateStepType.LLM_NEW_TOKEN,
                output_data="tok",
                timestamp_offset=0.6,
            ),
            _make_step(
                IntermediateStepType.SPAN_CHUNK,
                output_data="chunk",
                timestamp_offset=0.7,
            ),
            _make_step(
                IntermediateStepType.TASK_END,
                output_data="ignored",
                timestamp_offset=0.8,
            ),
            _make_step(
                IntermediateStepType.WORKFLOW_END,
                output_data="final",
                timestamp_offset=1.0,
            ),
        ]
        result = batch_converter.convert(steps)
        assert len(result.steps) == 2
        assert result.steps[0].source == "user"
        assert result.steps[0].message == "hello"
        assert result.steps[1].source == "agent"
        assert result.steps[1].message == "final"
        assert result.steps[1].tool_calls is None

    def test_implicit_subagent_delegation_is_disabled_by_default(self):
        """Implicit delegation inference is opt-in and disabled by default."""
        converter = IntermediateStepToATIFConverter()
        steps = [
            _make_step(IntermediateStepType.WORKFLOW_START, input_data="q", timestamp_offset=0.0),
            _make_step(IntermediateStepType.LLM_END, output_data="thinking", timestamp_offset=1.0),
            _make_step(
                IntermediateStepType.FUNCTION_END,
                name="child_agent",
                timestamp_offset=2.0,
                function_name="parent_agent",
                function_id="wrapper-fn",
                function_parent_id="root",
                step_uuid="wrapper-step",
            ),
            _make_step(
                IntermediateStepType.FUNCTION_END,
                name="inner_tool",
                timestamp_offset=3.0,
                function_name="child_agent",
                function_id="child-fn",
                function_parent_id="wrapper-fn",
                step_uuid="child-step",
            ),
            _make_step(IntermediateStepType.LLM_END, output_data="done", timestamp_offset=4.0),
            _make_step(IntermediateStepType.WORKFLOW_END, output_data="done", timestamp_offset=5.0),
        ]

        result = converter.convert(steps)
        first_agent_turn = result.steps[1]
        assert first_agent_turn.observation is not None
        assert first_agent_turn.observation.results[0].subagent_trajectory_ref is None

    def test_implicit_subagent_delegation_can_be_enabled(self):
        """Implicit delegation inference can be enabled explicitly."""
        converter = IntermediateStepToATIFConverter(allow_implicit_subagent_delegation=True)
        steps = [
            _make_step(IntermediateStepType.WORKFLOW_START, input_data="q", timestamp_offset=0.0),
            _make_step(IntermediateStepType.LLM_END, output_data="thinking", timestamp_offset=1.0),
            _make_step(
                IntermediateStepType.FUNCTION_END,
                name="child_agent",
                timestamp_offset=2.0,
                function_name="parent_agent",
                function_id="wrapper-fn",
                function_parent_id="root",
                step_uuid="wrapper-step",
            ),
            _make_step(
                IntermediateStepType.FUNCTION_END,
                name="inner_tool",
                timestamp_offset=3.0,
                function_name="child_agent",
                function_id="child-fn",
                function_parent_id="wrapper-fn",
                step_uuid="child-step",
            ),
            _make_step(IntermediateStepType.LLM_END, output_data="done", timestamp_offset=4.0),
            _make_step(IntermediateStepType.WORKFLOW_END, output_data="done", timestamp_offset=5.0),
        ]

        result = converter.convert(steps)
        first_agent_turn = result.steps[1]
        assert first_agent_turn.observation is not None
        refs = first_agent_turn.observation.results[0].subagent_trajectory_ref
        assert refs is not None
        assert len(refs) == 1


# ---------------------------------------------------------------------------
# Stream converter tests
# ---------------------------------------------------------------------------


class TestStreamConverter:
    """Tests for ATIFStreamConverter."""

    def test_workflow_start_emits_user_step(self):
        """WORKFLOW_START produces an immediate user step."""
        converter = ATIFStreamConverter()
        step = _make_step(
            IntermediateStepType.WORKFLOW_START,
            input_data="hello",
            timestamp_offset=0.0,
        )
        result = converter.push(step)
        assert result is not None
        assert result.source == "user"
        assert result.message == "hello"

    def test_llm_end_flushes_previous_turn(self):
        """Second LLM_END flushes the first turn."""
        converter = ATIFStreamConverter()
        converter.push(_make_step(
            IntermediateStepType.WORKFLOW_START,
            input_data="q",
            timestamp_offset=0.0,
        ))
        # First LLM_END → creates pending, nothing to flush yet
        result1 = converter.push(
            _make_step(
                IntermediateStepType.LLM_END,
                name="gpt-4",
                output_data="thinking...",
                timestamp_offset=1.0,
            ))
        assert result1 is None  # Nothing flushed yet

        # Second LLM_END → flushes the first turn
        result2 = converter.push(
            _make_step(
                IntermediateStepType.LLM_END,
                name="gpt-4",
                output_data="done",
                timestamp_offset=2.0,
            ))
        assert result2 is not None
        assert result2.source == "agent"
        assert result2.message == "thinking..."

    def test_tool_end_attaches_to_pending(self):
        """TOOL_END attaches to the current pending agent turn."""
        converter = ATIFStreamConverter()
        converter.push(_make_step(
            IntermediateStepType.WORKFLOW_START,
            input_data="q",
            timestamp_offset=0.0,
        ))
        converter.push(_make_step(
            IntermediateStepType.LLM_END,
            output_data="let me search",
            timestamp_offset=1.0,
        ))
        result = converter.push(
            _make_step(
                IntermediateStepType.TOOL_END,
                name="search",
                input_data={"query": "test"},
                output_data="found it",
                timestamp_offset=2.0,
                step_uuid="tool-1",
            ))
        # Tool attaches to pending, doesn't emit yet
        assert result is None

        # Finalize flushes
        remaining = converter.finalize()
        assert len(remaining) == 1
        flushed = remaining[0]
        assert flushed.tool_calls is not None
        assert len(flushed.tool_calls) == 1
        assert flushed.tool_calls[0].function_name == "search"
        assert flushed.observation.results[0].content == "found it"

    def test_stream_converter_emits_orphan_tool_end(self):
        """Orphan `TOOL_END` emits an immediate standalone agent step."""
        converter = ATIFStreamConverter()
        converter.push(_make_step(
            IntermediateStepType.WORKFLOW_START,
            input_data="q",
            timestamp_offset=0.0,
        ))
        orphan = converter.push(
            _make_step(
                IntermediateStepType.TOOL_END,
                name="search",
                input_data='{"query": "orphan"}',
                output_data="found orphan",
                timestamp_offset=1.0,
                step_uuid="stream-orphan-tool-1",
            ))
        assert orphan is not None
        assert orphan.source == "agent"
        assert orphan.message == ""
        assert orphan.tool_calls is not None
        assert len(orphan.tool_calls) == 1
        assert orphan.observation is not None
        assert len(orphan.observation.results) == 1
        assert orphan.observation.results[0].source_call_id == orphan.tool_calls[0].tool_call_id

    def test_stream_converter_populates_tool_definitions_from_llm_metadata(self):
        """`LLM_END` metadata tool schemas populate stream converter agent config."""
        from nat.data_models.intermediate_step import ToolDetails
        from nat.data_models.intermediate_step import ToolParameters
        from nat.data_models.intermediate_step import ToolSchema
        from nat.data_models.intermediate_step import TraceMetadata

        converter = ATIFStreamConverter()
        converter.push(_make_step(
            IntermediateStepType.WORKFLOW_START,
            input_data="q",
            timestamp_offset=0.0,
        ))
        llm_end = _make_step(
            IntermediateStepType.LLM_END,
            name="gpt-4",
            output_data="using tools",
            timestamp_offset=1.0,
        )
        llm_end.payload.metadata = TraceMetadata(tools_schema=[
            ToolSchema(
                type="function",
                function=ToolDetails(
                    name="weather",
                    description="Get weather",
                    parameters=ToolParameters(properties={}),
                ),
            )
        ])

        pushed = converter.push(llm_end)
        assert pushed is None
        assert converter.agent_config.tool_definitions is not None
        assert len(converter.agent_config.tool_definitions) == 1
        assert converter.agent_config.tool_definitions[0]["function"]["name"] == "weather"

    def test_finalize_flushes_pending(self):
        """finalize() returns any remaining pending turn."""
        converter = ATIFStreamConverter()
        converter.push(_make_step(
            IntermediateStepType.WORKFLOW_START,
            input_data="q",
            timestamp_offset=0.0,
        ))
        converter.push(_make_step(
            IntermediateStepType.LLM_END,
            output_data="answer",
            timestamp_offset=1.0,
        ))
        remaining = converter.finalize()
        assert len(remaining) == 1
        assert remaining[0].message == "answer"

    def test_finalize_empty_when_nothing_pending(self):
        """finalize() returns empty list if no pending turn."""
        converter = ATIFStreamConverter()
        assert converter.finalize() == []

    def test_get_trajectory_builds_complete(
        self,
        simple_trajectory: list[IntermediateStep],
    ):
        """get_trajectory() returns a complete trajectory after all steps."""
        converter = ATIFStreamConverter()
        for ist in simple_trajectory:
            converter.push(ist)
        converter.finalize()
        trajectory = converter.get_trajectory()

        assert isinstance(trajectory, ATIFTrajectory)
        assert trajectory.schema_version == "ATIF-v1.6"
        assert len(trajectory.steps) >= 2
        assert trajectory.steps[0].source == "user"

    def test_stream_matches_batch(
        self,
        simple_trajectory: list[IntermediateStep],
        batch_converter: IntermediateStepToATIFConverter,
    ):
        """Stream converter produces the same steps as batch converter."""
        batch_result = batch_converter.convert(simple_trajectory, session_id="test")

        stream_conv = ATIFStreamConverter()
        for ist in simple_trajectory:
            stream_conv.push(ist)
        stream_conv.finalize()
        stream_result = stream_conv.get_trajectory()

        assert len(stream_result.steps) == len(batch_result.steps)
        for s_step, b_step in zip(stream_result.steps, batch_result.steps, strict=True):
            assert s_step.source == b_step.source
            assert s_step.message == b_step.message
            if b_step.tool_calls:
                assert len(s_step.tool_calls) == len(b_step.tool_calls)
