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

import pytest

from nat.data_models.atif import ATIFTrajectory
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.intermediate_step import UsageInfo
from nat.data_models.invocation_node import InvocationNode
from nat.data_models.token_usage import TokenUsageBaseModel
from nat.utils.atif_converter import ATIFStreamConverter
from nat.utils.atif_converter import IntermediateStepToATIFConverter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TIME = 1700000000.0


def _make_step(
    event_type: IntermediateStepType,
    *,
    name: str = "test",
    input_data: str | dict | None = None,
    output_data: str | dict | None = None,
    timestamp_offset: float = 0.0,
    parent_id: str = "root",
    function_name: str = "my_workflow",
    usage: UsageInfo | None = None,
    step_uuid: str | None = None,
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
    if event_type.endswith("_END") and event_type != "LLM_NEW_TOKEN":
        payload_kwargs["span_event_timestamp"] = (_BASE_TIME + timestamp_offset - 0.5)
    return IntermediateStep(
        parent_id=parent_id,
        function_ancestry=InvocationNode(
            function_name=function_name,
            function_id="func-id-1",
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

        # No duplicate final step (workflow_end output == last LLM output)
        assert len(result.steps) == 3

    def test_no_tool_trajectory(
        self,
        batch_converter: IntermediateStepToATIFConverter,
        no_tool_trajectory: list[IntermediateStep],
    ):
        """Trajectory without tools produces user + single agent step."""
        result = batch_converter.convert(no_tool_trajectory)

        assert len(result.steps) == 2
        assert result.steps[0].source == "user"
        assert result.steps[0].message == "Say hello"
        assert result.steps[1].source == "agent"
        assert result.steps[1].message == "Hello!"
        assert result.steps[1].tool_calls is None

    def test_multi_tool_single_turn(
        self,
        batch_converter: IntermediateStepToATIFConverter,
        multi_tool_trajectory: list[IntermediateStep],
    ):
        """Multiple tool calls in one LLM turn are grouped correctly."""
        result = batch_converter.convert(multi_tool_trajectory)

        # user + agent(with 2 tools) + final agent
        assert len(result.steps) == 3
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
        assert result.final_metrics.total_steps == 2  # 2 agent steps

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
        for s_step, b_step in zip(stream_result.steps, batch_result.steps):
            assert s_step.source == b_step.source
            assert s_step.message == b_step.message
            if b_step.tool_calls:
                assert len(s_step.tool_calls) == len(b_step.tool_calls)
