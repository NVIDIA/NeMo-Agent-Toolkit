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

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from nat.data_models.api_server import ResponseIntermediateStep
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.invocation_node import InvocationNode
from nat.data_models.step_adaptor import StepAdaptorConfig
from nat.data_models.step_adaptor import StepAdaptorMode
from nat.front_ends.fastapi.step_adaptor import StepAdaptor


@pytest.fixture(name="default_config")
def default_config_fixture() -> StepAdaptorConfig:
    """Return a default StepAdaptorConfig object (mode=DEFAULT)."""
    return StepAdaptorConfig(mode=StepAdaptorMode.DEFAULT, custom_event_types=[])


@pytest.fixture(name="custom_config")
def custom_config_fixture() -> StepAdaptorConfig:
    """Return a custom StepAdaptorConfig object (mode=CUSTOM) with custom types."""
    return StepAdaptorConfig(
        mode=StepAdaptorMode.CUSTOM,
        custom_event_types=[
            IntermediateStepType.CUSTOM_START,
            IntermediateStepType.CUSTOM_END,
        ],
    )


@pytest.fixture(name="disabled_config")
def disabled_config_fixture() -> StepAdaptorConfig:
    """Return a custom StepAdaptorConfig object that disables intermediate steps."""
    return StepAdaptorConfig(
        mode=StepAdaptorMode.OFF,
        custom_event_types=[
            IntermediateStepType.CUSTOM_START,
            IntermediateStepType.CUSTOM_END,
        ],
    )


@pytest.fixture(name="step_adaptor_default")
def step_adaptor_default_fixture(default_config: StepAdaptorConfig) -> StepAdaptor:
    """Return a StepAdaptor using the default config."""
    return StepAdaptor(config=default_config)


@pytest.fixture(name="step_adaptor_custom")
def step_adaptor_custom_fixture(custom_config: StepAdaptorConfig) -> StepAdaptor:
    """Return a StepAdaptor using the custom config."""
    return StepAdaptor(config=custom_config)


@pytest.fixture(name="step_adaptor_disabled")
def step_adaptor_disabled_fixture(disabled_config: StepAdaptorConfig) -> StepAdaptor:
    """Return a StepAdaptor using the disabled config."""
    return StepAdaptor(config=disabled_config)


@pytest.fixture(name="make_intermediate_step")
def make_intermediate_step_fixture() -> Callable[..., IntermediateStep]:
    """A factory fixture to create an IntermediateStep with minimal defaults."""

    def _make_step(
        event_type: IntermediateStepType,
        data_input: Any = None,
        data_output: Any = None,
        name: str | None = None,
        UUID: str | None = None,
    ) -> IntermediateStep:
        payload = IntermediateStepPayload(
            event_type=event_type,
            name=name or "test_step",
            data=StreamEventData(input=data_input, output=data_output),
            UUID=UUID or "test-uuid-1234",
        )
        # The IntermediateStep constructor requires a function_ancestry,
        # but for testing we can just pass None or a placeholder.
        return IntermediateStep(
            parent_id="root",
            function_ancestry=InvocationNode(parent_id="abc", function_id="def", function_name="xyz"),
            payload=payload,
        )

    return _make_step


# --------------------
# Tests for DEFAULT mode
# --------------------
@pytest.mark.parametrize(
    "event_type, expected_result",
    [
        (IntermediateStepType.LLM_START, True),
        (IntermediateStepType.LLM_NEW_TOKEN, True),
        (IntermediateStepType.LLM_END, True),
    ],
)
def test_process_llm_events_in_default(
    step_adaptor_default: StepAdaptor,
    make_intermediate_step: Callable[..., IntermediateStep],
    event_type: IntermediateStepType,
    expected_result: bool,
) -> None:
    """
    In DEFAULT mode with stream_llm_tokens=True (default):
    - LLM_START returns a valid ResponseIntermediateStep.
    - LLM_NEW_TOKEN returns a valid ResponseIntermediateStep.
    - LLM_END returns a valid ResponseIntermediateStep.
    """
    step_start = make_intermediate_step(
        event_type=IntermediateStepType.LLM_START,
        data_input="LLM Input",
        UUID="test-default-llm-uuid",
    )

    if event_type == IntermediateStepType.LLM_START:
        step = step_start
    elif event_type == IntermediateStepType.LLM_NEW_TOKEN:
        step_adaptor_default.process(step_start)
        step = make_intermediate_step(
            event_type=IntermediateStepType.LLM_NEW_TOKEN,
            data_output="chunk",
            UUID="test-default-llm-uuid",
        )
    else:
        step_adaptor_default.process(step_start)
        step = make_intermediate_step(
            event_type=IntermediateStepType.LLM_END,
            data_output="LLM Output",
            UUID="test-default-llm-uuid",
        )

    result = step_adaptor_default.process(step)

    if expected_result:
        assert result is not None, f"Expected LLM event '{event_type}' to be processed in DEFAULT mode."
        assert isinstance(result, ResponseIntermediateStep)
    else:
        assert result is None, f"Expected LLM event '{event_type}' to be filtered out in DEFAULT mode."


def test_process_tool_in_default(
    step_adaptor_default: StepAdaptor,
    make_intermediate_step: Callable[..., IntermediateStep],
) -> None:
    """
    In DEFAULT mode, TOOL_START tracks pending start and TOOL_END evicts it.
    """
    uuid = "test-tool-uuid"
    step = make_intermediate_step(
        event_type=IntermediateStepType.TOOL_START,
        data_input="Tool Input Data",
        data_output="Tool Output Data",
        UUID=uuid,
    )

    result = step_adaptor_default.process(step)

    assert result is not None, "Expected TOOL_START event to be processed in DEFAULT mode."
    assert isinstance(result, ResponseIntermediateStep)
    assert "Tool:" in result.name
    assert "Input:" in result.payload
    assert uuid in step_adaptor_default._tool_starts

    step = make_intermediate_step(
        event_type=IntermediateStepType.TOOL_END,
        data_input="Tool Input Data",
        data_output="Tool Output Data",
        UUID=uuid,
    )

    result = step_adaptor_default.process(step)

    assert result is not None, "Expected TOOL_END event to be processed in DEFAULT mode."
    assert isinstance(result, ResponseIntermediateStep)
    assert "Tool:" in result.name
    assert "Input:" in result.payload
    assert "Output:" in result.payload
    assert uuid not in step_adaptor_default._tool_starts


@pytest.mark.parametrize("event_type",
                         [
                             (IntermediateStepType.WORKFLOW_START),
                             (IntermediateStepType.WORKFLOW_END),
                             (IntermediateStepType.CUSTOM_START),
                             (IntermediateStepType.CUSTOM_END),
                         ])
def test_process_other_events_in_default_returns_none(
    step_adaptor_default: StepAdaptor,
    make_intermediate_step: Callable[..., IntermediateStep],
    event_type: IntermediateStepType,
) -> None:
    """
    In DEFAULT mode, anything other than LLM, TOOL, or FUNCTION should return None.
    """
    step = make_intermediate_step(event_type=event_type)
    result = step_adaptor_default.process(step)

    assert result is None, f"Expected event {event_type} to be ignored in DEFAULT mode."


# --------------------
# Tests for CUSTOM mode
# --------------------
def test_process_custom_events_in_custom_mode(
    step_adaptor_custom: StepAdaptor,
    make_intermediate_step: Callable[..., IntermediateStep],
) -> None:
    """
    In CUSTOM mode with custom_event_types = [CUSTOM_START, CUSTOM_END],
    only those events should produce output.
    """
    # Should be processed
    step_start = make_intermediate_step(event_type=IntermediateStepType.CUSTOM_START)
    step_end = make_intermediate_step(event_type=IntermediateStepType.CUSTOM_END)

    # Should be ignored
    step_llm = make_intermediate_step(event_type=IntermediateStepType.LLM_END, data_output="LLM Output")
    step_tool = make_intermediate_step(event_type=IntermediateStepType.TOOL_END, data_output="Tool Output")

    result_start = step_adaptor_custom.process(step_start)
    result_end = step_adaptor_custom.process(step_end)
    result_llm = step_adaptor_custom.process(step_llm)
    result_tool = step_adaptor_custom.process(step_tool)

    # Validate the custom events produce an ResponseIntermediateStep
    assert result_start is not None
    assert isinstance(result_start, ResponseIntermediateStep)
    assert result_end is not None
    assert isinstance(result_end, ResponseIntermediateStep)

    # Validate we do not process LLM or TOOL_END in custom mode (with given custom_event_types)
    assert result_llm is None
    assert result_tool is None


def test_process_custom_events_empty_list(
    step_adaptor_custom: StepAdaptor,
    make_intermediate_step: Callable[..., IntermediateStep],
) -> None:
    """
    If the StepAdaptorConfig was set to CUSTOM but had an empty or non-matching
    custom_event_types, we expect no events to be processed. (In the fixture, it
    has custom_event_types pre-set, so let's override it by clearing them out.)
    """
    step_adaptor_custom.config.custom_event_types = []

    step_custom_start = make_intermediate_step(IntermediateStepType.CUSTOM_START)
    result_start = step_adaptor_custom.process(step_custom_start)

    assert result_start is None, "With empty custom_event_types, no events should be processed."


def test_process_llm_in_custom_mode_no_op(
    step_adaptor_custom: StepAdaptor,
    make_intermediate_step: Callable[..., IntermediateStep],
) -> None:
    """
    In CUSTOM mode with only CUSTOM_START/END in custom_event_types,
    an LLM event is not processed.
    """
    step_llm = make_intermediate_step(event_type=IntermediateStepType.LLM_START)
    result = step_adaptor_custom.process(step_llm)

    assert result is None


def test_process_llm_in_disabled_mode_no_op(
    step_adaptor_disabled: StepAdaptor,
    make_intermediate_step: Callable[..., IntermediateStep],
) -> None:
    """
    In DISABLED mode, LLM events should not be processed.
    """
    step_llm = make_intermediate_step(event_type=IntermediateStepType.LLM_START)
    result = step_adaptor_disabled.process(step_llm)

    assert result is None


# --------------------
# Test content generation / markdown structures
# --------------------
def test_llm_output_markdown_structure(
    step_adaptor_default: StepAdaptor,
    make_intermediate_step: Callable[..., IntermediateStep],
) -> None:
    """
    Verify that the adapter constructs the correct markdown for LLM output.
    LLM_NEW_TOKEN accumulates chunks. LLM_END has a final output string.
    """
    # LLM_START
    step_start = make_intermediate_step(
        event_type=IntermediateStepType.LLM_START,
        data_input="LLM Input Here",
        UUID="same-run-id",
    )
    # LLM_NEW_TOKEN
    step_token = make_intermediate_step(
        event_type=IntermediateStepType.LLM_NEW_TOKEN,
        data_input=None,
        name="test_llm",
        data_output="partial chunk",
        UUID="same-run-id",
    )
    # LLM_END
    step_end = make_intermediate_step(
        event_type=IntermediateStepType.LLM_END,
        data_input=None,
        data_output="Final LLM Output",
        UUID="same-run-id",
    )

    step_adaptor_default.process(step_start)
    # partial chunk
    step_adaptor_default.process(step_token)
    result_end = step_adaptor_default.process(step_end)

    # result_end should contain the entire markdown
    assert result_end is not None
    assert "Input:" in result_end.payload, "Should contain 'Input:'"
    assert "LLM Input Here" in result_end.payload, "Should display original input"
    assert "Output:" in result_end.payload, "Should contain 'Output:'"
    assert "Final LLM Output" in result_end.payload, "Should contain final output from LLM_END"


def test_tool_end_markdown_structure(
    step_adaptor_default: StepAdaptor,
    make_intermediate_step: Callable[..., IntermediateStep],
) -> None:
    """
    Verify that the adapter constructs the correct markdown for tool output in DEFAULT mode.
    """

    # Create a matching TOOL_START event with the same UUID
    step_tool_start = make_intermediate_step(
        event_type=IntermediateStepType.TOOL_START,
        data_input="TOOL INPUT STUFF",
        UUID="same-run-id",
    )
    step_tool_end = make_intermediate_step(
        event_type=IntermediateStepType.TOOL_END,
        data_input="TOOL INPUT STUFF",
        data_output="TOOL OUTPUT STUFF",
        UUID="same-run-id",
    )

    step_adaptor_default.process(step_tool_start)
    result = step_adaptor_default.process(step_tool_end)
    assert result is not None
    assert "Input:" in result.payload
    assert "Output:" in result.payload
    assert "TOOL INPUT STUFF" in result.payload
    assert "TOOL OUTPUT STUFF" in result.payload


def test_custom_end_markdown_structure(
    step_adaptor_custom: StepAdaptor,
    make_intermediate_step: Callable[..., IntermediateStep],
) -> None:
    """
    Verify that the adapter constructs correct markdown for a custom event.
    """
    step_custom_end = make_intermediate_step(
        event_type=IntermediateStepType.CUSTOM_END,
        data_input="CUSTOM EVENT INPUT",
        data_output="CUSTOM EVENT OUTPUT",
    )

    result = step_adaptor_custom.process(step_custom_end)
    assert result is not None
    assert isinstance(result, ResponseIntermediateStep)
    # We only generate minimal markdown for custom events; check if content is present
    assert "CUSTOM_END" in result.name, "Should show the event type in the name"
    # The entire payload is just a code block: ensure we see the string
    # The 'escaped_payload' from _handle_custom should contain the entire step payload info
    assert "CUSTOM EVENT INPUT" in result.payload or "CUSTOM EVENT OUTPUT" in result.payload


# --------------------
# Tests for FUNCTION events
# --------------------
def test_process_function_start_in_default(
    step_adaptor_default: StepAdaptor,
    make_intermediate_step: Callable[..., IntermediateStep],
) -> None:
    """
    In DEFAULT mode, FUNCTION_START events should be processed and tracked in pending starts.
    """
    uuid = "function-start-uuid"
    step = make_intermediate_step(
        event_type=IntermediateStepType.FUNCTION_START,
        data_input="Function Input Data",
        name="test_function",
        UUID=uuid,
    )

    result = step_adaptor_default.process(step)

    assert result is not None, "Expected FUNCTION_START event to be processed in DEFAULT mode."
    assert isinstance(result, ResponseIntermediateStep)
    assert "Function Start:" in result.name
    assert "test_function" in result.name
    assert "Function Input:" in result.payload
    assert "Function Input Data" in result.payload
    assert uuid in step_adaptor_default._function_starts


def test_process_function_end_in_default(
    step_adaptor_default: StepAdaptor,
    make_intermediate_step: Callable[..., IntermediateStep],
) -> None:
    """
    In DEFAULT mode, FUNCTION_END events should be processed.
    """
    uuid = "function-end-uuid"
    step = make_intermediate_step(
        event_type=IntermediateStepType.FUNCTION_END,
        data_output="Function Output Data",
        name="test_function",
        UUID=uuid,
    )

    result = step_adaptor_default.process(step)

    assert result is not None, "Expected FUNCTION_END event to be processed in DEFAULT mode."
    assert isinstance(result, ResponseIntermediateStep)
    assert "Function Complete:" in result.name
    assert "test_function" in result.name
    assert "Function Output:" in result.payload
    assert "Function Output Data" in result.payload
    assert uuid not in step_adaptor_default._function_starts


def test_function_end_with_matching_start_event(
    step_adaptor_default: StepAdaptor,
    make_intermediate_step: Callable[..., IntermediateStep],
) -> None:
    """
    Test that FUNCTION_END events include the input from the matching FUNCTION_START event.
    """
    # Create a FUNCTION_START event with a specific UUID
    uuid = "function-test-uuid"
    start_step = make_intermediate_step(
        event_type=IntermediateStepType.FUNCTION_START,
        data_input="Function Input Data",
        name="test_function",
        UUID=uuid,
    )

    # Create a matching FUNCTION_END event with the same UUID
    end_step = make_intermediate_step(
        event_type=IntermediateStepType.FUNCTION_END,
        data_output="Function Output Data",
        name="test_function",
        UUID=uuid,
    )

    # Process the start event first
    step_adaptor_default.process(start_step)
    assert uuid in step_adaptor_default._function_starts

    # Then process the end event
    result = step_adaptor_default.process(end_step)

    assert result is not None
    assert "Function Input:" in result.payload, "Should include input from matching start event"
    assert "Function Input Data" in result.payload, "Should contain original input data"
    assert "Function Output:" in result.payload, "Should include output data"
    assert "Function Output Data" in result.payload, "Should contain output data"
    assert uuid not in step_adaptor_default._function_starts


def test_function_events_markdown_structure(
    step_adaptor_default: StepAdaptor,
    make_intermediate_step: Callable[..., IntermediateStep],
) -> None:
    """
    Verify that the adapter constructs the correct markdown for function events.
    """
    # FUNCTION_START
    uuid = "function-markdown-test-uuid"
    step_start = make_intermediate_step(
        event_type=IntermediateStepType.FUNCTION_START,
        data_input={
            "arg1": "value1", "arg2": 42
        },
        name="test_complex_function",
        UUID=uuid,
    )

    # FUNCTION_END
    step_end = make_intermediate_step(
        event_type=IntermediateStepType.FUNCTION_END,
        data_output={
            "result": "success", "value": 42
        },
        name="test_complex_function",
        UUID=uuid,
    )

    # Process both events
    result_start = step_adaptor_default.process(step_start)
    result_end = step_adaptor_default.process(step_end)

    # Check start result
    assert result_start is not None
    assert "Function Start: test_complex_function" == result_start.name
    assert "Function Input:" in result_start.payload
    assert '"arg1": "value1"' in result_start.payload or "'arg1': 'value1'" in result_start.payload
    assert '"arg2": 42' in result_start.payload or "'arg2': 42" in result_start.payload

    # Check end result
    assert result_end is not None
    assert "Function Complete: test_complex_function" == result_end.name
    assert "Function Input:" in result_end.payload, "End event should include input from matching start event"
    assert "Function Output:" in result_end.payload
    assert '"result": "success"' in result_end.payload or "'result': 'success'" in result_end.payload
    assert '"value": 42' in result_end.payload or "'value': 42" in result_end.payload
    assert uuid not in step_adaptor_default._function_starts


def test_process_function_start_without_input(
    step_adaptor_default: StepAdaptor,
    make_intermediate_step: Callable[..., IntermediateStep],
) -> None:
    """
    Test that FUNCTION_START events with None input are still processed.
    """
    step = make_intermediate_step(
        event_type=IntermediateStepType.FUNCTION_START,
        data_input=None,
        name="test_function_no_input",
    )

    result = step_adaptor_default.process(step)

    assert result is not None, "FUNCTION_START events should be processed even with None input"
    assert isinstance(result, ResponseIntermediateStep)
    assert "Function Start:" in result.name
    assert "test_function_no_input" in result.name
    assert "Function Input:" in result.payload
    assert "None" in result.payload


def test_process_function_end_without_output(
    step_adaptor_default: StepAdaptor,
    make_intermediate_step: Callable[..., IntermediateStep],
) -> None:
    """
    Test that FUNCTION_END events with None output are still processed.
    """
    step = make_intermediate_step(
        event_type=IntermediateStepType.FUNCTION_END,
        data_output=None,
        name="test_function_no_output",
    )

    result = step_adaptor_default.process(step)

    assert result is not None, "FUNCTION_END events should be processed even with None output"
    assert isinstance(result, ResponseIntermediateStep)
    assert "Function Complete:" in result.name
    assert "test_function_no_output" in result.name
    assert "Function Output:" in result.payload
    assert "None" in result.payload


def test_function_events_in_custom_mode(
    step_adaptor_custom: StepAdaptor,
    make_intermediate_step: Callable[..., IntermediateStep],
) -> None:
    """
    In CUSTOM mode without FUNCTION_START/END in custom_event_types,
    function events should not be processed.
    """
    # Create function events
    step_start = make_intermediate_step(
        event_type=IntermediateStepType.FUNCTION_START,
        data_input="Function Input Data",
    )

    step_end = make_intermediate_step(
        event_type=IntermediateStepType.FUNCTION_END,
        data_output="Function Output Data",
    )

    # Process the events in custom mode
    result_start = step_adaptor_custom.process(step_start)
    result_end = step_adaptor_custom.process(step_end)

    # Both should return None since they're not in the custom_event_types list
    assert result_start is None, (
        "FUNCTION_START should not be processed in CUSTOM mode without being in custom_event_types")
    assert result_end is None, (
        "FUNCTION_END should not be processed in CUSTOM mode without being in custom_event_types")


def test_truncate_text_helper(default_config: StepAdaptorConfig) -> None:
    """
    Verify that the `_truncate_text` helper correctly truncates strings exceeding limits and handles empty inputs.
    """
    adaptor = StepAdaptor(config=default_config)

    assert adaptor._truncate_text(None, 10) == ""
    assert adaptor._truncate_text("", 10) == ""
    assert adaptor._truncate_text("hello", 10) == "hello"
    assert adaptor._truncate_text("hello world", 11) == "hello world"
    assert adaptor._truncate_text("hello world", None) == "hello world"
    assert adaptor._truncate_text("hello world", 0) == "hello world"
    assert adaptor._truncate_text("hello world", -1) == "hello world"

    truncated_short = adaptor._truncate_text("hello world", 5)
    assert truncated_short == "hello"
    assert len(truncated_short) <= 5

    long_text = "A" * 100
    truncated_long = adaptor._truncate_text(long_text, 50)
    assert len(truncated_long) <= 50
    assert "[truncated 81 characters]" in truncated_long


def test_tool_truncation(make_intermediate_step: Callable[..., IntermediateStep]) -> None:
    """
    Verify that tool inputs and outputs exceeding configured character limits are truncated.
    """
    config = StepAdaptorConfig(max_input_length=50, max_output_length=55)
    adaptor = StepAdaptor(config=config)

    long_input = "A" * 100
    long_output = "B" * 100

    step_start = make_intermediate_step(
        event_type=IntermediateStepType.TOOL_START,
        data_input=long_input,
        UUID="tool-trunc-uuid",
    )
    result_start = adaptor.process(step_start)
    assert result_start is not None
    assert "[truncated 81 characters]" in result_start.payload

    step_end = make_intermediate_step(
        event_type=IntermediateStepType.TOOL_END,
        data_input=long_input,
        data_output=long_output,
        UUID="tool-trunc-uuid",
    )
    result_end = adaptor.process(step_end)
    assert result_end is not None
    assert "[truncated 81 characters]" in result_end.payload
    assert "[truncated 76 characters]" in result_end.payload


def test_function_truncation(make_intermediate_step: Callable[..., IntermediateStep]) -> None:
    """
    Verify that function inputs and outputs exceeding configured character limits are truncated.
    """
    config = StepAdaptorConfig(max_input_length=50, max_output_length=60)
    adaptor = StepAdaptor(config=config)

    long_input = "X" * 100
    long_output = "Y" * 100
    uuid = "func-trunc-uuid"

    step_start = make_intermediate_step(
        event_type=IntermediateStepType.FUNCTION_START,
        data_input=long_input,
        name="long_fn",
        UUID=uuid,
    )
    result_start = adaptor.process(step_start)
    assert result_start is not None
    assert "[truncated 81 characters]" in result_start.payload

    step_end = make_intermediate_step(
        event_type=IntermediateStepType.FUNCTION_END,
        data_output=long_output,
        name="long_fn",
        UUID=uuid,
    )
    result_end = adaptor.process(step_end)
    assert result_end is not None
    assert "[truncated 81 characters]" in result_end.payload
    assert "[truncated 71 characters]" in result_end.payload


def test_llm_token_streaming_opt_out(make_intermediate_step: Callable[..., IntermediateStep], ) -> None:
    """
    Verify that when `stream_llm_tokens` is explicitly `False`, token chunks return `None` and start/end events emit.
    """
    config = StepAdaptorConfig(stream_llm_tokens=False)
    adaptor = StepAdaptor(config=config)
    uuid = "llm-stream-test-uuid"
    step_start = make_intermediate_step(
        event_type=IntermediateStepType.LLM_START,
        data_input="Prompt input",
        UUID=uuid,
    )
    step_token = make_intermediate_step(
        event_type=IntermediateStepType.LLM_NEW_TOKEN,
        data_output="token chunk",
        UUID=uuid,
    )
    step_end = make_intermediate_step(
        event_type=IntermediateStepType.LLM_END,
        data_output="Final response",
        UUID=uuid,
    )

    result_start = adaptor.process(step_start)
    assert result_start is not None
    assert isinstance(result_start, ResponseIntermediateStep)
    assert "Prompt input" in result_start.payload

    result_token = adaptor.process(step_token)
    assert result_token is None

    result_end = adaptor.process(step_end)
    assert result_end is not None
    assert isinstance(result_end, ResponseIntermediateStep)
    assert "Final response" in result_end.payload


def test_llm_token_streaming_default_enabled(
    step_adaptor_default: StepAdaptor,
    make_intermediate_step: Callable[..., IntermediateStep],
) -> None:
    """
    Verify that when `stream_llm_tokens` is `True` (default), intermediate token chunks emit
    cumulative streaming updates.
    """
    uuid = "llm-stream-default-uuid"

    step_start = make_intermediate_step(
        event_type=IntermediateStepType.LLM_START,
        data_input="Prompt input",
        UUID=uuid,
    )
    step_token1 = make_intermediate_step(
        event_type=IntermediateStepType.LLM_NEW_TOKEN,
        data_output="hello ",
        UUID=uuid,
    )
    step_token2 = make_intermediate_step(
        event_type=IntermediateStepType.LLM_NEW_TOKEN,
        data_output="world",
        UUID=uuid,
    )
    step_end = make_intermediate_step(
        event_type=IntermediateStepType.LLM_END,
        data_output="hello world!",
        UUID=uuid,
    )

    step_adaptor_default.process(step_start)

    result_t1 = step_adaptor_default.process(step_token1)
    assert result_t1 is not None
    assert "hello " in result_t1.payload

    result_t2 = step_adaptor_default.process(step_token2)
    assert result_t2 is not None
    assert "hello world" in result_t2.payload

    result_end = step_adaptor_default.process(step_end)
    assert result_end is not None
    assert "hello world!" in result_end.payload
    assert uuid not in step_adaptor_default._llm_tokens
    assert uuid not in step_adaptor_default._llm_token_counts


def test_llm_truncation(make_intermediate_step: Callable[..., IntermediateStep]) -> None:
    """
    Verify that LLM inputs and outputs exceeding configured character limits are truncated.
    """
    config = StepAdaptorConfig(stream_llm_tokens=True, max_input_length=50, max_output_length=55)
    adaptor = StepAdaptor(config=config)
    uuid = "llm-trunc-uuid"

    long_prompt = "P" * 100
    long_output = "R" * 100

    step_start = make_intermediate_step(
        event_type=IntermediateStepType.LLM_START,
        data_input=long_prompt,
        UUID=uuid,
    )
    result_start = adaptor.process(step_start)
    assert result_start is not None
    assert "[truncated 81 characters]" in result_start.payload

    step_end = make_intermediate_step(
        event_type=IntermediateStepType.LLM_END,
        data_output=long_output,
        UUID=uuid,
    )
    result_end = adaptor.process(step_end)
    assert result_end is not None
    assert "[truncated 81 characters]" in result_end.payload
    assert "[truncated 76 characters]" in result_end.payload


def test_config_negative_length_validation() -> None:
    """
    Verify that `StepAdaptorConfig` raises a `ValidationError` when negative lengths are supplied.
    """
    with pytest.raises(ValidationError):
        StepAdaptorConfig(max_input_length=-1)

    with pytest.raises(ValidationError):
        StepAdaptorConfig(max_output_length=-1)


def test_llm_chunk_payload_fallback(make_intermediate_step: Callable[..., IntermediateStep]) -> None:
    """
    Verify that `_handle_llm` extracts token chunks from `step.data.payload` if `chunk` and `output` are `None`.
    """
    config = StepAdaptorConfig(stream_llm_tokens=True)
    adaptor = StepAdaptor(config=config)
    uuid = "llm-payload-fallback-uuid"

    step_start = make_intermediate_step(
        event_type=IntermediateStepType.LLM_START,
        data_input="Prompt input",
        UUID=uuid,
    )
    adaptor.process(step_start)

    step_token = make_intermediate_step(
        event_type=IntermediateStepType.LLM_NEW_TOKEN,
        UUID=uuid,
    )
    assert step_token.data is not None
    step_token.data.payload = "chunk from payload"

    result_token = adaptor.process(step_token)
    assert result_token is not None
    assert "chunk from payload" in result_token.payload


def test_llm_tokens_cleanup_when_end_filtered(make_intermediate_step: Callable[..., IntermediateStep]) -> None:
    """
    Verify that `_llm_tokens` buffer is cleaned up upon `LLM_END` even if the end event is filtered out.
    """
    config = StepAdaptorConfig(
        mode=StepAdaptorMode.CUSTOM,
        custom_event_types=[IntermediateStepType.LLM_START, IntermediateStepType.LLM_NEW_TOKEN],
        stream_llm_tokens=True,
    )
    adaptor = StepAdaptor(config=config)
    uuid = "llm-filtered-end-uuid"

    step_start = make_intermediate_step(
        event_type=IntermediateStepType.LLM_START,
        data_input="Prompt",
        UUID=uuid,
    )
    step_token = make_intermediate_step(
        event_type=IntermediateStepType.LLM_NEW_TOKEN,
        data_output="streamed chunk",
        UUID=uuid,
    )
    step_end = make_intermediate_step(
        event_type=IntermediateStepType.LLM_END,
        data_output="Final Output",
        UUID=uuid,
    )

    adaptor.process(step_start)
    adaptor.process(step_token)
    assert uuid in adaptor._llm_tokens
    assert adaptor._llm_tokens[uuid] == "streamed chunk"
    assert adaptor._llm_token_counts[uuid] == len("streamed chunk")
    assert uuid in adaptor._llm_inputs
    assert adaptor._llm_inputs[uuid] == "Prompt"

    result_end = adaptor.process(step_end)
    assert result_end is None, "LLM_END should be filtered out in this custom mode"
    assert uuid not in adaptor._llm_tokens, "_llm_tokens must be cleaned up on LLM_END even when filtered"
    assert uuid not in adaptor._llm_token_counts, "_llm_token_counts must be cleaned up on LLM_END even when filtered"
    assert uuid not in adaptor._llm_inputs, "_llm_inputs must be cleaned up on LLM_END even when filtered"


@pytest.mark.parametrize("empty_output", [None, "", "   "])
def test_llm_end_output_fallback_to_tokens(
    make_intermediate_step: Callable[..., IntermediateStep],
    empty_output: str | None,
) -> None:
    """
    Verify that `LLM_END` falls back to accumulated streaming tokens when `step.data.output` is `None` or empty.
    """
    config = StepAdaptorConfig(stream_llm_tokens=True)
    adaptor = StepAdaptor(config=config)
    uuid = "llm-end-fallback-uuid"

    step_start = make_intermediate_step(
        event_type=IntermediateStepType.LLM_START,
        data_input="Prompt input",
        UUID=uuid,
    )
    step_token1 = make_intermediate_step(
        event_type=IntermediateStepType.LLM_NEW_TOKEN,
        data_output="streamed ",
        UUID=uuid,
    )
    step_token2 = make_intermediate_step(
        event_type=IntermediateStepType.LLM_NEW_TOKEN,
        data_output="content",
        UUID=uuid,
    )
    step_end = make_intermediate_step(
        event_type=IntermediateStepType.LLM_END,
        data_output=empty_output,
        UUID=uuid,
    )

    adaptor.process(step_start)
    adaptor.process(step_token1)
    adaptor.process(step_token2)
    assert uuid in adaptor._llm_tokens
    assert uuid in adaptor._llm_inputs

    result_end = adaptor.process(step_end)
    assert result_end is not None
    assert "streamed content" in result_end.payload
    assert uuid not in adaptor._llm_tokens
    assert uuid not in adaptor._llm_token_counts
    assert uuid not in adaptor._llm_inputs


def test_llm_token_accumulation_when_streaming_disabled(
    make_intermediate_step: Callable[..., IntermediateStep], ) -> None:
    """
    Verify that when `stream_llm_tokens` is `False`, tokens accumulate and emit full output on `LLM_END`.
    """
    config = StepAdaptorConfig(stream_llm_tokens=False)
    adaptor = StepAdaptor(config=config)
    uuid = "llm-accum-disabled-uuid"

    step_start = make_intermediate_step(
        event_type=IntermediateStepType.LLM_START,
        data_input="Prompt input",
        UUID=uuid,
    )
    step_token1 = make_intermediate_step(
        event_type=IntermediateStepType.LLM_NEW_TOKEN,
        data_output="Hello ",
        UUID=uuid,
    )
    step_token2 = make_intermediate_step(
        event_type=IntermediateStepType.LLM_NEW_TOKEN,
        data_output="World!",
        UUID=uuid,
    )
    step_end = make_intermediate_step(
        event_type=IntermediateStepType.LLM_END,
        data_output=None,
        UUID=uuid,
    )

    result_start = adaptor.process(step_start)
    assert result_start is not None
    assert uuid in adaptor._llm_inputs

    result_t1 = adaptor.process(step_token1)
    assert result_t1 is None

    result_t2 = adaptor.process(step_token2)
    assert result_t2 is None

    result_end = adaptor.process(step_end)
    assert result_end is not None
    assert "Hello World!" in result_end.payload
    assert uuid not in adaptor._llm_tokens
    assert uuid not in adaptor._llm_token_counts
    assert uuid not in adaptor._llm_inputs


def test_llm_inputs_caching_and_eviction(make_intermediate_step: Callable[..., IntermediateStep]) -> None:
    """
    Verify that `_llm_inputs` caches start input during streaming and is evicted on `LLM_END`.
    """
    config = StepAdaptorConfig(stream_llm_tokens=True)
    adaptor = StepAdaptor(config=config)
    uuid = "llm-inputs-cache-uuid"

    step_start = make_intermediate_step(
        event_type=IntermediateStepType.LLM_START,
        data_input="Cached prompt input",
        UUID=uuid,
    )
    step_token = make_intermediate_step(
        event_type=IntermediateStepType.LLM_NEW_TOKEN,
        data_output="token",
        UUID=uuid,
    )
    step_end = make_intermediate_step(
        event_type=IntermediateStepType.LLM_END,
        data_output="Final output",
        UUID=uuid,
    )

    res_start = adaptor.process(step_start)
    assert res_start is not None
    assert uuid in adaptor._llm_inputs
    assert adaptor._llm_inputs[uuid] == "Cached prompt input"

    res_token = adaptor.process(step_token)
    assert res_token is not None
    assert "Cached prompt input" in res_token.payload
    assert "token" in res_token.payload

    res_end = adaptor.process(step_end)
    assert res_end is not None
    assert "Cached prompt input" in res_end.payload
    assert "Final output" in res_end.payload
    assert uuid not in adaptor._llm_inputs
    assert uuid not in adaptor._llm_tokens
    assert uuid not in adaptor._llm_token_counts


def test_llm_streaming_bounded_memory_many_chunks(make_intermediate_step: Callable[..., IntermediateStep]) -> None:
    """
    Verify that streaming many chunks bounds memory in `_llm_tokens` and formats truncation correctly.
    """
    max_output_len = 50
    config = StepAdaptorConfig(stream_llm_tokens=True, max_output_length=max_output_len)
    adaptor = StepAdaptor(config=config)
    uuid = "llm-many-chunks-uuid"

    step_start = make_intermediate_step(
        event_type=IntermediateStepType.LLM_START,
        data_input="Start Prompt",
        UUID=uuid,
    )
    adaptor.process(step_start)

    # Stream 500 chunks of 11 characters each (total 5500 characters)
    for i in range(500):
        chunk_step = make_intermediate_step(
            event_type=IntermediateStepType.LLM_NEW_TOKEN,
            data_output=f"chunk_{i:04d}_",
            UUID=uuid,
        )
        res = adaptor.process(chunk_step)
        assert res is not None
        # Bounded buffer must never exceed max_output_len
        assert len(adaptor._llm_tokens[uuid]) <= max_output_len

    assert adaptor._llm_token_counts[uuid] == 500 * 11
    assert len(adaptor._llm_tokens[uuid]) == max_output_len

    # Check the emitted result of the last token
    assert res is not None
    assert f"[truncated {500 * 11 - 18} characters]" in res.payload

    # End the stream and verify eviction
    step_end = make_intermediate_step(
        event_type=IntermediateStepType.LLM_END,
        data_output=None,
        UUID=uuid,
    )
    res_end = adaptor.process(step_end)
    assert res_end is not None
    assert f"[truncated {500 * 11 - 18} characters]" in res_end.payload
    assert uuid not in adaptor._llm_tokens
    assert uuid not in adaptor._llm_token_counts
    assert uuid not in adaptor._llm_inputs


def test_tool_and_function_starts_caching_and_eviction(
    step_adaptor_default: StepAdaptor,
    make_intermediate_step: Callable[..., IntermediateStep],
) -> None:
    """
    Verify that pending starts for tools and functions are cached on START and evicted on END.
    """
    tool_uuid = "tool-eviction-uuid"
    tool_start = make_intermediate_step(
        event_type=IntermediateStepType.TOOL_START,
        data_input="tool input",
        UUID=tool_uuid,
    )
    tool_end = make_intermediate_step(
        event_type=IntermediateStepType.TOOL_END,
        data_output="tool output",
        UUID=tool_uuid,
    )

    step_adaptor_default.process(tool_start)
    assert tool_uuid in step_adaptor_default._tool_starts

    step_adaptor_default.process(tool_end)
    assert tool_uuid not in step_adaptor_default._tool_starts

    func_uuid = "func-eviction-uuid"
    func_start = make_intermediate_step(
        event_type=IntermediateStepType.FUNCTION_START,
        data_input="func input",
        name="my_func",
        UUID=func_uuid,
    )
    func_end = make_intermediate_step(
        event_type=IntermediateStepType.FUNCTION_END,
        data_output="func output",
        name="my_func",
        UUID=func_uuid,
    )

    step_adaptor_default.process(func_start)
    assert func_uuid in step_adaptor_default._function_starts

    step_adaptor_default.process(func_end)
    assert func_uuid not in step_adaptor_default._function_starts


def test_clear_state_clears_all_cached_structures(
    step_adaptor_default: StepAdaptor,
    make_intermediate_step: Callable[..., IntermediateStep],
) -> None:
    """
    Verify that `clear_state` empties all pending starts and LLM streaming state buffers.
    """
    uuid_tool = "tool-clear-uuid"
    uuid_func = "func-clear-uuid"
    uuid_llm = "llm-clear-uuid"

    step_adaptor_default.process(
        make_intermediate_step(
            event_type=IntermediateStepType.TOOL_START,
            data_input="tool input",
            UUID=uuid_tool,
        ))
    step_adaptor_default.process(
        make_intermediate_step(
            event_type=IntermediateStepType.FUNCTION_START,
            data_input="func input",
            name="func",
            UUID=uuid_func,
        ))
    step_adaptor_default.process(
        make_intermediate_step(
            event_type=IntermediateStepType.LLM_START,
            data_input="llm input",
            UUID=uuid_llm,
        ))
    step_adaptor_default.process(
        make_intermediate_step(
            event_type=IntermediateStepType.LLM_NEW_TOKEN,
            data_output="chunk",
            UUID=uuid_llm,
        ))

    assert uuid_tool in step_adaptor_default._tool_starts
    assert uuid_func in step_adaptor_default._function_starts
    assert uuid_llm in step_adaptor_default._llm_inputs
    assert uuid_llm in step_adaptor_default._llm_tokens
    assert uuid_llm in step_adaptor_default._llm_token_counts

    step_adaptor_default.clear_state()

    assert len(step_adaptor_default._tool_starts) == 0
    assert len(step_adaptor_default._function_starts) == 0
    assert len(step_adaptor_default._llm_inputs) == 0
    assert len(step_adaptor_default._llm_tokens) == 0
    assert len(step_adaptor_default._llm_token_counts) == 0


def test_llm_new_token_empty_stream_event_data_ignored(
    step_adaptor_default: StepAdaptor,
    make_intermediate_step: Callable[..., IntermediateStep],
) -> None:
    """
    Verify that `LLM_NEW_TOKEN` events with empty object data (e.g. `StreamEventData()`) are ignored.
    """
    uuid = "llm-empty-data-uuid"
    step_start = make_intermediate_step(
        event_type=IntermediateStepType.LLM_START,
        data_input="Prompt input",
        UUID=uuid,
    )
    step_adaptor_default.process(step_start)

    step_empty = make_intermediate_step(
        event_type=IntermediateStepType.LLM_NEW_TOKEN,
        UUID=uuid,
    )
    step_empty.payload.data = StreamEventData()

    step_adaptor_default.process(step_empty)
    # Empty object must not add characters to _llm_token_counts or _llm_tokens
    assert uuid not in step_adaptor_default._llm_token_counts
    assert uuid not in step_adaptor_default._llm_tokens


@pytest.mark.parametrize(
    ("max_in", "max_out"),
    [
        (None, None),
        (0, 0),
        (None, 0),
        (0, None),
    ],
)
def test_truncation_disabled_with_none_or_zero(
    make_intermediate_step: Callable[..., IntermediateStep],
    max_in: int | None,
    max_out: int | None,
) -> None:
    """
    Verify that setting `max_input_length` and `max_output_length` to `None` or `0` disables
    truncation entirely across LLM, tool, and function events.
    """
    config = StepAdaptorConfig(
        stream_llm_tokens=True,
        max_input_length=max_in,
        max_output_length=max_out,
    )
    adaptor = StepAdaptor(config=config)

    long_prompt = "P" * 5000
    long_output = "R" * 5000
    uuid = f"disable-trunc-{max_in}-{max_out}"

    step_start = make_intermediate_step(
        event_type=IntermediateStepType.LLM_START,
        data_input=long_prompt,
        UUID=uuid,
    )
    res_start = adaptor.process(step_start)
    assert res_start is not None
    assert long_prompt in res_start.payload
    assert "[truncated" not in res_start.payload

    step_token = make_intermediate_step(
        event_type=IntermediateStepType.LLM_NEW_TOKEN,
        data_output=long_output,
        UUID=uuid,
    )
    res_token = adaptor.process(step_token)
    assert res_token is not None
    assert long_output in res_token.payload
    assert "[truncated" not in res_token.payload

    step_end = make_intermediate_step(
        event_type=IntermediateStepType.LLM_END,
        data_output=long_output,
        UUID=uuid,
    )
    res_end = adaptor.process(step_end)
    assert res_end is not None
    assert long_prompt in res_end.payload
    assert long_output in res_end.payload
    assert "[truncated" not in res_end.payload

    tool_uuid = f"tool-disable-trunc-{max_in}-{max_out}"
    tool_start = make_intermediate_step(
        event_type=IntermediateStepType.TOOL_START,
        data_input=long_prompt,
        UUID=tool_uuid,
    )
    res_tstart = adaptor.process(tool_start)
    assert res_tstart is not None
    assert long_prompt in res_tstart.payload
    assert "[truncated" not in res_tstart.payload

    tool_end = make_intermediate_step(
        event_type=IntermediateStepType.TOOL_END,
        data_input=long_prompt,
        data_output=long_output,
        UUID=tool_uuid,
    )
    res_tend = adaptor.process(tool_end)
    assert res_tend is not None
    assert long_prompt in res_tend.payload
    assert long_output in res_tend.payload
    assert "[truncated" not in res_tend.payload

    func_uuid = f"func-disable-trunc-{max_in}-{max_out}"
    func_start = make_intermediate_step(
        event_type=IntermediateStepType.FUNCTION_START,
        data_input=long_prompt,
        name="test_func",
        UUID=func_uuid,
    )
    res_fstart = adaptor.process(func_start)
    assert res_fstart is not None
    assert long_prompt in res_fstart.payload
    assert "[truncated" not in res_fstart.payload

    func_end = make_intermediate_step(
        event_type=IntermediateStepType.FUNCTION_END,
        data_output=long_output,
        name="test_func",
        UUID=func_uuid,
    )
    res_fend = adaptor.process(func_end)
    assert res_fend is not None
    assert long_prompt in res_fend.payload
    assert long_output in res_fend.payload
    assert "[truncated" not in res_fend.payload
