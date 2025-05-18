# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# pylint: disable=redefined-outer-name, invalid-name
import pytest

from aiq.data_models.api_server import AIQResponseIntermediateStep
from aiq.data_models.intermediate_step import IntermediateStep
from aiq.data_models.intermediate_step import IntermediateStepPayload
from aiq.data_models.intermediate_step import IntermediateStepType
from aiq.data_models.intermediate_step import StreamEventData
from aiq.data_models.invocation_node import InvocationNode
from aiq.data_models.step_adaptor import StepAdaptorConfig
from aiq.data_models.step_adaptor import StepAdaptorMode
from aiq.front_ends.fastapi.step_adaptor import StepAdaptor


@pytest.fixture
def default_config():
    """Return a default StepAdaptorConfig object (mode=DEFAULT)."""
    return StepAdaptorConfig(mode=StepAdaptorMode.DEFAULT, custom_event_types=[])


@pytest.fixture
def custom_config():
    """Return a custom StepAdaptorConfig object (mode=CUSTOM) with custom types."""
    return StepAdaptorConfig(
        mode=StepAdaptorMode.CUSTOM,
        custom_event_types=[
            IntermediateStepType.CUSTOM_START,
            IntermediateStepType.CUSTOM_END,
        ],
    )


@pytest.fixture
def disabled_config():
    """Return a custom StepAdaptorConfig object that disables intermediate steps."""
    return StepAdaptorConfig(
        mode=StepAdaptorMode.OFF,
        custom_event_types=[
            IntermediateStepType.CUSTOM_START,
            IntermediateStepType.CUSTOM_END,
        ],
    )


@pytest.fixture
def step_adaptor_default(default_config):
    """Return a StepAdaptor using the default config."""
    return StepAdaptor(config=default_config)


@pytest.fixture
def step_adaptor_custom(custom_config):
    """Return a StepAdaptor using the custom config."""
    return StepAdaptor(config=custom_config)


@pytest.fixture
def step_adaptor_disabled(disabled_config):
    """Return a StepAdaptor using the disabled config."""
    return StepAdaptor(config=disabled_config)


@pytest.fixture
def make_intermediate_step():
    """A factory fixture to create an IntermediateStep with minimal defaults."""

    def _make_step(event_type: IntermediateStepType, data_input=None, data_output=None, name=None, UUID=None):
        payload = IntermediateStepPayload(
            event_type=event_type,
            name=name or "test_step",
            data=StreamEventData(input=data_input, output=data_output),
            UUID=UUID or "test-uuid-1234",
        )
        # The IntermediateStep constructor requires a function_ancestry,
        # but for testing we can just pass None or a placeholder.
        return IntermediateStep(function_ancestry=InvocationNode(parent_id="abc",
                                                                 function_id="def",
                                                                 function_name="xyz"),
                                payload=payload)

    return _make_step


# --------------------
# Tests for DEFAULT mode
# --------------------
@pytest.mark.parametrize("event_type", [(IntermediateStepType.LLM_START)])
def test_process_llm_events_in_default(step_adaptor_default, make_intermediate_step, event_type):
    """
    In DEFAULT mode, LLM_START, LLM_NEW_TOKEN, and LLM_END events are processed.
    We expect a valid AIQResponseIntermediateStep for each.
    """
    step = make_intermediate_step(event_type=event_type, data_input="LLM Input", data_output="LLM Output")

    result = step_adaptor_default.process(step)

    assert result is not None, f"Expected LLM event '{event_type}' to be processed in DEFAULT mode."
    assert isinstance(result, AIQResponseIntermediateStep)
    assert step_adaptor_default._history[-1] is step, "Step must be appended to _history."


def test_process_tool_in_default(step_adaptor_default, make_intermediate_step):
    """
    In DEFAULT mode, TOOL_END events should be processed.
    """
    step = make_intermediate_step(
        event_type=IntermediateStepType.TOOL_START,
        data_input="Tool Input Data",
        data_output="Tool Output Data",
    )

    result = step_adaptor_default.process(step)

    assert result is not None, "Expected TOOL_START event to be processed in DEFAULT mode."
    assert isinstance(result, AIQResponseIntermediateStep)
    assert "Tool:" in result.name
    assert "Input:" in result.payload
    assert step_adaptor_default._history[-1] is step

    step = make_intermediate_step(
        event_type=IntermediateStepType.TOOL_END,
        data_input="Tool Input Data",
        data_output="Tool Output Data",
    )

    result = step_adaptor_default.process(step)

    assert result is not None, "Expected TOOL_END event to be processed in DEFAULT mode."
    assert isinstance(result, AIQResponseIntermediateStep)
    assert "Tool:" in result.name
    assert "Input:" in result.payload
    assert "Output:" in result.payload
    assert step_adaptor_default._history[-1] is step


@pytest.mark.parametrize("event_type",
                         [
                             (IntermediateStepType.WORKFLOW_START),
                             (IntermediateStepType.WORKFLOW_END),
                             (IntermediateStepType.CUSTOM_START),
                             (IntermediateStepType.CUSTOM_END),
                         ])
def test_process_other_events_in_default_returns_none(step_adaptor_default, make_intermediate_step, event_type):
    """
    In DEFAULT mode, anything other than LLM or TOOL_END should return None.
    """
    step = make_intermediate_step(event_type=event_type)
    result = step_adaptor_default.process(step)

    assert result is None, f"Expected event {event_type} to be ignored in DEFAULT mode."
    # The step should still be appended to _history
    assert step_adaptor_default._history[-1] is step


# --------------------
# Tests for CUSTOM mode
# --------------------
def test_process_custom_events_in_custom_mode(step_adaptor_custom, make_intermediate_step):
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

    # Validate the custom events produce an AIQResponseIntermediateStep
    assert result_start is not None
    assert isinstance(result_start, AIQResponseIntermediateStep)
    assert result_end is not None
    assert isinstance(result_end, AIQResponseIntermediateStep)

    # Validate we do not process LLM or TOOL_END in custom mode (with given custom_event_types)
    assert result_llm is None
    assert result_tool is None

    # Ensure all steps are appended to _history in the order they were processed
    assert step_adaptor_custom._history == [step_start, step_end, step_llm, step_tool]


def test_process_custom_events_empty_list(step_adaptor_custom, make_intermediate_step):
    """
    If the StepAdaptorConfig was set to CUSTOM but had an empty or non-matching
    custom_event_types, we expect no events to be processed. (In the fixture, it
    has custom_event_types pre-set, so let's override it by clearing them out.)
    """
    step_adaptor_custom.config.custom_event_types = []

    step_custom_start = make_intermediate_step(IntermediateStepType.CUSTOM_START)
    result_start = step_adaptor_custom.process(step_custom_start)

    assert result_start is None, "With empty custom_event_types, no events should be processed."
    assert step_adaptor_custom._history[-1] is step_custom_start


def test_process_llm_in_custom_mode_no_op(step_adaptor_custom, make_intermediate_step):
    """
    In CUSTOM mode with only CUSTOM_START/END in custom_event_types,
    an LLM event is not processed.
    """
    step_llm = make_intermediate_step(event_type=IntermediateStepType.LLM_START)
    result = step_adaptor_custom.process(step_llm)

    assert result is None
    assert step_adaptor_custom._history[-1] is step_llm


def test_process_llm_in_disabled_mode_no_op(step_adaptor_disabled, make_intermediate_step):
    """
    In DISABLED mode, LLM events should not be processed.
    """
    step_llm = make_intermediate_step(event_type=IntermediateStepType.LLM_START)
    result = step_adaptor_disabled.process(step_llm)

    assert result is None
    assert step_adaptor_disabled._history[-1] is step_llm


# --------------------
# Test content generation / markdown structures
# --------------------
def test_llm_output_markdown_structure(step_adaptor_default, make_intermediate_step):
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


def test_tool_end_markdown_structure(step_adaptor_default, make_intermediate_step):
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


def test_custom_end_markdown_structure(step_adaptor_custom, make_intermediate_step):
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
    assert isinstance(result, AIQResponseIntermediateStep)
    # We only generate minimal markdown for custom events; check if content is present
    assert "CUSTOM_END" in result.name, "Should show the event type in the name"
    # The entire payload is just a code block: ensure we see the string
    # The 'escaped_payload' from _handle_custom should contain the entire step payload info
    assert "CUSTOM EVENT INPUT" in result.payload or "CUSTOM EVENT OUTPUT" in result.payload


# --------------------
# Tests for FUNCTION events
# --------------------
def test_process_function_start_in_default(step_adaptor_default, make_intermediate_step):
    """
    In DEFAULT mode, FUNCTION_START events should be processed and return a valid AIQResponseIntermediateStep.
    """
    step = make_intermediate_step(
        event_type=IntermediateStepType.FUNCTION_START,
        data_input="Function Input Data",
        name="test_function",
    )

    result = step_adaptor_default.process(step)

    assert result is not None, "Expected FUNCTION_START event to be processed in DEFAULT mode."
    assert isinstance(result, AIQResponseIntermediateStep)
    assert "Function Start:" in result.name
    assert "test_function" in result.name
    assert "Function Input:" in result.payload
    assert "Function Input Data" in result.payload
    assert step_adaptor_default._history[-1] is step


def test_process_function_end_in_default(step_adaptor_default, make_intermediate_step):
    """
    In DEFAULT mode, FUNCTION_END events should be processed.
    """
    step = make_intermediate_step(
        event_type=IntermediateStepType.FUNCTION_END,
        data_output="Function Output Data",
        name="test_function",
    )

    result = step_adaptor_default.process(step)

    assert result is not None, "Expected FUNCTION_END event to be processed in DEFAULT mode."
    assert isinstance(result, AIQResponseIntermediateStep)
    assert "Function Complete:" in result.name
    assert "test_function" in result.name
    assert "Function Output:" in result.payload
    assert "Function Output Data" in result.payload
    assert step_adaptor_default._history[-1] is step


def test_function_end_with_matching_start_event(step_adaptor_default, make_intermediate_step):
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

    # Then process the end event
    result = step_adaptor_default.process(end_step)

    assert result is not None
    assert "Function Input:" in result.payload, "Should include input from matching start event"
    assert "Function Input Data" in result.payload, "Should contain original input data"
    assert "Function Output:" in result.payload, "Should include output data"
    assert "Function Output Data" in result.payload, "Should contain output data"


def test_function_events_markdown_structure(step_adaptor_default, make_intermediate_step):
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


def test_process_function_start_without_input(step_adaptor_default, make_intermediate_step):
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
    assert isinstance(result, AIQResponseIntermediateStep)
    assert "Function Start:" in result.name
    assert "test_function_no_input" in result.name
    assert "Function Input:" in result.payload
    assert "None" in result.payload
    assert step_adaptor_default._history[-1] is step


def test_process_function_end_without_output(step_adaptor_default, make_intermediate_step):
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
    assert isinstance(result, AIQResponseIntermediateStep)
    assert "Function Complete:" in result.name
    assert "test_function_no_output" in result.name
    assert "Function Output:" in result.payload
    assert "None" in result.payload
    assert step_adaptor_default._history[-1] is step


def test_function_events_in_custom_mode(step_adaptor_custom, make_intermediate_step):
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

    # Steps should still be added to history
    assert step_adaptor_custom._history[-2] is step_start
    assert step_adaptor_custom._history[-1] is step_end
