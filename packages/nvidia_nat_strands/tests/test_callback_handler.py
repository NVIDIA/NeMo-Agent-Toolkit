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

import pytest

from nat.utils.reactive.subject import Subject


async def test_strands_handler_tool_execution(reactive_stream: Subject):
    """
    Test that Strands handler correctly tracks tool execution:
    - It should generate TOOL_START event when a tool is executed
    - It should generate TOOL_END event after tool execution completes
    - The events should contain correct input args and output results
    """

    from nat.builder.framework_enum import LLMFrameworkEnum
    from nat.data_models.intermediate_step import IntermediateStepType
    from nat.plugins.strands.callback_handler import StrandsProfilerHandler
    from nat.plugins.strands.callback_handler import StrandsToolInstrumentationHook

    # Set up handler and collect results
    all_stats = []
    handler = StrandsProfilerHandler()
    reactive_stream.subscribe(all_stats.append)

    # Create a tool hook instance (this is normally done per-agent-instance)
    tool_hook = StrandsToolInstrumentationHook(handler)

    # Simulate tool execution events that would come from Strands hooks
    tool_use_id = "strands-tool-123"
    tool_name = "test_strands_tool"
    tool_input = {"param1": "value1", "param2": "value2"}
    tool_output = "Strands tool execution result"

    # Create mock events similar to what Strands would generate
    class MockBeforeEvent:

        def __init__(self):
            self.tool_use = {"toolUseId": tool_use_id, "name": tool_name, "input": tool_input}
            self.selected_tool = type(
                'MockTool', (), {
                    'tool_name': tool_name, 'tool_spec': {
                        "name": tool_name, "description": "Test tool"
                    }
                })()

    class MockAfterEvent:

        def __init__(self):
            self.tool_use = {"toolUseId": tool_use_id, "name": tool_name, "input": tool_input}
            self.selected_tool = type('MockTool', (), {'tool_name': tool_name, 'tool_spec': {"name": tool_name}})()
            self.result = {"content": [{"text": tool_output}]}
            self.exception = None

    # Simulate the tool execution flow
    before_event = MockBeforeEvent()
    after_event = MockAfterEvent()

    # Call the hook methods directly
    tool_hook.on_before_tool_invocation(before_event)
    tool_hook.on_after_tool_invocation(after_event)

    # Verify events were generated
    assert len(all_stats) >= 2, f"Expected at least 2 events, got {len(all_stats)}"

    # Find TOOL_START and TOOL_END events
    tool_start_events = [
        event for event in all_stats if event.payload.event_type == IntermediateStepType.TOOL_START
        and event.payload.framework == LLMFrameworkEnum.STRANDS
    ]
    tool_end_events = [
        event for event in all_stats if event.payload.event_type == IntermediateStepType.TOOL_END
        and event.payload.framework == LLMFrameworkEnum.STRANDS
    ]

    assert len(tool_start_events) > 0, "No TOOL_START events found for Strands"
    assert len(tool_end_events) > 0, "No TOOL_END events found for Strands"

    # Verify event details
    start_event = tool_start_events[-1]
    end_event = tool_end_events[-1]

    # Check TOOL_START event
    assert start_event.payload.name == tool_name
    assert start_event.payload.UUID == tool_use_id
    assert start_event.payload.framework == LLMFrameworkEnum.STRANDS
    assert start_event.payload.metadata.tool_inputs == tool_input

    # Check TOOL_END event
    assert end_event.payload.name == tool_name
    assert end_event.payload.UUID == tool_use_id
    assert end_event.payload.framework == LLMFrameworkEnum.STRANDS
    assert tool_output in end_event.payload.metadata.tool_outputs


@pytest.fixture(name="scripted_agent_factory")
def fixture_scripted_agent_factory(monkeypatch):
    """Instrument ``Agent.__init__`` and return a factory for agents whose model calls ``get_weather`` once."""
    from strands import Agent
    from strands import tool
    from strands.hooks import BeforeToolCallEvent
    from strands.models.model import Model

    from nat.plugins.strands.callback_handler import StrandsProfilerHandler

    class ScriptedModel(Model):
        """Asks for ``get_weather`` and answers once it has the tool result."""

        def update_config(self, **model_config):
            pass

        def get_config(self):
            return {}

        async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
            raise NotImplementedError
            yield  # pylint: disable=unreachable

        async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
            yield {"messageStart": {"role": "assistant"}}
            if messages[-1]["content"][0].get("toolResult"):
                yield {"contentBlockDelta": {"delta": {"text": "Done"}}}
                yield {"contentBlockStop": {}}
                yield {"messageStop": {"stopReason": "end_turn"}}
                return
            yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "call-1", "name": "get_weather"}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"city": "Paris"}'}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}

    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f"Sunny in {city}"

    # Restore Agent.__init__ after the test
    monkeypatch.setattr(Agent, "__init__", Agent.__init__)
    StrandsProfilerHandler()._instrument_agent_init()  # pylint: disable=protected-access

    def make_agent(before_tool_call):
        agent = Agent(model=ScriptedModel(), tools=[get_weather], callback_handler=None)
        agent.add_hook(before_tool_call, BeforeToolCallEvent)
        return agent

    return make_agent


def _tool_steps(steps: list) -> list[tuple]:
    """Event type, UUID and (for end events) output of each tool step."""
    from nat.data_models.intermediate_step import IntermediateStepState

    return [(step.event_type.value,
             step.payload.UUID,
             step.payload.data.output if step.payload.event_state == IntermediateStepState.END else None)
            for step in steps if step.event_type.value.startswith("TOOL")]


async def test_strands_handler_ends_cancelled_tool_call(reactive_stream: Subject, scripted_agent_factory):
    """A tool call that a hook cancels with ``cancel_tool`` ends its tool step with the cancel message."""
    all_stats = []
    reactive_stream.subscribe(all_stats.append)

    def deny(event):
        event.cancel_tool = "Tool call denied by policy"

    await scripted_agent_factory(deny).invoke_async("weather in Paris?")

    assert _tool_steps(all_stats) == [
        ("TOOL_START", "call-1", None),
        ("TOOL_END", "call-1", "Tool call denied by policy"),
    ]
    tool_end = [step for step in all_stats if step.event_type.value == "TOOL_END"][0]
    assert tool_end.payload.name == "get_weather"


async def test_strands_handler_ends_interrupted_tool_call(reactive_stream: Subject, scripted_agent_factory):
    """A tool call that a hook interrupts for approval ends its tool step when the invocation stops."""
    all_stats = []
    reactive_stream.subscribe(all_stats.append)

    def ask_approval(event):
        if event.interrupt("approval", reason={"tool": event.tool_use["name"]}) != "y":
            event.cancel_tool = "User denied the tool call"

    agent = scripted_agent_factory(ask_approval)
    result = await agent.invoke_async("weather in Paris?")

    assert result.stop_reason == "interrupt"
    assert _tool_steps(all_stats) == [("TOOL_START", "call-1", None), ("TOOL_END", "call-1", "")]
    start, end = [step for step in all_stats if step.event_type.value.startswith("TOOL")]
    assert end.span_event_timestamp <= start.event_timestamp

    # Approving the call runs the tool as a new tool step
    all_stats.clear()
    await agent.invoke_async([{"interruptResponse": {"interruptId": result.interrupts[0].id, "response": "y"}}])

    assert _tool_steps(all_stats) == [("TOOL_START", "call-1", None), ("TOOL_END", "call-1", "Sunny in Paris")]
