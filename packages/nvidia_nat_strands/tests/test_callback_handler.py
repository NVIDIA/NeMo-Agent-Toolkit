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
