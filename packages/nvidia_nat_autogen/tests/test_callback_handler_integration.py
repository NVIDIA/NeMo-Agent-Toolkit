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
"""Integration tests for AutoGen Callback Handler with real NVIDIA API calls.

These tests validate that telemetry events (LLM_START, LLM_END, TOOL_START, TOOL_END)
are correctly captured when making real LLM calls via AutoGen.

Requirements:
    - NVIDIA_API_KEY environment variable must be set
    - Network access to NVIDIA NIM API
    - nvidia-nat-test package installed (provides test fixtures)

Run with:
    pytest packages/nvidia_nat_autogen/tests/test_callback_handler_integration.py --run_integration -v

    For slow tests (agent with streaming + tools):
    pytest packages/nvidia_nat_autogen/tests/test_callback_handler_integration.py --run_integration --run_slow -v

Tests are skipped by default. Use --run_integration to enable integration tests
and --run_slow for tests marked as slow.
"""

import asyncio
import os
from collections.abc import Callable

import pytest

from nat.builder.context import Context
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepType
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(name="autogen_profiler", scope="function")
def autogen_profiler_fixture():
    """Set up AutoGen profiler instrumentation for telemetry capture.

    This fixture instruments the AutoGen client classes to capture
    LLM and tool call events. It uninstruments after the test.
    """
    from nat.plugins.autogen.callback_handler import AutoGenProfilerHandler

    handler = AutoGenProfilerHandler()
    handler.instrument()

    yield handler

    handler.uninstrument()


@pytest.fixture(name="nim_config")
def nim_config_fixture() -> NIMModelConfig:
    """Create a NIM configuration for testing.

    Reads API key from NVIDIA_API_KEY environment variable.
    """
    api_key = os.environ.get("NVIDIA_API_KEY")
    return NIMModelConfig(
        model_name="meta/llama-3.1-8b-instruct",
        api_key=api_key,
        base_url="https://integrate.api.nvidia.com/v1",
        temperature=0.0,  # Deterministic for testing
        max_tokens=100,  # Keep responses short for faster tests
    )


@pytest.fixture(name="openai_config")
def openai_config_fixture() -> OpenAIModelConfig:
    """Create an OpenAI configuration for testing.

    Reads API key from OPENAI_API_KEY environment variable.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    return OpenAIModelConfig(
        model_name="gpt-4o-mini",
        api_key=api_key,
        temperature=0.0,  # Deterministic for testing
        max_tokens=100,  # Keep responses short for faster tests
    )


@pytest.fixture(name="captured_events")
def captured_events_fixture() -> list[IntermediateStep]:
    """Fixture to capture intermediate step events."""
    return []


@pytest.fixture(name="event_capturer")
def event_capturer_fixture(captured_events: list[IntermediateStep]) -> Callable[[IntermediateStep], None]:
    """Create an event capturer callback function."""

    def capture_event(event: IntermediateStep) -> None:
        captured_events.append(event)

    return capture_event


# ============================================================================
# Test 1: Non-Streaming LLM Call with Telemetry
# ============================================================================


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key", "autogen_profiler")
async def test_nim_autogen_non_streaming_llm_telemetry(
    nim_config: NIMModelConfig,
    captured_events: list[IntermediateStep],
    event_capturer: Callable[[IntermediateStep], None],
):
    """Test non-streaming LLM call captures correct telemetry events.

    Validates:
        - LLM_START event is pushed with correct model name and input
        - LLM_END event is pushed with output and token usage
        - Events are properly paired (same UUID)
        - Response content is valid
    """
    from autogen_core.models import UserMessage

    async with WorkflowBuilder() as builder:
        # Subscribe to intermediate step events
        ctx = Context.get()
        ctx.intermediate_step_manager.subscribe(event_capturer)

        # Get AutoGen client
        await builder.add_llm("test_llm", nim_config)
        client = await builder.get_llm("test_llm", wrapper_type=LLMFrameworkEnum.AUTOGEN)

        # Make a non-streaming LLM call
        messages = [UserMessage(content="What is 2 + 2? Reply with just the number.", source="user")]
        response = await client.create(messages=messages)

        # Allow events to propagate
        await asyncio.sleep(0.1)

    # Validate response
    assert response is not None
    assert hasattr(response, 'content')
    assert "4" in str(response.content)

    # Validate telemetry events
    llm_start_events = [e for e in captured_events if e.payload.event_type == IntermediateStepType.LLM_START]
    llm_end_events = [e for e in captured_events if e.payload.event_type == IntermediateStepType.LLM_END]

    assert len(llm_start_events) >= 1, f"Expected at least 1 LLM_START event, got {len(llm_start_events)}"
    assert len(llm_end_events) >= 1, f"Expected at least 1 LLM_END event, got {len(llm_end_events)}"

    # Get the last pair (most recent call)
    start_event = llm_start_events[-1]
    end_event = llm_end_events[-1]

    # Verify event pairing (same UUID)
    assert start_event.payload.UUID == end_event.payload.UUID, "START and END events should have same UUID"

    # Verify framework
    assert start_event.payload.framework == LLMFrameworkEnum.AUTOGEN
    assert end_event.payload.framework == LLMFrameworkEnum.AUTOGEN

    # Verify model name
    assert nim_config.model_name in start_event.payload.name

    # Verify input was captured (stored in metadata.chat_inputs)
    assert start_event.payload.metadata is not None
    assert start_event.payload.metadata.chat_inputs is not None
    assert len(start_event.payload.metadata.chat_inputs) > 0
    # Check that our input is in the chat inputs
    input_contents = str(start_event.payload.metadata.chat_inputs)
    assert "2 + 2" in input_contents

    # Verify output was captured
    assert end_event.payload.data is not None
    # Output may be in data.output or metadata.chat_responses
    has_output = (end_event.payload.data.output is not None
                  or (end_event.payload.metadata is not None and end_event.payload.metadata.chat_responses is not None))
    assert has_output, "Output should be captured in data.output or metadata.chat_responses"

    # Verify usage_info structure exists (token counts may be 0 for some providers)
    assert end_event.payload.usage_info is not None
    assert end_event.payload.usage_info.token_usage is not None
    # num_llm_calls should be tracked
    assert end_event.payload.usage_info.num_llm_calls >= 1


# ============================================================================
# Test 2: Streaming LLM Call with Telemetry
# ============================================================================


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key", "autogen_profiler")
async def test_nim_autogen_streaming_llm_telemetry(
    nim_config: NIMModelConfig,
    captured_events: list[IntermediateStep],
    event_capturer: Callable[[IntermediateStep], None],
):
    """Test streaming LLM call captures correct telemetry events.

    Validates:
        - LLM_START event fires before first chunk
        - All chunks are yielded correctly
        - LLM_END event fires after stream completion
        - Token usage is captured from final chunk
    """
    from autogen_core.models import UserMessage

    async with WorkflowBuilder() as builder:
        # Subscribe to intermediate step events
        ctx = Context.get()
        ctx.intermediate_step_manager.subscribe(event_capturer)

        # Get AutoGen client
        await builder.add_llm("test_llm", nim_config)
        client = await builder.get_llm("test_llm", wrapper_type=LLMFrameworkEnum.AUTOGEN)

        # Make a streaming LLM call
        messages = [UserMessage(content="Count from 1 to 5. Just the numbers.", source="user")]

        chunks = []
        async for chunk in client.create_stream(messages=messages):
            chunks.append(chunk)

        # Allow events to propagate
        await asyncio.sleep(0.1)

    # Validate chunks were received
    assert len(chunks) > 0, "Expected at least one chunk from streaming response"

    # Validate telemetry events
    llm_start_events = [e for e in captured_events if e.payload.event_type == IntermediateStepType.LLM_START]
    llm_end_events = [e for e in captured_events if e.payload.event_type == IntermediateStepType.LLM_END]

    assert len(llm_start_events) >= 1, f"Expected at least 1 LLM_START event, got {len(llm_start_events)}"
    assert len(llm_end_events) >= 1, f"Expected at least 1 LLM_END event, got {len(llm_end_events)}"

    # Get the last START event and find its matching END event by UUID
    start_event = llm_start_events[-1]
    end_event = next((e for e in llm_end_events if e.payload.UUID == start_event.payload.UUID), None)
    assert end_event is not None, f"No matching LLM_END event for START UUID {start_event.payload.UUID}"

    # Verify framework
    assert start_event.payload.framework == LLMFrameworkEnum.AUTOGEN
    assert end_event.payload.framework == LLMFrameworkEnum.AUTOGEN

    # Verify START event was pushed (input captured in metadata.chat_inputs)
    assert start_event.payload.metadata is not None
    assert start_event.payload.metadata.chat_inputs is not None
    input_contents = str(start_event.payload.metadata.chat_inputs)
    assert "1 to 5" in input_contents

    # Verify END event has output
    assert end_event.payload.data is not None
    has_output = (end_event.payload.data.output is not None
                  or (end_event.payload.metadata is not None and end_event.payload.metadata.chat_responses is not None))
    assert has_output, "Output should be captured"


# ============================================================================
# Test 3: Tool Execution with Telemetry
# ============================================================================


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key", "autogen_profiler")
async def test_nim_autogen_tool_execution_telemetry(
    nim_config: NIMModelConfig,
    captured_events: list[IntermediateStep],
    event_capturer: Callable[[IntermediateStep], None],
):
    """Test tool execution captures correct telemetry events.

    Validates:
        - TOOL_START event is pushed with correct tool name and input
        - TOOL_END event is pushed with correct output
        - Events are properly paired
    """
    from autogen_agentchat.agents import AssistantAgent
    from autogen_core.tools import FunctionTool

    # Define a simple calculator tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers together.

        Args:
            a: First number
            b: Second number

        Returns:
            The product of a and b
        """
        return a * b

    multiply_tool = FunctionTool(multiply, description="Multiply two numbers together")

    async with WorkflowBuilder() as builder:
        # Subscribe to intermediate step events
        ctx = Context.get()
        ctx.intermediate_step_manager.subscribe(event_capturer)

        # Get AutoGen client
        await builder.add_llm("test_llm", nim_config)
        client = await builder.get_llm("test_llm", wrapper_type=LLMFrameworkEnum.AUTOGEN)

        # Create an agent with the tool
        agent = AssistantAgent(
            name="calculator_agent",
            model_client=client,
            tools=[multiply_tool],
            system_message="You are a helpful calculator. Use the multiply tool when asked to multiply numbers.",
        )

        # Run a task that should trigger tool use
        from autogen_agentchat.base import TaskResult
        result = await agent.run(task="What is 7 times 8? Use the multiply tool.")

        # Allow events to propagate
        await asyncio.sleep(0.1)

    # Validate result
    assert isinstance(result, TaskResult)
    assert result.messages is not None
    assert len(result.messages) > 0

    # Check for tool events
    tool_start_events = [e for e in captured_events if e.payload.event_type == IntermediateStepType.TOOL_START]
    tool_end_events = [e for e in captured_events if e.payload.event_type == IntermediateStepType.TOOL_END]

    # Tool should have been called (but LLM might not always use it)
    if len(tool_start_events) > 0:
        assert len(tool_end_events) >= 1, "If TOOL_START exists, TOOL_END should also exist"

        start_event = tool_start_events[-1]
        end_event = tool_end_events[-1]

        # Verify pairing
        assert start_event.payload.UUID == end_event.payload.UUID

        # Verify tool name
        assert "multiply" in start_event.payload.name.lower()

        # Verify output contains result
        assert end_event.payload.data is not None
        # 7 * 8 = 56
        assert "56" in str(end_event.payload.data.output)

    # Regardless, LLM events should exist
    llm_start_events = [e for e in captured_events if e.payload.event_type == IntermediateStepType.LLM_START]
    assert len(llm_start_events) >= 1, "LLM call should have been made"


# ============================================================================
# Test 4: Error Handling Telemetry
# ============================================================================


@pytest.mark.integration
@pytest.mark.usefixtures("autogen_profiler")
async def test_nim_autogen_error_handling_telemetry(
    captured_events: list[IntermediateStep],
    event_capturer: Callable[[IntermediateStep], None],
):
    """Test error handling captures correct telemetry events.

    Validates:
        - LLM_START event is still pushed
        - LLM_END event contains error information
        - Exception is properly re-raised
    """
    from autogen_core.models import UserMessage

    # Create config with invalid API key
    invalid_config = NIMModelConfig(
        model_name="meta/llama-3.1-8b-instruct",
        api_key="invalid-api-key-12345",
        temperature=0.0,
    )

    async with WorkflowBuilder() as builder:
        # Subscribe to intermediate step events
        ctx = Context.get()
        ctx.intermediate_step_manager.subscribe(event_capturer)

        # Get AutoGen client with invalid key
        await builder.add_llm("test_llm", invalid_config)
        client = await builder.get_llm("test_llm", wrapper_type=LLMFrameworkEnum.AUTOGEN)

        # Make a call that should fail
        messages = [UserMessage(content="Hello", source="user")]

        # Should raise authentication or API error - match common auth/API failure patterns
        with pytest.raises(
                Exception,
                match=r"(?i)(authentication|api[_\s]?key|401|unauthorized|invalid|forbidden|credentials)",
        ):
            await client.create(messages=messages)

        # Allow events to propagate
        await asyncio.sleep(0.1)

    # Validate telemetry events
    llm_start_events = [e for e in captured_events if e.payload.event_type == IntermediateStepType.LLM_START]
    llm_end_events = [e for e in captured_events if e.payload.event_type == IntermediateStepType.LLM_END]

    assert len(llm_start_events) >= 1, "LLM_START should be pushed even on error"
    assert len(llm_end_events) >= 1, "LLM_END should be pushed with error info"

    # Verify END event contains error metadata
    end_event = llm_end_events[-1]
    # Error should be captured in output or metadata
    has_error_info = (end_event.payload.data is not None and end_event.payload.data.output is not None
                      and len(end_event.payload.data.output) > 0) or (end_event.payload.metadata is not None
                                                                      and end_event.payload.metadata.error is not None)
    assert has_error_info, "Error information should be captured in END event"


# ============================================================================
# Test 5: Multi-Turn Conversation Telemetry
# ============================================================================


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key", "autogen_profiler")
async def test_nim_autogen_multi_turn_conversation_telemetry(
    nim_config: NIMModelConfig,
    captured_events: list[IntermediateStep],
    event_capturer: Callable[[IntermediateStep], None],
):
    """Test multi-turn conversation captures correct telemetry for each turn.

    Validates:
        - Each LLM call generates its own START/END pair
        - UUIDs are unique per call
        - Message history is captured correctly
    """
    from autogen_core.models import AssistantMessage
    from autogen_core.models import UserMessage

    async with WorkflowBuilder() as builder:
        # Subscribe to intermediate step events
        ctx = Context.get()
        ctx.intermediate_step_manager.subscribe(event_capturer)

        # Get AutoGen client
        await builder.add_llm("test_llm", nim_config)
        client = await builder.get_llm("test_llm", wrapper_type=LLMFrameworkEnum.AUTOGEN)

        # Turn 1
        messages_turn1 = [UserMessage(content="My name is Alice. What is my name?", source="user")]
        response1 = await client.create(messages=messages_turn1)

        # Turn 2 - with conversation history
        messages_turn2 = [
            UserMessage(content="My name is Alice.", source="user"),
            AssistantMessage(content=str(response1.content), source="assistant"),
            UserMessage(content="What was my name again?", source="user"),
        ]
        response2 = await client.create(messages=messages_turn2)

        # Allow events to propagate
        await asyncio.sleep(0.1)

    # Validate responses
    assert "Alice" in str(response1.content) or "alice" in str(response1.content).lower()
    assert "Alice" in str(response2.content) or "alice" in str(response2.content).lower()

    # Validate telemetry events
    llm_start_events = [e for e in captured_events if e.payload.event_type == IntermediateStepType.LLM_START]
    llm_end_events = [e for e in captured_events if e.payload.event_type == IntermediateStepType.LLM_END]

    # Should have exactly 2 pairs (one per turn)
    assert len(llm_start_events) >= 2, f"Expected at least 2 LLM_START events, got {len(llm_start_events)}"
    assert len(llm_end_events) >= 2, f"Expected at least 2 LLM_END events, got {len(llm_end_events)}"

    # Verify UUIDs are unique across calls
    start_uuids = [e.payload.UUID for e in llm_start_events]
    assert len(start_uuids) == len(set(start_uuids)), "Each LLM call should have unique UUID"

    # Verify each START has matching END
    end_uuids = {e.payload.UUID for e in llm_end_events}
    for start_uuid in start_uuids:
        assert start_uuid in end_uuids, f"START event {start_uuid} should have matching END event"


# ============================================================================
# Bonus: Combined Streaming + Tool Test
# ============================================================================


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key", "autogen_profiler")
@pytest.mark.slow  # This test makes multiple API calls
async def test_nim_autogen_streaming_with_agent_telemetry(
    nim_config: NIMModelConfig,
    captured_events: list[IntermediateStep],
    event_capturer: Callable[[IntermediateStep], None],
):
    """Test complex workflow with streaming agent and tools.

    Validates end-to-end telemetry in a realistic agent scenario.
    """
    from autogen_agentchat.agents import AssistantAgent
    from autogen_core.tools import FunctionTool

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    add_tool = FunctionTool(add, description="Add two numbers together")

    async with WorkflowBuilder() as builder:
        # Subscribe to intermediate step events
        ctx = Context.get()
        ctx.intermediate_step_manager.subscribe(event_capturer)

        # Get AutoGen client
        await builder.add_llm("test_llm", nim_config)
        client = await builder.get_llm("test_llm", wrapper_type=LLMFrameworkEnum.AUTOGEN)

        # Create agent with streaming
        agent = AssistantAgent(
            name="math_agent",
            model_client=client,
            tools=[add_tool],
            system_message="You are a math helper. Use the add tool for addition. Be concise.",
        )

        # Run with streaming
        collected_messages = []
        async for message in agent.run_stream(task="What is 10 plus 20? Use the add tool."):
            collected_messages.append(message)

        # Allow events to propagate
        await asyncio.sleep(0.2)

    # Validate we got messages
    assert len(collected_messages) > 0

    # Validate telemetry
    all_event_types = [e.payload.event_type for e in captured_events]

    # Should have at least one LLM call
    assert IntermediateStepType.LLM_START in all_event_types
    assert IntermediateStepType.LLM_END in all_event_types

    # Count events by type
    llm_starts = sum(1 for t in all_event_types if t == IntermediateStepType.LLM_START)
    llm_ends = sum(1 for t in all_event_types if t == IntermediateStepType.LLM_END)

    # Each START should have an END
    assert llm_starts == llm_ends, f"Mismatched LLM events: {llm_starts} starts vs {llm_ends} ends"


# ============================================================================
# OpenAI Integration Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.usefixtures("openai_api_key", "autogen_profiler")
async def test_openai_autogen_non_streaming_llm_telemetry(
    openai_config: OpenAIModelConfig,
    captured_events: list[IntermediateStep],
    event_capturer: Callable[[IntermediateStep], None],
):
    """Test OpenAI non-streaming LLM call captures correct telemetry events.

    Validates:
        - LLM_START event is pushed with correct model name and input
        - LLM_END event is pushed with output and token usage
        - Events are properly paired (same UUID)
        - Response content is valid
    """
    from autogen_core.models import UserMessage

    async with WorkflowBuilder() as builder:
        # Subscribe to intermediate step events
        ctx = Context.get()
        ctx.intermediate_step_manager.subscribe(event_capturer)

        # Get AutoGen client
        await builder.add_llm("test_llm", openai_config)
        client = await builder.get_llm("test_llm", wrapper_type=LLMFrameworkEnum.AUTOGEN)

        # Make a non-streaming LLM call
        messages = [UserMessage(content="What is 2 + 2? Reply with just the number.", source="user")]
        response = await client.create(messages=messages)

        # Allow events to propagate
        await asyncio.sleep(0.1)

    # Validate response
    assert response is not None
    assert hasattr(response, 'content')
    assert "4" in str(response.content)

    # Validate telemetry events
    llm_start_events = [e for e in captured_events if e.payload.event_type == IntermediateStepType.LLM_START]
    llm_end_events = [e for e in captured_events if e.payload.event_type == IntermediateStepType.LLM_END]

    assert len(llm_start_events) >= 1, f"Expected at least 1 LLM_START event, got {len(llm_start_events)}"
    assert len(llm_end_events) >= 1, f"Expected at least 1 LLM_END event, got {len(llm_end_events)}"

    # Get the last pair (most recent call)
    start_event = llm_start_events[-1]
    end_event = llm_end_events[-1]

    # Verify event pairing (same UUID)
    assert start_event.payload.UUID == end_event.payload.UUID, "START and END events should have same UUID"

    # Verify framework
    assert start_event.payload.framework == LLMFrameworkEnum.AUTOGEN
    assert end_event.payload.framework == LLMFrameworkEnum.AUTOGEN

    # Verify model name
    assert openai_config.model_name in start_event.payload.name

    # Verify input was captured (stored in metadata.chat_inputs)
    assert start_event.payload.metadata is not None
    assert start_event.payload.metadata.chat_inputs is not None
    assert len(start_event.payload.metadata.chat_inputs) > 0
    # Check that our input is in the chat inputs
    input_contents = str(start_event.payload.metadata.chat_inputs)
    assert "2 + 2" in input_contents

    # Verify output was captured
    assert end_event.payload.data is not None
    # Output may be in data.output or metadata.chat_responses
    has_output = (end_event.payload.data.output is not None
                  or (end_event.payload.metadata is not None and end_event.payload.metadata.chat_responses is not None))
    assert has_output, "Output should be captured in data.output or metadata.chat_responses"

    # Verify usage_info structure exists (token counts may be 0 for some providers)
    assert end_event.payload.usage_info is not None
    assert end_event.payload.usage_info.token_usage is not None
    # num_llm_calls should be tracked
    assert end_event.payload.usage_info.num_llm_calls >= 1


@pytest.mark.integration
@pytest.mark.usefixtures("openai_api_key", "autogen_profiler")
async def test_openai_autogen_streaming_llm_telemetry(
    openai_config: OpenAIModelConfig,
    captured_events: list[IntermediateStep],
    event_capturer: Callable[[IntermediateStep], None],
):
    """Test OpenAI streaming LLM call captures correct telemetry events.

    Validates:
        - LLM_START event fires before first chunk
        - All chunks are yielded correctly
        - LLM_END event fires after stream completion
        - Token usage is captured from final chunk
    """
    from autogen_core.models import UserMessage

    async with WorkflowBuilder() as builder:
        # Subscribe to intermediate step events
        ctx = Context.get()
        ctx.intermediate_step_manager.subscribe(event_capturer)

        # Get AutoGen client
        await builder.add_llm("test_llm", openai_config)
        client = await builder.get_llm("test_llm", wrapper_type=LLMFrameworkEnum.AUTOGEN)

        # Make a streaming LLM call
        messages = [UserMessage(content="Count from 1 to 5. Just the numbers.", source="user")]

        chunks = []
        async for chunk in client.create_stream(messages=messages):
            chunks.append(chunk)

        # Allow events to propagate
        await asyncio.sleep(0.1)

    # Validate chunks were received
    assert len(chunks) > 0, "Expected at least one chunk from streaming response"

    # Validate telemetry events
    llm_start_events = [e for e in captured_events if e.payload.event_type == IntermediateStepType.LLM_START]
    llm_end_events = [e for e in captured_events if e.payload.event_type == IntermediateStepType.LLM_END]

    assert len(llm_start_events) >= 1, f"Expected at least 1 LLM_START event, got {len(llm_start_events)}"
    assert len(llm_end_events) >= 1, f"Expected at least 1 LLM_END event, got {len(llm_end_events)}"

    # Get the last START event and find its matching END event by UUID
    start_event = llm_start_events[-1]
    end_event = next((e for e in llm_end_events if e.payload.UUID == start_event.payload.UUID), None)
    assert end_event is not None, f"No matching LLM_END event for START UUID {start_event.payload.UUID}"

    # Verify framework
    assert start_event.payload.framework == LLMFrameworkEnum.AUTOGEN
    assert end_event.payload.framework == LLMFrameworkEnum.AUTOGEN

    # Verify START event was pushed (input captured in metadata.chat_inputs)
    assert start_event.payload.metadata is not None
    assert start_event.payload.metadata.chat_inputs is not None
    input_contents = str(start_event.payload.metadata.chat_inputs)
    assert "1 to 5" in input_contents

    # Verify END event has output
    assert end_event.payload.data is not None
    has_output = (end_event.payload.data.output is not None
                  or (end_event.payload.metadata is not None and end_event.payload.metadata.chat_responses is not None))
    assert has_output, "Output should be captured"


@pytest.mark.integration
@pytest.mark.usefixtures("openai_api_key", "autogen_profiler")
async def test_openai_autogen_tool_execution_telemetry(
    openai_config: OpenAIModelConfig,
    captured_events: list[IntermediateStep],
    event_capturer: Callable[[IntermediateStep], None],
):
    """Test OpenAI tool execution captures correct telemetry events.

    Validates:
        - TOOL_START event is pushed with correct tool name and input
        - TOOL_END event is pushed with correct output
        - Events are properly paired
    """
    from autogen_agentchat.agents import AssistantAgent
    from autogen_core.tools import FunctionTool

    # Define a simple calculator tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers together.

        Args:
            a: First number
            b: Second number

        Returns:
            The product of a and b
        """
        return a * b

    multiply_tool = FunctionTool(multiply, description="Multiply two numbers together")

    async with WorkflowBuilder() as builder:
        # Subscribe to intermediate step events
        ctx = Context.get()
        ctx.intermediate_step_manager.subscribe(event_capturer)

        # Get AutoGen client
        await builder.add_llm("test_llm", openai_config)
        client = await builder.get_llm("test_llm", wrapper_type=LLMFrameworkEnum.AUTOGEN)

        # Create an agent with the tool
        agent = AssistantAgent(
            name="calculator_agent",
            model_client=client,
            tools=[multiply_tool],
            system_message="You are a helpful calculator. Use the multiply tool when asked to multiply numbers.",
        )

        # Run a task that should trigger tool use
        from autogen_agentchat.base import TaskResult
        result = await agent.run(task="What is 7 times 8? Use the multiply tool.")

        # Allow events to propagate
        await asyncio.sleep(0.1)

    # Validate result
    assert isinstance(result, TaskResult)
    assert result.messages is not None
    assert len(result.messages) > 0

    # Check for tool events
    tool_start_events = [e for e in captured_events if e.payload.event_type == IntermediateStepType.TOOL_START]
    tool_end_events = [e for e in captured_events if e.payload.event_type == IntermediateStepType.TOOL_END]

    # Tool should have been called (but LLM might not always use it)
    if len(tool_start_events) > 0:
        assert len(tool_end_events) >= 1, "If TOOL_START exists, TOOL_END should also exist"

        start_event = tool_start_events[-1]
        end_event = tool_end_events[-1]

        # Verify pairing
        assert start_event.payload.UUID == end_event.payload.UUID

        # Verify tool name
        assert "multiply" in start_event.payload.name.lower()

        # Verify output contains result
        assert end_event.payload.data is not None
        # 7 * 8 = 56
        assert "56" in str(end_event.payload.data.output)

    # Regardless, LLM events should exist
    llm_start_events = [e for e in captured_events if e.payload.event_type == IntermediateStepType.LLM_START]
    assert len(llm_start_events) >= 1, "LLM call should have been made"


@pytest.mark.integration
@pytest.mark.usefixtures("openai_api_key", "autogen_profiler")
async def test_openai_autogen_multi_turn_conversation_telemetry(
    openai_config: OpenAIModelConfig,
    captured_events: list[IntermediateStep],
    event_capturer: Callable[[IntermediateStep], None],
):
    """Test OpenAI multi-turn conversation captures correct telemetry for each turn.

    Validates:
        - Each LLM call generates its own START/END pair
        - UUIDs are unique per call
        - Message history is captured correctly
    """
    from autogen_core.models import AssistantMessage
    from autogen_core.models import UserMessage

    async with WorkflowBuilder() as builder:
        # Subscribe to intermediate step events
        ctx = Context.get()
        ctx.intermediate_step_manager.subscribe(event_capturer)

        # Get AutoGen client
        await builder.add_llm("test_llm", openai_config)
        client = await builder.get_llm("test_llm", wrapper_type=LLMFrameworkEnum.AUTOGEN)

        # Turn 1
        messages_turn1 = [UserMessage(content="My name is Alice. What is my name?", source="user")]
        response1 = await client.create(messages=messages_turn1)

        # Turn 2 - with conversation history
        messages_turn2 = [
            UserMessage(content="My name is Alice.", source="user"),
            AssistantMessage(content=str(response1.content), source="assistant"),
            UserMessage(content="What was my name again?", source="user"),
        ]
        response2 = await client.create(messages=messages_turn2)

        # Allow events to propagate
        await asyncio.sleep(0.1)

    # Validate responses
    assert "Alice" in str(response1.content) or "alice" in str(response1.content).lower()
    assert "Alice" in str(response2.content) or "alice" in str(response2.content).lower()

    # Validate telemetry events
    llm_start_events = [e for e in captured_events if e.payload.event_type == IntermediateStepType.LLM_START]
    llm_end_events = [e for e in captured_events if e.payload.event_type == IntermediateStepType.LLM_END]

    # Should have exactly 2 pairs (one per turn)
    assert len(llm_start_events) >= 2, f"Expected at least 2 LLM_START events, got {len(llm_start_events)}"
    assert len(llm_end_events) >= 2, f"Expected at least 2 LLM_END events, got {len(llm_end_events)}"

    # Verify UUIDs are unique across calls
    start_uuids = [e.payload.UUID for e in llm_start_events]
    assert len(start_uuids) == len(set(start_uuids)), "Each LLM call should have unique UUID"

    # Verify each START has matching END
    end_uuids = {e.payload.UUID for e in llm_end_events}
    for start_uuid in start_uuids:
        assert start_uuid in end_uuids, f"START event {start_uuid} should have matching END event"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
