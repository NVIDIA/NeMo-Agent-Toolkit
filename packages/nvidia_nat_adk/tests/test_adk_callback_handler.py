# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import time
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import PropertyMock
from unittest.mock import patch

import pytest

from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import LLMFrameworkEnum
from nat.data_models.profiler_callback import BaseProfilerCallback
from nat.plugins.adk.callback_handler import ADKProfilerHandler


# ----------------------------
# Test Fixtures and Helpers
# ----------------------------
@pytest.fixture(autouse=True)
def reset_patches():
    import litellm
    from google.adk.tools.function_tool import FunctionTool

    # Store original functions
    original_acompletion = litellm.acompletion
    original_function_tool_run_async = FunctionTool.run_async

    yield

    # Restore original functions
    litellm.acompletion = original_acompletion
    FunctionTool.run_async = original_function_tool_run_async


@pytest.fixture
def mock_context():
    """Mock context with intermediate step manager."""
    with patch('nat.plugins.adk.callback_handler.Context') as mock_context_class:
        mock_context_instance = MagicMock()
        mock_step_manager = MagicMock()
        mock_context_instance.intermediate_step_manager = mock_step_manager
        mock_context_class.get.return_value = mock_context_instance
        yield mock_step_manager


@pytest.fixture
def handler(mock_context: MagicMock) -> ADKProfilerHandler:
    """Create ADKProfilerHandler instance for testing."""
    return ADKProfilerHandler()


# ----------------------------
# Pytest Unit Tests
# ----------------------------


def test_no_double_patching():
    a1 = ADKProfilerHandler()
    a2 = ADKProfilerHandler()
    a1.instrument()
    a2.instrument()
    assert a1._original_llm_call is a2._original_llm_call
    assert a1._original_tool_call is a2._original_tool_call


def test_uninstrument_restores_originals():
    import litellm
    from google.adk.tools.function_tool import FunctionTool

    original_acompletion = litellm.acompletion
    original_function_tool_run_async = FunctionTool.run_async

    handler = ADKProfilerHandler()
    handler.instrument()
    assert handler._instrumented
    assert handler._original_llm_call is original_acompletion
    assert handler._original_tool_call is original_function_tool_run_async

    handler.uninstrument()
    assert not handler._instrumented
    assert handler._original_llm_call is None
    assert handler._original_tool_call is None


def test_adk_profiler_handler_initialization(handler, mock_context):
    """Test ADKProfilerHandler initialization."""
    assert handler._original_tool_call is None
    assert handler._original_llm_call is None
    assert handler.step_manager == mock_context
    assert hasattr(handler, '_lock')
    assert hasattr(handler, 'last_call_ts')


@patch('litellm.acompletion')
def test_instrument_patches_litellm(mock_acompletion, handler):
    """Test that instrument method patches litellm.acompletion."""
    # Setup mock
    mock_acompletion.return_value = AsyncMock()

    # Call instrument
    handler.instrument()

    # Verify original was saved
    assert handler._original_llm_call == mock_acompletion

    # Verify litellm.acompletion was replaced (by checking it's been wrapped)
    import litellm
    assert litellm.acompletion != mock_acompletion


@patch('litellm.acompletion')
@pytest.mark.asyncio
async def test_llm_call_monkey_patch(mock_acompletion, handler, mock_context):
    """Test the LLM call monkey patch functionality."""
    # Setup mocks
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
    mock_response.model_extra = {
        'usage':
            MagicMock(model_dump=MagicMock(return_value={
                'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15
            }))
    }
    mock_response.choices[0].model_dump.return_value = {"role": "assistant", "content": "Test response"}
    mock_acompletion.return_value = mock_response

    # Instrument and get the wrapped function
    handler.instrument()
    import litellm
    wrapped_func = litellm.acompletion

    # Prepare test arguments
    test_kwargs = {'model': 'gpt-5.4-mini', 'messages': [{'content': 'Hello, world!'}]}

    # Call the wrapped function
    result = await wrapped_func(**test_kwargs)

    # Verify original function was called
    mock_acompletion.assert_called_once_with(**test_kwargs)

    # Verify intermediate steps were pushed (start and end events)
    assert mock_context.push_intermediate_step.call_count == 2

    # Verify start event
    start_call = mock_context.push_intermediate_step.call_args_list[0][0][0]
    assert start_call.event_type == IntermediateStepType.LLM_START
    assert start_call.framework == LLMFrameworkEnum.ADK
    assert start_call.name == 'gpt-5.4-mini'
    assert start_call.data.input == 'Hello, world!'

    # Verify end event
    end_call = mock_context.push_intermediate_step.call_args_list[1][0][0]
    assert end_call.event_type == IntermediateStepType.LLM_END
    assert end_call.framework == LLMFrameworkEnum.ADK
    assert end_call.name == 'gpt-5.4-mini'
    assert end_call.data.output == 'Test response'

    # Verify response is returned
    assert result == mock_response


async def test_adk_handler_failed_llm_call_ends_llm_step(handler, mock_context):
    """A failed litellm.acompletion call is closed with LLM_END and its error reaches the caller."""
    import litellm

    handler.instrument()

    with pytest.raises(litellm.RateLimitError):
        await litellm.acompletion(model="openai/gpt-4o-mini",
                                  messages=[{
                                      "role": "user", "content": "hi"
                                  }],
                                  api_key="x",
                                  mock_response="litellm.RateLimitError")

    steps = [c[0][0] for c in mock_context.push_intermediate_step.call_args_list]
    assert [s.event_type for s in steps] == [IntermediateStepType.LLM_START, IntermediateStepType.LLM_END]
    start, end = steps
    assert end.UUID == start.UUID
    assert end.name == "openai/gpt-4o-mini"
    assert end.data.output == ""
    assert end.usage_info.token_usage.total_tokens == 0
    # The end event carries the time the call started
    assert end.span_event_timestamp <= start.event_timestamp


async def test_adk_handler_cancelled_calls_end_their_steps(handler, mock_context):
    """A cancelled LLM or tool call is closed with its end event and the cancellation propagates."""
    import asyncio

    handler._original_llm_call = AsyncMock(side_effect=asyncio.CancelledError())
    handler._original_tool_call = AsyncMock(side_effect=asyncio.CancelledError())
    tool = MagicMock()
    tool.name = "test_tool"

    with pytest.raises(asyncio.CancelledError):
        await handler._llm_call_monkey_patch()(model="gpt-5.4-mini", messages=[{"content": "hi"}])
    with pytest.raises(asyncio.CancelledError):
        await handler._tool_use_monkey_patch()(tool, args={})

    steps = [c[0][0] for c in mock_context.push_intermediate_step.call_args_list]
    assert [s.event_type for s in steps] == [
        IntermediateStepType.LLM_START,
        IntermediateStepType.LLM_END,
        IntermediateStepType.TOOL_START,
        IntermediateStepType.TOOL_END,
    ]
    assert steps[1].UUID == steps[0].UUID
    assert steps[3].UUID == steps[2].UUID


async def test_adk_handler_end_event_error_keeps_provider_error(handler, mock_context):
    """If recording the end event of a failed call raises, the caller still gets the provider's error."""

    def push(step):
        if step.event_type == IntermediateStepType.LLM_END:
            raise RuntimeError("event stream failed")

    mock_context.push_intermediate_step.side_effect = push
    handler._original_llm_call = AsyncMock(side_effect=ValueError("provider error"))

    with pytest.raises(ValueError, match="provider error"):
        await handler._llm_call_monkey_patch()(model="gpt-5.4-mini", messages=[{"content": "hi"}])


async def test_adk_handler_streamed_llm_call_records_output(handler, mock_context):
    """A streamed litellm.acompletion call records LLM_END with its output and usage once the stream ends."""
    import litellm

    handler.instrument()

    stream = await litellm.acompletion(model="openai/gpt-4o-mini",
                                       messages=[{
                                           "role": "user", "content": "hi"
                                       }],
                                       api_key="x",
                                       mock_response="Hi there",
                                       stream=True,
                                       stream_options={"include_usage": True})

    # The end event is recorded only after the caller has consumed the stream
    assert mock_context.push_intermediate_step.call_count == 1
    text = "".join([chunk.choices[0].delta.content or "" async for chunk in stream if chunk.choices])
    assert text == "Hi there"

    steps = [c[0][0] for c in mock_context.push_intermediate_step.call_args_list]
    assert [s.event_type for s in steps] == [IntermediateStepType.LLM_START, IntermediateStepType.LLM_END]
    start, end = steps
    assert end.UUID == start.UUID
    assert end.data.output == "Hi there"
    assert end.usage_info.token_usage.completion_tokens > 0
    assert end.span_event_timestamp <= start.event_timestamp


async def test_adk_handler_closed_stream_ends_llm_step(handler, mock_context):
    """Closing a streamed call before it is exhausted records LLM_END with the chunks received so far."""
    import litellm

    handler.instrument()

    stream = await litellm.acompletion(model="openai/gpt-4o-mini",
                                       messages=[{
                                           "role": "user", "content": "hi"
                                       }],
                                       api_key="x",
                                       mock_response="Hi there",
                                       stream=True)
    first = await stream.__anext__()
    await stream.aclose()

    steps = [c[0][0] for c in mock_context.push_intermediate_step.call_args_list]
    assert [s.event_type for s in steps] == [IntermediateStepType.LLM_START, IntermediateStepType.LLM_END]
    assert steps[1].data.output == first.choices[0].delta.content


@pytest.mark.asyncio
async def test_tool_use_monkey_patch_functionality(handler, mock_context):
    """Test the tool use monkey patch functionality."""
    # Create a mock tool instance
    mock_tool_instance = MagicMock()
    mock_tool_instance.name = "test_tool"

    # Create mock original function
    mock_original_func = AsyncMock(return_value="tool_result")
    handler._original_tool_call = mock_original_func

    # Get the wrapped function
    wrapped_func = handler._tool_use_monkey_patch()

    # Test arguments
    test_args = ("arg1", "arg2")
    test_kwargs = {"args": {"param1": "value1"}}

    # Call the wrapped function
    result = await wrapped_func(mock_tool_instance, *test_args, **test_kwargs)

    # Verify original function was called
    mock_original_func.assert_called_once_with(mock_tool_instance, *test_args, **test_kwargs)

    # Verify intermediate steps were pushed (start and end events)
    assert mock_context.push_intermediate_step.call_count == 2

    # Verify start event
    start_call = mock_context.push_intermediate_step.call_args_list[0][0][0]
    assert start_call.event_type == IntermediateStepType.TOOL_START
    assert start_call.framework == LLMFrameworkEnum.ADK
    assert start_call.name == "test_tool"

    # Verify end event
    end_call = mock_context.push_intermediate_step.call_args_list[1][0][0]
    assert end_call.event_type == IntermediateStepType.TOOL_END
    assert end_call.framework == LLMFrameworkEnum.ADK
    assert end_call.name == "test_tool"
    assert end_call.data.output == "tool_result"

    # Verify result is returned
    assert result == "tool_result"


@pytest.mark.asyncio
async def test_tool_use_monkey_patch_with_exception(handler, mock_context):
    """Test tool use monkey patch handles exceptions properly."""
    # Create a mock tool instance
    mock_tool_instance = MagicMock()
    mock_tool_instance.name = "test_tool"

    # Create mock original function that raises an exception
    mock_original_func = AsyncMock(side_effect=Exception("Tool error"))
    handler._original_tool_call = mock_original_func

    # Get the wrapped function
    wrapped_func = handler._tool_use_monkey_patch()

    # Test that exception is re-raised
    with pytest.raises(Exception, match="Tool error"):
        await wrapped_func(mock_tool_instance, "arg1")

    # Verify original function was called
    mock_original_func.assert_called_once()

    # Verify the start event was pushed and closed by an end event with the same UUID
    assert mock_context.push_intermediate_step.call_count == 2
    start_call = mock_context.push_intermediate_step.call_args_list[0][0][0]
    assert start_call.event_type == IntermediateStepType.TOOL_START
    end_call = mock_context.push_intermediate_step.call_args_list[1][0][0]
    assert end_call.event_type == IntermediateStepType.TOOL_END
    assert end_call.UUID == start_call.UUID
    assert end_call.name == "test_tool"


@pytest.mark.asyncio
async def test_tool_use_monkey_patch_tool_name_error(handler, mock_context):
    """Test tool use monkey patch handles tool name retrieval errors."""
    # Create a mock tool instance that raises error when accessing name
    mock_tool_instance = MagicMock()
    # Make .name attribute access raise an exception
    type(mock_tool_instance).name = PropertyMock(side_effect=Exception("Name error"))

    # Create mock original function
    mock_original_func = AsyncMock(return_value="tool_result")
    handler._original_tool_call = mock_original_func

    # Get the wrapped function
    wrapped_func = handler._tool_use_monkey_patch()

    # Call should still work despite name error
    result = await wrapped_func(mock_tool_instance, "arg1")

    # Verify result is returned
    assert result == "tool_result"

    # Verify intermediate steps were still pushed with empty tool name
    assert mock_context.push_intermediate_step.call_count == 2
    start_call = mock_context.push_intermediate_step.call_args_list[0][0][0]
    assert start_call.name == ""  # Empty due to error


@patch('litellm.acompletion')
@pytest.mark.asyncio
async def test_llm_call_monkey_patch_with_multiple_messages(mock_acompletion, handler, mock_context):
    """Test LLM call monkey patch with multiple messages."""
    # Setup mocks
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Response 1")), MagicMock(message=MagicMock(content="Response 2"))
    ]
    mock_response.model_extra = {
        'usage':
            MagicMock(model_dump=MagicMock(return_value={
                'prompt_tokens': 20, 'completion_tokens': 10, 'total_tokens': 30
            }))
    }
    mock_response.choices[0].model_dump.return_value = {"role": "assistant", "content": "Response 1"}
    mock_acompletion.return_value = mock_response

    handler.instrument()
    import litellm
    wrapped_func = litellm.acompletion

    # Test with multiple messages
    test_kwargs = {
        'model': 'gpt-4',
        'messages': [
            {
                'content': 'Message 1'
            },
            {
                'content': 'Message 2'
            },
            {
                'content': None
            },  # Test None content
        ]
    }

    await wrapped_func(**test_kwargs)

    # Verify input concatenation
    start_call = mock_context.push_intermediate_step.call_args_list[0][0][0]
    # assert start_call.data.input == 'Message 1Message 2'  # None content should be skipped
    assert 'Message 1' in start_call.data.input and 'Message 2' in start_call.data.input
    assert start_call.data.input.index('Message 1') < start_call.data.input.index('Message 2')  # preserves order

    # Verify output concatenation
    end_call = mock_context.push_intermediate_step.call_args_list[1][0][0]
    # assert end_call.data.output == 'Response 1Response 2'
    assert 'Response 1' in end_call.data.output and 'Response 2' in end_call.data.output
    assert end_call.data.output.index('Response 1') < end_call.data.output.index('Response 2')


def test_handler_inheritance(handler):
    """Test that ADKProfilerHandler inherits from BaseProfilerCallback."""
    assert isinstance(handler, BaseProfilerCallback)


def test_handler_thread_safety(handler):
    """Test that handler has thread safety mechanisms."""
    import threading

    assert isinstance(handler._lock, type(threading.Lock()))


def test_last_call_timestamp_initialization(handler):
    """Test that last_call_ts is initialized properly."""
    assert isinstance(handler.last_call_ts, float)
    assert handler.last_call_ts <= time.time()
