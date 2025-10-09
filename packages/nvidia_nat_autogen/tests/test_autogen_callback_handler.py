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
"""Test AutoGen Callback Handler"""

import threading
import time
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import PropertyMock
from unittest.mock import patch

import pytest

from nat.plugins.autogen.autogen_callback_handler import AutoGenProfilerHandler


class TestAutoGenProfilerHandler:
    """Test cases for AutoGenProfilerHandler."""

    def test_init(self):
        """Test initialization of AutoGenProfilerHandler."""
        handler = AutoGenProfilerHandler()
        assert isinstance(handler._lock, type(threading.Lock()))  # pylint: disable=protected-access
        assert isinstance(handler.last_call_ts, float)
        assert handler._original_tool_call is None  # pylint: disable=protected-access
        assert handler._original_llm_call is None  # pylint: disable=protected-access
        assert handler._instrumented is False  # pylint: disable=protected-access

    def test_instrument_already_instrumented(self):
        """Test instrument when already instrumented."""
        handler = AutoGenProfilerHandler()
        handler._instrumented = True  # pylint: disable=protected-access

        with patch('nat.plugins.autogen.autogen_callback_handler.logger') as mock_logger:
            handler.instrument()
            mock_logger.debug.assert_called_with("AutoGenProfilerHandler already instrumented; skipping.")

    def test_instrument_import_failure(self):
        """Test instrument when AutoGen imports fail."""
        handler = AutoGenProfilerHandler()
        # Force import failure by patching the import to raise an exception
        with patch('nat.plugins.autogen.autogen_callback_handler.logger') as mock_logger:
            with patch('builtins.__import__', side_effect=ImportError("Mock import error")):
                handler.instrument()
                mock_logger.exception.assert_called_with("AutoGen import failed; skipping instrumentation")

    def test_uninstrument_failure(self):
        """Test uninstrument with exception."""
        handler = AutoGenProfilerHandler()
        handler._instrumented = True  # pylint: disable=protected-access
        handler._original_llm_call = Mock()  # pylint: disable=protected-access

        with patch('nat.plugins.autogen.autogen_callback_handler.logger') as mock_logger:
            # Force an exception by making the import fail
            with patch('builtins.__import__', side_effect=ImportError("Mock import error")):
                handler.uninstrument()
                mock_logger.exception.assert_called_with("Failed to uninstrument AutoGenProfilerHandler")

    def test_lock_mechanism(self):
        """Test that lock mechanism works properly."""
        handler = AutoGenProfilerHandler()
        # Test that we can acquire the lock
        with handler._lock:  # pylint: disable=protected-access
            # Lock should be held
            pass

        # Lock should be released
        assert handler._lock.acquire(blocking=False)  # pylint: disable=protected-access
        handler._lock.release()  # pylint: disable=protected-access

    def test_last_call_timestamp_update(self):
        """Test that last call timestamp is updated correctly."""
        handler = AutoGenProfilerHandler()
        original_ts = handler.last_call_ts
        time.sleep(0.01)  # Small delay

        # Simulate updating timestamp
        with handler._lock:  # pylint: disable=protected-access
            handler.last_call_ts = time.time()

        assert handler.last_call_ts > original_ts

    @patch('nat.plugins.autogen.autogen_callback_handler.Context.get')
    def test_step_manager_property(self, mock_get: Mock):
        """Test step_manager property access.

        Args:
            mock_get (Mock): Mock for Context.get method.
        """
        mock_context = Mock()
        mock_step_manager = Mock()
        mock_context.intermediate_step_manager = mock_step_manager
        mock_get.return_value = mock_context

        handler = AutoGenProfilerHandler()
        assert handler.step_manager == mock_step_manager

    def test_monkey_patch_functions_exist(self):
        """Test that monkey patch functions can be created."""
        handler = AutoGenProfilerHandler()
        llm_patch = handler._llm_call_monkey_patch()  # pylint: disable=protected-access
        tool_patch = handler._tool_call_monkey_patch()  # pylint: disable=protected-access

        assert callable(llm_patch)
        assert callable(tool_patch)

    def test_time_tracking(self):
        """Test last call timestamp tracking."""
        handler = AutoGenProfilerHandler()
        original_ts = handler.last_call_ts

        # Update timestamp
        time.sleep(0.001)  # Small delay to ensure timestamp changes
        handler.last_call_ts = time.time()

        assert handler.last_call_ts > original_ts

    def test_context_integration(self):
        """Test context and step manager integration."""
        with patch('nat.plugins.autogen.autogen_callback_handler.Context.get') as mock_get:
            mock_context = Mock()
            mock_step_manager = Mock()
            mock_context.intermediate_step_manager = mock_step_manager
            mock_get.return_value = mock_context

            handler = AutoGenProfilerHandler()
            assert handler.step_manager == mock_step_manager

    @patch('nat.plugins.autogen.autogen_callback_handler.logger')
    def test_successful_instrumentation(self, mock_logger):
        """Test successful instrumentation path.

        Args:
            mock_logger (Mock): Mock for the logger.
        """
        _ = mock_logger  # Unused in this test
        handler = AutoGenProfilerHandler()

        with patch('builtins.__import__') as mock_import:
            # Mock successful imports
            mock_autogen_core = Mock()
            mock_autogen_ext = Mock()
            mock_base_tool = Mock()
            mock_client = Mock()

            mock_autogen_core.tools.BaseTool = mock_base_tool
            mock_autogen_ext.models.openai.OpenAIChatCompletionClient = mock_client

            def side_effect(name, *args, **kwargs):
                if 'autogen_core.tools' in name:
                    return mock_autogen_core.tools
                elif 'autogen_ext.models.openai' in name:
                    return mock_autogen_ext.models.openai
                return Mock()

            mock_import.side_effect = side_effect

            # Mock original methods
            mock_base_tool.run_json = Mock()
            mock_client.create = Mock()

            handler.instrument()

            # Verify instrumentation succeeded
            assert handler._instrumented  # pylint: disable=protected-access
            assert handler._original_tool_call is not None  # pylint: disable=protected-access
            assert handler._original_llm_call is not None  # pylint: disable=protected-access

    @patch('nat.plugins.autogen.autogen_callback_handler.logger')
    def test_successful_uninstrumentation(self, mock_logger):
        """Test successful uninstrumentation path.

        Args:
            mock_logger (Mock): Mock for the logger.
        """
        handler = AutoGenProfilerHandler()
        handler._instrumented = True  # pylint: disable=protected-access

        # Mock original methods
        mock_original_llm = Mock()
        mock_original_tool = Mock()
        handler._original_llm_call = mock_original_llm  # pylint: disable=protected-access
        handler._original_tool_call = mock_original_tool  # pylint: disable=protected-access

        with patch('builtins.__import__') as mock_import:
            # Mock successful imports
            mock_autogen_core = Mock()
            mock_autogen_ext = Mock()
            mock_base_tool = Mock()
            mock_client = Mock()

            mock_autogen_core.tools.BaseTool = mock_base_tool
            mock_autogen_ext.models.openai.OpenAIChatCompletionClient = mock_client

            def side_effect(name, *args, **kwargs):
                if 'autogen_core.tools' in name:
                    return mock_autogen_core.tools
                elif 'autogen_ext.models.openai' in name:
                    return mock_autogen_ext.models.openai
                return Mock()

            mock_import.side_effect = side_effect

            handler.uninstrument()

            # Verify restoration
            assert mock_client.create == mock_original_llm
            assert mock_base_tool.run_json == mock_original_tool
            assert not handler._instrumented  # pylint: disable=protected-access
            mock_logger.debug.assert_called_with("AutoGenProfilerHandler uninstrumented successfully.")


@pytest.mark.asyncio
async def test_integration_flow():
    """Test the complete integration flow."""
    handler = AutoGenProfilerHandler()

    # Test that handler starts uninitialized
    assert not handler._instrumented  # pylint: disable=protected-access

    # Test instrument/uninstrument cycle
    handler.instrument()  # Should handle missing imports gracefully
    handler.uninstrument()  # Should handle gracefully


class TestLLMCallMonkeyPatch:
    """Test LLM call monkey patch functionality."""

    def test_llm_call_monkey_patch_creation(self):
        """Test that LLM call monkey patch can be created."""
        handler = AutoGenProfilerHandler()
        patch_func = handler._llm_call_monkey_patch()  # pylint: disable=protected-access
        assert callable(patch_func)

    @pytest.mark.asyncio
    @patch('nat.plugins.autogen.autogen_callback_handler.Context.get')
    @patch('nat.plugins.autogen.autogen_callback_handler.logger')
    async def test_llm_wrapped_call_basic_flow(self, mock_logger: Mock, mock_get: Mock):
        """Test basic LLM wrapped call flow.

        Args:
            mock_logger (Mock): Mock for the logger.
            mock_get (Mock): Mock for Context.get method.
        """
        # Setup mocks
        _ = mock_logger  # Unused in this test
        mock_context = Mock()
        mock_step_manager = Mock()
        mock_context.intermediate_step_manager = mock_step_manager
        mock_get.return_value = mock_context

        handler = AutoGenProfilerHandler()

        # Mock original function
        mock_output = Mock()
        mock_output.content = ["test output"]
        mock_output.choices = []  # Empty choices to avoid processing
        mock_output.usage = None  # No usage to avoid type error
        mock_output.model_extra = {}  # Empty model_extra
        original_func = AsyncMock(return_value=mock_output)
        handler._original_llm_call = original_func  # pylint: disable=protected-access

        # Get wrapped function
        wrapped_func = handler._llm_call_monkey_patch()  # pylint: disable=protected-access

        # Mock args with proper structure
        mock_args = [Mock()]
        mock_args[0]._raw_config = {"model": "gpt-4"}
        mock_args[0].model = "gpt-4"

        kwargs = {"messages": [{"content": "Hello"}, {"content": ["text part", {"text": "nested text"}]}]}

        # Call wrapped function
        await wrapped_func(*mock_args, **kwargs)

        # Verify original function was called
        original_func.assert_called_once_with(*mock_args, **kwargs)

        # Verify step manager interactions
        assert mock_step_manager.push_intermediate_step.call_count == 2  # Start and end events

    @pytest.mark.asyncio
    @patch('nat.plugins.autogen.autogen_callback_handler.Context.get')
    @patch('nat.plugins.autogen.autogen_callback_handler.logger')
    async def test_llm_wrapped_call_with_exception(self, mock_logger: Mock, mock_get: Mock):
        """Test LLM wrapped call with exception handling.

        Args:
            mock_logger (Mock): Mock for the logger.
            mock_get (Mock): Mock for Context.get method.
        """
        # Setup mocks
        mock_context = Mock()
        mock_step_manager = Mock()
        mock_context.intermediate_step_manager = mock_step_manager
        mock_get.return_value = mock_context

        handler = AutoGenProfilerHandler()

        # Mock original function that raises exception
        original_func = AsyncMock(side_effect=Exception("LLM error"))
        handler._original_llm_call = original_func  # pylint: disable=protected-access

        # Get wrapped function
        wrapped_func = handler._llm_call_monkey_patch()  # pylint: disable=protected-access

        # Mock args
        mock_args = [Mock()]
        mock_args[0]._raw_config = {}
        mock_args[0].model = "gpt-4"
        kwargs = {"messages": [{"content": "test"}]}

        # Call wrapped function
        with pytest.raises(Exception):
            await wrapped_func(*mock_args, **kwargs)

        # Verify error handling
        mock_logger.error.assert_called()

    @pytest.mark.asyncio
    @patch('nat.plugins.autogen.autogen_callback_handler.Context.get')
    async def test_llm_wrapped_call_model_name_fallback(self, mock_get: Mock):
        """Test model name fallback when _raw_config fails.

        Args:
            mock_logger (Mock): Mock for the logger.
        """
        # Setup mocks
        mock_context = Mock()
        mock_step_manager = Mock()
        mock_context.intermediate_step_manager = mock_step_manager
        mock_get.return_value = mock_context

        handler = AutoGenProfilerHandler()

        # Mock original function
        mock_output = Mock()
        mock_output.content = ["output"]
        mock_output.choices = []  # Empty choices
        mock_output.usage = None  # No usage
        mock_output.model_extra = {}  # Empty model_extra
        original_func = AsyncMock(return_value=mock_output)
        handler._original_llm_call = original_func  # pylint: disable=protected-access

        # Get wrapped function
        wrapped_func = handler._llm_call_monkey_patch()  # pylint: disable=protected-access

        # Mock args without _raw_config but ensure model returns a string
        mock_args = [Mock()]
        mock_args[0]._raw_config = {}  # Empty dict so get() returns None
        mock_args[0].model = "fallback-model"
        kwargs = {"messages": []}

        # Call wrapped function
        await wrapped_func(*mock_args, **kwargs)

        # Verify call completed
        original_func.assert_called_once()

    @pytest.mark.asyncio
    @patch('nat.plugins.autogen.autogen_callback_handler.Context.get')
    async def test_llm_wrapped_call_complex_content(self, mock_get: Mock):
        """Test LLM wrapped call with complex message content.

        Args:
            mock_get (Mock): Mock for Context.get method.
        """
        # Setup mocks
        mock_context = Mock()
        mock_step_manager = Mock()
        mock_context.intermediate_step_manager = mock_step_manager
        mock_get.return_value = mock_context

        handler = AutoGenProfilerHandler()

        # Mock original function with complex output
        mock_output = Mock()
        mock_output.content = ["part1", "part2"]
        mock_output.choices = [Mock()]
        mock_output.choices[0].model_dump = Mock(return_value={"role": "assistant"})
        mock_output.usage = Mock()
        mock_output.usage.model_dump = Mock(return_value={"total_tokens": 50})

        original_func = AsyncMock(return_value=mock_output)
        handler._original_llm_call = original_func  # pylint: disable=protected-access

        # Get wrapped function
        wrapped_func = handler._llm_call_monkey_patch()  # pylint: disable=protected-access

        # Mock args
        mock_args = [Mock()]
        mock_args[0]._raw_config = {"model": "gpt-4"}
        kwargs = {"messages": [{"content": ["text_part", {"text": "nested_text"}, {"type": "image"}]}]}

        # Call wrapped function
        result = await wrapped_func(*mock_args, **kwargs)

        # Verify result
        assert result == mock_output
        assert mock_step_manager.push_intermediate_step.call_count == 2


class TestToolCallMonkeyPatch:
    """Test tool call monkey patch functionality."""

    def test_tool_call_monkey_patch_creation(self):
        """Test that tool call monkey patch can be created."""
        handler = AutoGenProfilerHandler()
        patch_func = handler._tool_call_monkey_patch()  # pylint: disable=protected-access
        assert callable(patch_func)

    @pytest.mark.asyncio
    @patch('nat.plugins.autogen.autogen_callback_handler.Context.get')
    async def test_tool_wrapped_call_basic_flow(self, mock_get: Mock):
        """Test basic tool wrapped call flow.

        Args:
            mock_get (Mock): Mock for Context.get method.
        """
        # Setup mocks
        mock_context = Mock()
        mock_step_manager = Mock()
        mock_context.intermediate_step_manager = mock_step_manager
        mock_get.return_value = mock_context

        handler = AutoGenProfilerHandler()

        # Mock original function
        original_func = AsyncMock(return_value="tool result")
        handler._original_tool_call = original_func  # pylint: disable=protected-access

        # Get wrapped function
        wrapped_func = handler._tool_call_monkey_patch()  # pylint: disable=protected-access

        # Mock args
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_call_data = Mock()
        mock_call_data.kwargs = {"param": "value"}

        # Call wrapped function
        result = await wrapped_func(mock_tool, mock_call_data)

        # Verify result
        assert result == "tool result"
        original_func.assert_called_once_with(mock_tool, mock_call_data)
        assert mock_step_manager.push_intermediate_step.call_count == 2

    @pytest.mark.asyncio
    @patch('nat.plugins.autogen.autogen_callback_handler.Context.get')
    @patch('nat.plugins.autogen.autogen_callback_handler.logger')
    async def test_tool_wrapped_call_with_exception(self, mock_logger: Mock, mock_get: Mock):
        """Test tool wrapped call with exception handling.

        Args:
            mock_logger (Mock): Mock for the logger.
            mock_get (Mock): Mock for Context.get method.
        """
        # Setup mocks
        mock_context = Mock()
        mock_step_manager = Mock()
        mock_context.intermediate_step_manager = mock_step_manager
        mock_get.return_value = mock_context

        handler = AutoGenProfilerHandler()

        # Mock original function that raises exception
        original_func = AsyncMock(side_effect=Exception("Tool error"))
        handler._original_tool_call = original_func  # pylint: disable=protected-access

        # Get wrapped function
        wrapped_func = handler._tool_call_monkey_patch()  # pylint: disable=protected-access

        # Mock args
        mock_tool = Mock()
        mock_tool.name = "failing_tool"
        mock_call_data = {"kwargs": {"param": "value"}}

        # Call wrapped function
        with pytest.raises(Exception) as exc_info:
            await wrapped_func(mock_tool, mock_call_data)

        # Verify error handling
        assert "Tool error" in str(exc_info.value)
        mock_logger.error.assert_called()

    @pytest.mark.asyncio
    @patch('nat.plugins.autogen.autogen_callback_handler.Context.get')
    async def test_tool_wrapped_call_various_input_formats(self, mock_get: Mock):
        """Test tool wrapped call with various input formats."""
        # Setup mocks
        mock_context = Mock()
        mock_step_manager = Mock()
        mock_context.intermediate_step_manager = mock_step_manager
        mock_get.return_value = mock_context

        handler = AutoGenProfilerHandler()

        # Mock original function
        original_func = AsyncMock(return_value="result")
        handler._original_tool_call = original_func  # pylint: disable=protected-access

        # Get wrapped function
        wrapped_func = handler._tool_call_monkey_patch()  # pylint: disable=protected-access

        # Test with dict input format
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_call_data = {"kwargs": {"param": "value"}}

        result = await wrapped_func(mock_tool, mock_call_data)
        assert result == "result"


class TestErrorHandlingPaths:
    """Test error handling in various code paths."""

    @pytest.mark.asyncio
    @patch('nat.plugins.autogen.autogen_callback_handler.Context.get')
    @patch('nat.plugins.autogen.autogen_callback_handler.logger')
    async def test_llm_model_name_error_handling(self, mock_logger: Mock, mock_get: Mock):
        """Test error handling when getting model name fails."""
        # Setup mocks
        mock_context = Mock()
        mock_step_manager = Mock()
        mock_context.intermediate_step_manager = mock_step_manager
        mock_get.return_value = mock_context

        handler = AutoGenProfilerHandler()

        # Mock original function
        mock_output = Mock()
        mock_output.content = ["output"]
        mock_output.choices = []  # Empty choices to avoid processing
        # Create proper usage mock that has model_dump method
        mock_usage = Mock()
        mock_usage.model_dump.return_value = {
            'completion_tokens': 100,
            'prompt_tokens': 50,
            'total_tokens': 150,
            'completion_tokens_details': None,
            'prompt_tokens_details': None
        }
        mock_output.usage = mock_usage
        original_func = AsyncMock(return_value=mock_output)
        handler._original_llm_call = original_func  # pylint: disable=protected-access

        # Get wrapped function
        wrapped_func = handler._llm_call_monkey_patch()  # pylint: disable=protected-access

        # Mock args that will cause error in model name retrieval
        mock_args = [Mock()]
        # Make _raw_config access raise exception
        type(mock_args[0])._raw_config = PropertyMock(side_effect=Exception("Config error"))
        mock_args[0].model = "fallback"

        kwargs = {"messages": []}

        # Call wrapped function
        await wrapped_func(*mock_args, **kwargs)

        # Verify exception was logged
        mock_logger.exception.assert_called()

    @pytest.mark.asyncio
    @patch('nat.plugins.autogen.autogen_callback_handler.Context.get')
    @patch('nat.plugins.autogen.autogen_callback_handler.logger')
    async def test_llm_input_processing_error(self, mock_logger: Mock, mock_get: Mock):
        """Test error handling in input processing."""
        # Setup mocks
        mock_context = Mock()
        mock_step_manager = Mock()
        mock_context.intermediate_step_manager = mock_step_manager
        mock_get.return_value = mock_context

        handler = AutoGenProfilerHandler()

        # Mock original function
        original_func = AsyncMock(return_value=Mock(content=["output"]))
        handler._original_llm_call = original_func  # pylint: disable=protected-access

        # Get wrapped function
        wrapped_func = handler._tool_call_monkey_patch()  # pylint: disable=protected-access

        # Mock args that will cause error in input processing
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        # Create problematic input that will cause exception
        problematic_input = Mock()
        type(problematic_input).kwargs = PropertyMock(side_effect=Exception("Input error"))

        # Call wrapped function and expect exception
        with pytest.raises(Exception) as exc_info:
            await wrapped_func(mock_tool, problematic_input)

        # Verify error was logged and exception message is correct
        mock_logger.error.assert_called()
        assert "Input error" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
