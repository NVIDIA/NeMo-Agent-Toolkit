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

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.builder.context import Context
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.intermediate_step import IntermediateStepType
from nat.plugins.strands.strands_callback_handler import StrandsProfilerHandler
from nat.plugins.strands.strands_callback_handler import StrandsToolInstrumentationHook


class TestStrandsToolInstrumentationHook:
    """Tests for StrandsToolInstrumentationHook."""

    @pytest.fixture
    def mock_handler(self):
        """Create a mock StrandsProfilerHandler."""
        return MagicMock(spec=StrandsProfilerHandler)

    @pytest.fixture
    def mock_step_manager(self):
        """Create a mock intermediate step manager."""
        manager = MagicMock()
        manager.push_intermediate_step = MagicMock()
        return manager

    @pytest.fixture
    def tool_hook(self, mock_handler, mock_step_manager):
        """Create a StrandsToolInstrumentationHook instance."""
        with patch.object(Context, "get", return_value=MagicMock(intermediate_step_manager=mock_step_manager)):
            hook = StrandsToolInstrumentationHook(mock_handler)
            return hook

    def test_hook_initialization(self, mock_handler):
        """Test that hook initializes correctly."""
        with patch.object(
                Context,
                "get",
                return_value=MagicMock(intermediate_step_manager=MagicMock()),
        ):
            hook = StrandsToolInstrumentationHook(mock_handler)
            assert hook.handler == mock_handler
            # pylint: disable=protected-access
            assert isinstance(hook._tool_start_times, dict)
            assert hook._step_manager is not None

    def test_on_before_tool_invocation_emits_start_span(self, tool_hook, mock_step_manager):
        """Test that before hook emits TOOL_START span."""
        # Create mock event
        mock_event = MagicMock()
        mock_event.tool_use = {
            "toolUseId": "test-id-123",
            "name": "test_tool",
            "input": {
                "param": "value"
            },
        }
        mock_event.selected_tool = MagicMock()
        mock_event.selected_tool.tool_name = "test_tool"
        mock_event.selected_tool.tool_spec = {"name": "test_tool"}

        # Call the hook
        tool_hook.on_before_tool_invocation(mock_event)

        # Verify TOOL_START span was pushed
        mock_step_manager.push_intermediate_step.assert_called_once()
        call_args = mock_step_manager.push_intermediate_step.call_args[0][0]
        assert call_args.event_type == IntermediateStepType.TOOL_START
        assert call_args.name == "test_tool"
        assert call_args.UUID == "test-id-123"

    def test_on_before_tool_invocation_instruments_nat_wrapped_tools(self, tool_hook, mock_step_manager):
        """Test that NAT-wrapped tools are properly instrumented."""
        from nat.plugins.strands.tool_wrapper import NATFunctionAgentTool

        # Create mock event with NAT-wrapped tool
        mock_event = MagicMock()
        mock_event.tool_use = {"toolUseId": "nat-tool-id-123", "name": "nat_tool", "input": {"param": "value"}}
        mock_event.selected_tool = MagicMock(spec=NATFunctionAgentTool)
        mock_event.selected_tool.tool_name = "nat_tool"
        mock_event.selected_tool.tool_spec = {"name": "nat_tool"}

        # Call the hook
        tool_hook.on_before_tool_invocation(mock_event)

        # Verify TOOL_START span was pushed (NAT tools are now instrumented)
        mock_step_manager.push_intermediate_step.assert_called_once()
        call_args = mock_step_manager.push_intermediate_step.call_args[0][0]
        assert call_args.event_type == IntermediateStepType.TOOL_START
        assert call_args.name == "nat_tool"
        assert call_args.UUID == "nat-tool-id-123"

    def test_on_after_tool_invocation_emits_end_span(self, tool_hook, mock_step_manager):
        """Test that after hook emits TOOL_END span."""
        # First emit a start span to populate start times
        tool_use_id = "test-id-456"
        # pylint: disable=protected-access
        tool_hook._tool_start_times[tool_use_id] = 1234567890.0

        # Create mock event
        mock_event = MagicMock()
        mock_event.tool_use = {
            "toolUseId": tool_use_id,
            "name": "test_tool",
            "input": {
                "param": "value"
            },
        }
        mock_event.selected_tool = MagicMock()
        mock_event.selected_tool.tool_name = "test_tool"
        mock_event.result = {"content": [{"text": "tool output"}]}
        mock_event.exception = None

        # Call the hook
        tool_hook.on_after_tool_invocation(mock_event)

        # Verify TOOL_END span was pushed
        assert mock_step_manager.push_intermediate_step.call_count > 0
        call_args = mock_step_manager.push_intermediate_step.call_args[0][0]
        assert call_args.event_type == IntermediateStepType.TOOL_END
        assert call_args.name == "test_tool"
        assert call_args.UUID == tool_use_id


class TestStrandsProfilerHandler:
    """Tests for StrandsProfilerHandler."""

    def test_handler_initialization(self):
        """Test that handler initializes correctly."""
        handler = StrandsProfilerHandler()
        # pylint: disable=protected-access
        assert handler._patched is False
        assert handler._hooks_registered is False
        assert handler.tool_hook is not None
        assert isinstance(handler.tool_hook, StrandsToolInstrumentationHook)

    @patch("nat.plugins.strands.strands_callback_handler.importlib")
    def test_instrument_patches_llm_methods(self, mock_importlib):
        """Test that instrument patches LLM streaming methods."""
        # Create mock OpenAI model with __name__ attribute
        mock_openai_model = type("OpenAIModel", (), {"stream": MagicMock(), "structured_output": MagicMock()})

        mock_openai_mod = MagicMock()
        mock_openai_mod.OpenAIModel = mock_openai_model

        # Create mock Bedrock model with __name__ attribute
        mock_bedrock_model = type("BedrockModel", (), {"stream": MagicMock(), "structured_output": MagicMock()})

        mock_bedrock_mod = MagicMock()
        mock_bedrock_mod.BedrockModel = mock_bedrock_model

        def import_side_effect(module_name):
            if "openai" in module_name:
                return mock_openai_mod
            elif "bedrock" in module_name:
                return mock_bedrock_mod
            raise ImportError(f"No module named {module_name}")

        mock_importlib.import_module.side_effect = import_side_effect

        handler = StrandsProfilerHandler()
        handler.instrument()

        # Verify patching occurred
        assert handler._patched is True  # pylint: disable=protected-access

    def test_instrument_only_runs_once(self):
        """Test that instrument only patches once."""
        handler = StrandsProfilerHandler()
        handler._patched = True  # pylint: disable=protected-access

        # Should return early without patching
        with patch("nat.plugins.strands.strands_callback_handler.importlib"):
            handler.instrument()

        # Still patched
        assert handler._patched is True  # pylint: disable=protected-access

    def test_extract_model_info_extracts_name(self):
        """Test model info extraction."""
        handler = StrandsProfilerHandler()

        mock_model = MagicMock()
        mock_model.config = {"model": "test-model-name"}
        mock_model.params = {
            "temperature": 0.7,
            "max_tokens": 1000,
        }

        # pylint: disable=protected-access
        model_name, model_params = handler._extract_model_info(mock_model)

        assert model_name == "test-model-name"
        assert "temperature" in model_params
        assert model_params["temperature"] == 0.7

    def test_extract_model_info_handles_missing_attrs(self):
        """Test model info extraction with missing attributes."""
        handler = StrandsProfilerHandler()

        mock_model = MagicMock(spec=[])  # No attributes

        # pylint: disable=protected-access
        model_name, model_params = handler._extract_model_info(mock_model)

        assert model_name == ""
        assert isinstance(model_params, dict)


class TestStrandsProfilerHandlerIntegration:
    """Integration tests for profiler handler."""

    @patch("nat.plugins.strands.strands_callback_handler.importlib")
    def test_full_instrumentation_flow(self, mock_importlib):  # pylint: disable=too-many-locals
        """Test complete instrumentation flow."""
        # Mock the models
        mock_openai_model = type(
            "OpenAIModel",
            (),
            {
                "stream": MagicMock(), "__name__": "OpenAIModel"
            },
        )
        mock_openai_mod = MagicMock()
        mock_openai_mod.OpenAIModel = mock_openai_model

        # Mock Agent class - use a real class to allow __init__ patching
        class MockAgent:

            def __init__(self, *args, **kwargs):
                self.hooks = MagicMock()

        mock_agent_mod = MagicMock()
        mock_agent_mod.Agent = MockAgent

        def import_side_effect(module_name):
            if "openai" in module_name:
                return mock_openai_mod
            elif "bedrock" in module_name:
                raise ImportError("Bedrock not available")
            elif "agent.agent" in module_name:
                return mock_agent_mod
            raise ImportError(f"No module named {module_name}")

        mock_importlib.import_module.side_effect = import_side_effect

        handler = StrandsProfilerHandler()
        handler.instrument()

        # Verify handler is fully instrumented
        assert handler._patched is True  # pylint: disable=protected-access
        assert hasattr(mock_openai_model, "stream")


class TestStrandsProfilerHandlerEventExtraction:
    """Tests for event extraction methods in StrandsProfilerHandler."""

    @pytest.fixture
    def handler(self):
        """Create a StrandsProfilerHandler instance."""
        return StrandsProfilerHandler()

    def test_extract_text_from_event_with_data(self, handler):
        """Test _extract_text_from_event with data field."""
        event = {"data": "Hello world"}

        # pylint: disable=protected-access
        result = handler._extract_text_from_event(event)

        assert result == "Hello world"

    def test_extract_text_from_event_without_data(self, handler):
        """Test _extract_text_from_event without data field."""
        event = {"other_field": "value"}

        # pylint: disable=protected-access
        result = handler._extract_text_from_event(event)

        assert result == ""

    def test_extract_text_from_event_non_dict(self, handler):
        """Test _extract_text_from_event with non-dict input."""
        event = "not a dict"

        # pylint: disable=protected-access
        result = handler._extract_text_from_event(event)

        assert result == ""

    def test_extract_text_from_event_none_data(self, handler):
        """Test _extract_text_from_event with None data."""
        event = {"data": None}

        # pylint: disable=protected-access
        result = handler._extract_text_from_event(event)

        assert result == "None"

    def test_extract_usage_from_event_valid(self, handler):
        """Test _extract_usage_from_event with valid usage data."""
        event = {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30}}}

        # pylint: disable=protected-access
        result = handler._extract_usage_from_event(event)

        expected = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        assert result == expected

    def test_extract_usage_from_event_missing_metadata(self, handler):
        """Test _extract_usage_from_event without metadata."""
        event = {"other_field": "value"}

        # pylint: disable=protected-access
        result = handler._extract_usage_from_event(event)

        assert result is None

    def test_extract_usage_from_event_missing_usage(self, handler):
        """Test _extract_usage_from_event without usage in metadata."""
        event = {"metadata": {"other_field": "value"}}

        # pylint: disable=protected-access
        result = handler._extract_usage_from_event(event)

        assert result is None

    def test_extract_usage_from_event_non_dict_metadata(self, handler):
        """Test _extract_usage_from_event with non-dict metadata."""
        event = {"metadata": "not a dict"}

        # pylint: disable=protected-access
        result = handler._extract_usage_from_event(event)

        assert result is None

    def test_extract_usage_from_event_non_dict_usage(self, handler):
        """Test _extract_usage_from_event with non-dict usage."""
        event = {"metadata": {"usage": "not a dict"}}

        # pylint: disable=protected-access
        result = handler._extract_usage_from_event(event)

        assert result is None

    def test_extract_usage_from_event_invalid_values(self, handler):
        """Test _extract_usage_from_event with invalid token values."""
        event = {"metadata": {"usage": {"inputTokens": "invalid", "outputTokens": None, "totalTokens": 30}}}

        # pylint: disable=protected-access
        result = handler._extract_usage_from_event(event)

        assert result is None

    def test_extract_usage_from_event_partial_data(self, handler):
        """Test _extract_usage_from_event with partial token data."""
        event = {
            "metadata": {
                "usage": {
                    "inputTokens": 15,  # Missing outputTokens and totalTokens
                }
            }
        }

        # pylint: disable=protected-access
        result = handler._extract_usage_from_event(event)

        expected = {"prompt_tokens": 15, "completion_tokens": 0, "total_tokens": 0}
        assert result == expected

    def test_extract_usage_from_event_non_dict_input(self, handler):
        """Test _extract_usage_from_event with non-dict input."""
        event = "not a dict"

        # pylint: disable=protected-access
        result = handler._extract_usage_from_event(event)

        assert result is None


class TestStrandsProfilerHandlerStreamWrapper:
    """Tests for _wrap_stream_method functionality."""

    @pytest.fixture
    def handler(self):
        """Create a StrandsProfilerHandler instance."""
        return StrandsProfilerHandler()

    @pytest.fixture
    def mock_context(self):
        """Create a mock context with step manager."""
        with patch('nat.plugins.strands.strands_callback_handler.Context') as mock_context_class:
            mock_context_instance = MagicMock()
            mock_step_manager = MagicMock()
            mock_context_instance.intermediate_step_manager = mock_step_manager
            mock_context_class.get.return_value = mock_context_instance
            yield mock_step_manager

    @pytest.mark.asyncio
    async def test_wrap_stream_method_basic_flow(self, handler, mock_context):
        """Test basic streaming flow with _wrap_stream_method."""

        # Create a mock original streaming method
        async def mock_original(model_self, messages, tool_specs=None, system_prompt=None, **kwargs):
            # Simulate streaming events
            yield {"data": "Hello"}
            yield {"data": " world"}
            yield {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}}}

        # Create mock model instance
        mock_model = MagicMock()
        mock_model.config = {"model": "test-model"}
        mock_model.params = {"temperature": 0.7}

        # Get wrapped method
        # pylint: disable=protected-access
        wrapped_method = handler._wrap_stream_method(mock_original)

        # Test messages
        messages = [{"role": "user", "content": "Hello"}]
        system_prompt = "You are helpful"

        # Call wrapped method
        output_chunks = []
        async for chunk in wrapped_method(mock_model, messages, None, system_prompt):
            output_chunks.append(chunk)

        # Verify we got the expected chunks
        assert len(output_chunks) == 3
        assert output_chunks[0]["data"] == "Hello"
        assert output_chunks[1]["data"] == " world"

        # Verify intermediate steps were pushed (START and END)
        assert mock_context.push_intermediate_step.call_count == 2

        # Verify START event
        start_call = mock_context.push_intermediate_step.call_args_list[0][0][0]
        assert start_call.event_type == IntermediateStepType.LLM_START
        assert start_call.framework == LLMFrameworkEnum.STRANDS
        assert start_call.name == "test-model"

        # Verify END event
        end_call = mock_context.push_intermediate_step.call_args_list[1][0][0]
        assert end_call.event_type == IntermediateStepType.LLM_END
        assert end_call.framework == LLMFrameworkEnum.STRANDS
        assert end_call.data.output == "Hello world"

    @pytest.mark.asyncio
    async def test_wrap_stream_method_with_exception(self, handler, mock_context):
        """Test _wrap_stream_method handles exceptions properly."""

        # Create a mock original method that raises an exception
        async def mock_original_with_error(model_self, *args, **kwargs):
            yield {"data": "Start"}
            raise RuntimeError("Stream error")

        mock_model = MagicMock()
        mock_model.config = {"model": "test-model"}
        mock_model.params = {}

        # pylint: disable=protected-access
        wrapped_method = handler._wrap_stream_method(mock_original_with_error)

        messages = [{"role": "user", "content": "Test"}]

        # Should still handle the exception gracefully
        output_chunks = []
        with pytest.raises(RuntimeError, match="Stream error"):
            async for chunk in wrapped_method(mock_model, messages):
                output_chunks.append(chunk)

        # Should have gotten the first chunk before the error
        assert len(output_chunks) == 1
        assert output_chunks[0]["data"] == "Start"

        # Should still push START and END events (END in finally block)
        assert mock_context.push_intermediate_step.call_count == 2

    @pytest.mark.asyncio
    async def test_wrap_stream_method_non_async_generator(self, handler, mock_context):
        """Test _wrap_stream_method with non-async generator response."""

        # Create a mock original method that returns a coroutine instead of async generator
        async def mock_original_coroutine(model_self, *args, **kwargs):
            return {"result": "single response"}

        mock_model = MagicMock()
        mock_model.config = {"model": "test-model"}
        mock_model.params = {}

        # pylint: disable=protected-access
        wrapped_method = handler._wrap_stream_method(mock_original_coroutine)

        messages = [{"role": "user", "content": "Test"}]

        # Should handle non-streaming response
        output_chunks = []
        async for chunk in wrapped_method(mock_model, messages):
            output_chunks.append(chunk)

        # Should get the single response
        assert len(output_chunks) == 1
        assert output_chunks[0]["result"] == "single response"

        # Should still push START and END events
        assert mock_context.push_intermediate_step.call_count == 2

    @pytest.mark.asyncio
    async def test_wrap_stream_method_message_handling(self, handler, mock_context):
        """Test _wrap_stream_method properly handles different message formats."""

        async def mock_original(model_self, messages, tool_specs=None, system_prompt=None, **kwargs):
            yield {"data": "response"}

        mock_model = MagicMock()
        mock_model.config = {"model": "test-model"}
        mock_model.params = {}

        # pylint: disable=protected-access
        wrapped_method = handler._wrap_stream_method(mock_original)

        # Test with complex messages and system prompt
        messages = [{
            "role": "user", "content": "Hello"
        }, {
            "role": "assistant", "content": "Hi there"
        }, {
            "role": "user", "content": "How are you?"
        }]
        system_prompt = "You are a helpful assistant"

        output_chunks = []
        async for chunk in wrapped_method(mock_model, messages, None, system_prompt):
            output_chunks.append(chunk)

        # Verify START event includes all messages
        start_call = mock_context.push_intermediate_step.call_args_list[0][0][0]
        chat_inputs = start_call.data.input

        # Should include system prompt and all messages
        assert len(chat_inputs) == 4  # system + 3 user messages
        assert chat_inputs[0]["role"] == "system"
        assert chat_inputs[0]["text"] == system_prompt


class TestStrandsProfilerHandlerAgentInstrumentation:
    """Tests for _instrument_agent_init method."""

    @pytest.fixture
    def handler(self):
        """Create a StrandsProfilerHandler instance."""
        return StrandsProfilerHandler()

    @patch("nat.plugins.strands.strands_callback_handler.importlib")
    def test_instrument_agent_init_success(self, mock_importlib, handler):
        """Test successful agent instrumentation."""

        # Create a mock Agent class
        class MockAgent:

            def __init__(self, *args, **kwargs):
                self.hooks = MagicMock()

        mock_agent_mod = MagicMock()
        mock_agent_mod.Agent = MockAgent

        def import_side_effect(module_name):
            if "agent.agent" in module_name:
                return mock_agent_mod
            elif "experimental.hooks" in module_name:
                # Import the actual hook classes for testing
                try:
                    from strands.experimental.hooks import AfterToolInvocationEvent
                    from strands.experimental.hooks import BeforeToolInvocationEvent
                    hook_mod = MagicMock()
                    hook_mod.BeforeToolInvocationEvent = BeforeToolInvocationEvent
                    hook_mod.AfterToolInvocationEvent = AfterToolInvocationEvent
                    return hook_mod
                except ImportError:
                    # Fallback to mocks if strands not available
                    hook_mod = MagicMock()
                    hook_mod.BeforeToolInvocationEvent = MagicMock()
                    hook_mod.AfterToolInvocationEvent = MagicMock()
                    return hook_mod
            raise ImportError(f"No module named {module_name}")

        mock_importlib.import_module.side_effect = import_side_effect

        # Call the method
        # pylint: disable=protected-access
        handler._instrument_agent_init()

        # Create an agent instance to test the wrapped __init__
        agent = MockAgent()

        # Verify hooks were registered
        assert agent.hooks.add_callback.call_count == 2

        # Verify callbacks were registered (don't check exact class equality)
        calls = agent.hooks.add_callback.call_args_list
        assert len(calls) == 2
        # Check that the callbacks are the correct functions
        assert calls[0][0][1] == handler.tool_hook.on_before_tool_invocation
        assert calls[1][0][1] == handler.tool_hook.on_after_tool_invocation

    @patch("nat.plugins.strands.strands_callback_handler.importlib")
    def test_instrument_agent_init_agent_not_found(self, mock_importlib, handler):
        """Test agent instrumentation when Agent class not found."""
        mock_agent_mod = MagicMock()
        mock_agent_mod.Agent = None  # Agent not found

        mock_importlib.import_module.return_value = mock_agent_mod

        # Should handle gracefully when Agent is None
        # pylint: disable=protected-access
        handler._instrument_agent_init()

        # Should not raise an exception
        assert True

    @patch("nat.plugins.strands.strands_callback_handler.importlib")
    def test_instrument_agent_init_import_error(self, mock_importlib, handler):
        """Test agent instrumentation with import error."""
        mock_importlib.import_module.side_effect = ImportError("Module not found")

        # Should handle import errors gracefully
        # pylint: disable=protected-access
        handler._instrument_agent_init()

        # Should not raise an exception
        assert True

    @patch("nat.plugins.strands.strands_callback_handler.importlib")
    def test_instrument_agent_init_hook_registration_error(self, mock_importlib, handler):
        """Test agent instrumentation with hook registration error."""

        # Create a mock Agent class
        class MockAgent:

            def __init__(self, *args, **kwargs):
                self.hooks = MagicMock()
                # Make add_callback raise an error
                self.hooks.add_callback.side_effect = Exception("Hook registration failed")

        mock_agent_mod = MagicMock()
        mock_agent_mod.Agent = MockAgent

        def import_side_effect(module_name):
            if "agent.agent" in module_name:
                return mock_agent_mod
            elif "experimental.hooks" in module_name:
                hook_mod = MagicMock()
                hook_mod.BeforeToolInvocationEvent = MagicMock()
                hook_mod.AfterToolInvocationEvent = MagicMock()
                return hook_mod
            raise ImportError(f"No module named {module_name}")

        mock_importlib.import_module.side_effect = import_side_effect

        # pylint: disable=protected-access
        handler._instrument_agent_init()

        # Create an agent instance - should handle hook registration errors gracefully
        agent = MockAgent()

        # Should have attempted to register hooks despite the error
        assert agent.hooks.add_callback.called

        # Should have attempted to register hooks despite the error
        assert agent.hooks.add_callback.called
