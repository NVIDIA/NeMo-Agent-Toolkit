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
"""Test AutoGen Callback Handler."""

import threading
import time
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from nat.plugins.autogen.callback_handler import AutoGenProfilerHandler
from nat.plugins.autogen.callback_handler import ClientPatchInfo
from nat.plugins.autogen.callback_handler import PatchedClients


class TestDataClasses:
    """Test the dataclass structures."""

    def test_client_patch_info_defaults(self):
        """Test ClientPatchInfo has correct defaults."""
        info = ClientPatchInfo()
        assert info.create is None
        assert info.create_stream is None

    def test_client_patch_info_with_values(self):
        """Test ClientPatchInfo stores values."""
        mock_create = Mock()
        mock_stream = Mock()
        info = ClientPatchInfo(create=mock_create, create_stream=mock_stream)
        assert info.create is mock_create
        assert info.create_stream is mock_stream

    def test_patched_clients_defaults(self):
        """Test PatchedClients has correct defaults."""
        patched = PatchedClients()
        assert isinstance(patched.openai, ClientPatchInfo)
        assert isinstance(patched.azure, ClientPatchInfo)
        assert isinstance(patched.bedrock, ClientPatchInfo)
        assert patched.tool is None


class TestAutoGenProfilerHandlerInit:
    """Test AutoGenProfilerHandler initialization."""

    def test_init_creates_lock(self):
        """Test handler creates a threading lock."""
        handler = AutoGenProfilerHandler()
        assert isinstance(handler._lock, type(threading.Lock()))

    def test_init_sets_timestamp(self):
        """Test handler initializes last_call_ts."""
        handler = AutoGenProfilerHandler()
        assert isinstance(handler.last_call_ts, float)
        assert handler.last_call_ts > 0

    def test_init_creates_patched_clients(self):
        """Test handler creates PatchedClients structure."""
        handler = AutoGenProfilerHandler()
        assert isinstance(handler._patched, PatchedClients)

    def test_init_not_instrumented(self):
        """Test handler starts not instrumented."""
        handler = AutoGenProfilerHandler()
        assert handler._instrumented is False

    @patch('nat.plugins.autogen.callback_handler.Context.get')
    def test_init_gets_step_manager(self, mock_get):
        """Test handler gets step_manager from context."""
        mock_context = Mock()
        mock_step_manager = Mock()
        mock_context.intermediate_step_manager = mock_step_manager
        mock_get.return_value = mock_context

        handler = AutoGenProfilerHandler()
        assert handler.step_manager is mock_step_manager


class TestInstrument:
    """Test instrument() method."""

    def test_instrument_skips_if_already_instrumented(self):
        """Test instrument() skips if already instrumented."""
        handler = AutoGenProfilerHandler()
        handler._instrumented = True

        with patch('nat.plugins.autogen.callback_handler.logger') as mock_logger:
            handler.instrument()
            mock_logger.debug.assert_any_call("AutoGenProfilerHandler already instrumented; skipping.")

    @patch('nat.plugins.autogen.callback_handler.logger')
    def test_instrument_handles_missing_tool_import(self, mock_logger):
        """Test instrument() handles missing autogen_core.tools."""
        handler = AutoGenProfilerHandler()

        with patch.dict('sys.modules', {'autogen_core': None, 'autogen_core.tools': None}):
            with patch('builtins.__import__', side_effect=ImportError("No module")):
                handler.instrument()
                # Should still complete (gracefully handle missing imports)
                mock_logger.debug.assert_any_call("autogen_core.tools not available; skipping tool instrumentation")
        # Always uninstrument to clean up any partial patches that may have succeeded
        # (e.g., if autogen_ext modules were already in sys.modules)
        handler.uninstrument()

    def test_instrument_patches_openai_client(self):
        """Test instrument() patches OpenAIChatCompletionClient."""
        handler = AutoGenProfilerHandler()

        mock_openai_client = Mock()
        mock_openai_client.create = Mock()
        mock_openai_client.create_stream = Mock()

        with patch.object(handler, '_create_llm_wrapper', return_value=Mock()) as mock_wrapper:
            with patch.object(handler, '_create_stream_wrapper', return_value=Mock()) as mock_stream_wrapper:
                with patch('nat.plugins.autogen.callback_handler.logger'):
                    # Mock the import
                    with patch.dict(
                            'sys.modules',
                        {
                            'autogen_core':
                                Mock(),
                            'autogen_core.tools':
                                Mock(BaseTool=Mock(run_json=Mock())),
                            'autogen_ext':
                                Mock(),
                            'autogen_ext.models':
                                Mock(),
                            'autogen_ext.models.openai':
                                Mock(OpenAIChatCompletionClient=mock_openai_client,
                                     AzureOpenAIChatCompletionClient=Mock()),
                            'autogen_ext.models.anthropic':
                                Mock(AnthropicBedrockChatCompletionClient=Mock())
                        }):
                        handler.instrument()

                        # Verify wrappers were created
                        assert handler._instrumented is True
                        mock_wrapper.assert_called()
                        mock_stream_wrapper.assert_called()

                        # Uninstrument within the mocked context to properly restore mocked classes
                        handler.uninstrument()

    def test_instrument_sets_instrumented_flag(self):
        """Test instrument() sets _instrumented to True."""
        handler = AutoGenProfilerHandler()

        with patch('nat.plugins.autogen.callback_handler.logger'):
            # Note: When AutoGen is installed, imports will succeed and patch real classes.
            # Always uninstrument to restore original class methods.
            handler.instrument()
            assert handler._instrumented is True
            handler.uninstrument()


class TestUninstrument:
    """Test uninstrument() method."""

    def test_uninstrument_resets_state(self):
        """Test uninstrument() resets handler state.

        This test verifies uninstrument() properly resets internal state.
        We must use mocked modules to avoid polluting the real OpenAIChatCompletionClient.
        """
        handler = AutoGenProfilerHandler()
        handler._instrumented = True
        handler._patched.tool = Mock()
        handler._patched.openai.create = Mock()

        # Mock the imports so uninstrument() operates on mocks, not the real classes
        mock_openai_client = Mock()
        mock_azure_client = Mock()
        mock_bedrock_client = Mock()
        mock_base_tool = Mock()

        with patch('nat.plugins.autogen.callback_handler.logger'):
            with patch.dict(
                    'sys.modules',
                {
                    'autogen_core.tools':
                        Mock(BaseTool=mock_base_tool),
                    'autogen_ext.models.openai':
                        Mock(OpenAIChatCompletionClient=mock_openai_client,
                             AzureOpenAIChatCompletionClient=mock_azure_client),
                    'autogen_ext.models.anthropic':
                        Mock(AnthropicBedrockChatCompletionClient=mock_bedrock_client),
                }):
                handler.uninstrument()

        assert handler._instrumented is False
        assert handler._patched.tool is None
        assert handler._patched.openai.create is None

    def test_uninstrument_handles_import_errors(self):
        """Test uninstrument() handles import errors gracefully."""
        handler = AutoGenProfilerHandler()
        handler._instrumented = True
        handler._patched.openai.create = Mock()

        with patch('nat.plugins.autogen.callback_handler.logger') as mock_logger:
            with patch('builtins.__import__', side_effect=ImportError("No module")):
                handler.uninstrument()
                mock_logger.exception.assert_called_with("Failed to uninstrument AutoGenProfilerHandler")


class TestHelperMethods:
    """Test helper extraction methods."""

    def test_extract_model_name_from_raw_config(self):
        """Test _extract_model_name extracts from _raw_config."""
        handler = AutoGenProfilerHandler()
        client = Mock()
        client._raw_config = {"model": "gpt-4-turbo"}

        result = handler._extract_model_name(client)
        assert result == "gpt-4-turbo"

    def test_extract_model_name_fallback_to_model_attr(self):
        """Test _extract_model_name falls back to model attribute."""
        handler = AutoGenProfilerHandler()
        client = Mock()
        client._raw_config = {}
        client.model = "fallback-model"

        result = handler._extract_model_name(client)
        assert result == "fallback-model"

    def test_extract_model_name_returns_unknown(self):
        """Test _extract_model_name returns 'unknown_model' on failure."""
        handler = AutoGenProfilerHandler()
        client = Mock(spec=[])  # No attributes

        result = handler._extract_model_name(client)
        assert result == "unknown_model"

    def test_extract_input_text_simple_content(self):
        """Test _extract_input_text with simple string content."""
        handler = AutoGenProfilerHandler()
        messages = [{"content": "Hello"}, {"content": "World"}]

        result = handler._extract_input_text(messages)
        assert result == "HelloWorld"

    def test_extract_input_text_list_content(self):
        """Test _extract_input_text with list content."""
        handler = AutoGenProfilerHandler()
        messages = [{"content": ["Part 1", {"text": "Part 2"}, "Part 3"]}]

        result = handler._extract_input_text(messages)
        assert "Part 1" in result
        assert "Part 2" in result
        assert "Part 3" in result

    def test_extract_input_text_handles_none(self):
        """Test _extract_input_text handles None content."""
        handler = AutoGenProfilerHandler()
        messages = [{"content": None}]

        result = handler._extract_input_text(messages)
        assert result == ""

    def test_extract_output_text(self):
        """Test _extract_output_text extracts from response."""
        handler = AutoGenProfilerHandler()
        output = Mock()
        output.content = ["Hello ", "World"]

        result = handler._extract_output_text(output)
        assert result == "Hello World"

    def test_extract_output_text_handles_error(self):
        """Test _extract_output_text returns empty on error."""
        handler = AutoGenProfilerHandler()
        output = Mock(spec=[])  # No content attribute

        result = handler._extract_output_text(output)
        assert result == ""

    def test_extract_usage_with_model_dump(self):
        """Test _extract_usage with model_dump method."""
        handler = AutoGenProfilerHandler()
        output = Mock()
        output.usage = Mock()
        output.usage.model_dump.return_value = {"total_tokens": 100}

        result = handler._extract_usage(output)
        assert result == {"total_tokens": 100}

    def test_extract_usage_with_dict(self):
        """Test _extract_usage with dict usage."""
        handler = AutoGenProfilerHandler()
        output = Mock()
        output.usage = {"prompt_tokens": 50, "completion_tokens": 50}

        result = handler._extract_usage(output)
        assert result["prompt_tokens"] == 50

    def test_extract_usage_from_model_extra(self):
        """Test _extract_usage falls back to model_extra."""
        handler = AutoGenProfilerHandler()
        output = Mock()
        output.usage = None
        output.model_extra = {"usage": {"total_tokens": 75}}

        result = handler._extract_usage(output)
        assert result == {"total_tokens": 75}

    def test_extract_chat_response(self):
        """Test _extract_chat_response extracts first choice."""
        handler = AutoGenProfilerHandler()
        output = Mock()
        output.choices = [Mock()]
        output.choices[0].model_dump.return_value = {"role": "assistant", "content": "Hi"}

        result = handler._extract_chat_response(output)
        assert result == {"role": "assistant", "content": "Hi"}

    def test_extract_chat_response_empty_choices(self):
        """Test _extract_chat_response handles empty choices."""
        handler = AutoGenProfilerHandler()
        output = Mock()
        output.choices = []

        result = handler._extract_chat_response(output)
        assert result == {}


class TestLLMWrapper:
    """Test _create_llm_wrapper functionality."""

    @patch('nat.plugins.autogen.callback_handler.Context.get')
    async def test_llm_wrapper_pushes_start_and_end_events(self, mock_get):
        """Test LLM wrapper pushes START and END events."""
        mock_context = Mock()
        mock_step_manager = Mock()
        mock_context.intermediate_step_manager = mock_step_manager
        mock_get.return_value = mock_context

        handler = AutoGenProfilerHandler()

        # Create mock response
        mock_output = Mock()
        mock_output.content = ["Test response"]
        mock_output.usage = None
        mock_output.choices = []
        mock_output.model_extra = {}

        original_func = AsyncMock(return_value=mock_output)
        wrapped = handler._create_llm_wrapper(original_func)

        # Call the wrapper
        client = Mock()
        client._raw_config = {"model": "test-model"}
        await wrapped(client, messages=[{"content": "Hello"}])

        # Verify both events pushed
        assert mock_step_manager.push_intermediate_step.call_count == 2

        # Verify event types (enum values are uppercase)
        calls = mock_step_manager.push_intermediate_step.call_args_list
        assert calls[0][0][0].event_type.value == "LLM_START"
        assert calls[1][0][0].event_type.value == "LLM_END"

    @patch('nat.plugins.autogen.callback_handler.Context.get')
    async def test_llm_wrapper_handles_exception(self, mock_get):
        """Test LLM wrapper handles exceptions correctly."""
        mock_context = Mock()
        mock_step_manager = Mock()
        mock_context.intermediate_step_manager = mock_step_manager
        mock_get.return_value = mock_context

        handler = AutoGenProfilerHandler()

        original_func = AsyncMock(side_effect=ValueError("LLM Error"))
        wrapped = handler._create_llm_wrapper(original_func)

        client = Mock()
        client._raw_config = {"model": "test-model"}

        with pytest.raises(ValueError, match="LLM Error"):
            await wrapped(client, messages=[])

        # Should have START and error END
        assert mock_step_manager.push_intermediate_step.call_count == 2
        error_call = mock_step_manager.push_intermediate_step.call_args_list[1][0][0]
        assert "LLM Error" in error_call.data.output

    @patch('nat.plugins.autogen.callback_handler.Context.get')
    async def test_llm_wrapper_extracts_usage(self, mock_get):
        """Test LLM wrapper extracts token usage."""
        mock_context = Mock()
        mock_step_manager = Mock()
        mock_context.intermediate_step_manager = mock_step_manager
        mock_get.return_value = mock_context

        handler = AutoGenProfilerHandler()

        mock_output = Mock()
        mock_output.content = ["Response"]
        mock_output.choices = []
        mock_output.usage = Mock()
        mock_output.usage.model_dump.return_value = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

        original_func = AsyncMock(return_value=mock_output)
        wrapped = handler._create_llm_wrapper(original_func)

        client = Mock()
        client._raw_config = {"model": "test-model"}
        await wrapped(client, messages=[])

        # Check the END event has usage
        end_call = mock_step_manager.push_intermediate_step.call_args_list[1][0][0]
        assert end_call.usage_info.token_usage.prompt_tokens == 10
        assert end_call.usage_info.token_usage.completion_tokens == 20


class TestStreamWrapper:
    """Test _create_stream_wrapper functionality."""

    @patch('nat.plugins.autogen.callback_handler.Context.get')
    async def test_stream_wrapper_yields_chunks(self, mock_get):
        """Test stream wrapper yields all chunks."""
        mock_context = Mock()
        mock_step_manager = Mock()
        mock_context.intermediate_step_manager = mock_step_manager
        mock_get.return_value = mock_context

        handler = AutoGenProfilerHandler()

        # Create async generator that yields chunks
        async def mock_stream(*args, **kwargs):
            yield Mock(content="chunk1", usage=None)
            yield Mock(content="chunk2", usage=None)
            yield Mock(content="chunk3", usage={"total_tokens": 30})

        wrapped = handler._create_stream_wrapper(mock_stream)

        client = Mock()
        client._raw_config = {"model": "test-model"}

        chunks = []
        async for chunk in wrapped(client, messages=[]):
            chunks.append(chunk)

        assert len(chunks) == 3

    @patch('nat.plugins.autogen.callback_handler.Context.get')
    async def test_stream_wrapper_pushes_events(self, mock_get):
        """Test stream wrapper pushes START and END events."""
        mock_context = Mock()
        mock_step_manager = Mock()
        mock_context.intermediate_step_manager = mock_step_manager
        mock_get.return_value = mock_context

        handler = AutoGenProfilerHandler()

        async def mock_stream(*args, **kwargs):
            yield Mock(content="test", usage=None)

        wrapped = handler._create_stream_wrapper(mock_stream)

        client = Mock()
        client._raw_config = {"model": "test-model"}

        async for _ in wrapped(client, messages=[]):
            pass

        # Should have START and END
        assert mock_step_manager.push_intermediate_step.call_count == 2

    @patch('nat.plugins.autogen.callback_handler.Context.get')
    async def test_stream_wrapper_handles_error(self, mock_get):
        """Test stream wrapper handles errors during streaming."""
        mock_context = Mock()
        mock_step_manager = Mock()
        mock_context.intermediate_step_manager = mock_step_manager
        mock_get.return_value = mock_context

        handler = AutoGenProfilerHandler()

        async def mock_stream(*args, **kwargs):
            yield Mock(content="test", usage=None)
            raise RuntimeError("Stream error")

        wrapped = handler._create_stream_wrapper(mock_stream)

        client = Mock()
        client._raw_config = {"model": "test-model"}

        with pytest.raises(RuntimeError, match="Stream error"):
            async for _ in wrapped(client, messages=[]):
                pass

        # Should have START and error END
        assert mock_step_manager.push_intermediate_step.call_count == 2


class TestToolWrapper:
    """Test _create_tool_wrapper functionality."""

    @patch('nat.plugins.autogen.callback_handler.Context.get')
    async def test_tool_wrapper_basic_flow(self, mock_get):
        """Test tool wrapper pushes START and END events."""
        mock_context = Mock()
        mock_step_manager = Mock()
        mock_context.intermediate_step_manager = mock_step_manager
        mock_get.return_value = mock_context

        handler = AutoGenProfilerHandler()

        original_func = AsyncMock(return_value="tool result")
        wrapped = handler._create_tool_wrapper(original_func)

        tool = Mock()
        tool.name = "test_tool"
        call_data = Mock()
        call_data.kwargs = {"param": "value"}

        result = await wrapped(tool, call_data)

        assert result == "tool result"
        assert mock_step_manager.push_intermediate_step.call_count == 2

        # Verify event types (enum values are uppercase)
        calls = mock_step_manager.push_intermediate_step.call_args_list
        assert calls[0][0][0].event_type.value == "TOOL_START"
        assert calls[1][0][0].event_type.value == "TOOL_END"

    @patch('nat.plugins.autogen.callback_handler.Context.get')
    async def test_tool_wrapper_handles_dict_input(self, mock_get):
        """Test tool wrapper handles dict input format."""
        mock_context = Mock()
        mock_step_manager = Mock()
        mock_context.intermediate_step_manager = mock_step_manager
        mock_get.return_value = mock_context

        handler = AutoGenProfilerHandler()

        original_func = AsyncMock(return_value="result")
        wrapped = handler._create_tool_wrapper(original_func)

        tool = Mock()
        tool.name = "test_tool"
        call_data = {"kwargs": {"key": "value"}}

        result = await wrapped(tool, call_data)
        assert result == "result"

    @patch('nat.plugins.autogen.callback_handler.Context.get')
    async def test_tool_wrapper_handles_exception(self, mock_get):
        """Test tool wrapper handles tool execution errors."""
        mock_context = Mock()
        mock_step_manager = Mock()
        mock_context.intermediate_step_manager = mock_step_manager
        mock_get.return_value = mock_context

        handler = AutoGenProfilerHandler()

        original_func = AsyncMock(side_effect=ValueError("Tool failed"))
        wrapped = handler._create_tool_wrapper(original_func)

        tool = Mock()
        tool.name = "failing_tool"
        call_data = Mock()
        call_data.kwargs = {}

        with pytest.raises(ValueError, match="Tool failed"):
            await wrapped(tool, call_data)

        # Should have START and error END
        assert mock_step_manager.push_intermediate_step.call_count == 2
        error_call = mock_step_manager.push_intermediate_step.call_args_list[1][0][0]
        assert "Tool failed" in error_call.data.output


class TestIntegration:
    """Integration tests for full workflow."""

    async def test_full_instrument_uninstrument_cycle(self):
        """Test complete instrument/uninstrument cycle."""
        handler = AutoGenProfilerHandler()

        # Should start not instrumented
        assert not handler._instrumented

        # Instrument (will handle missing imports gracefully)
        handler.instrument()
        assert handler._instrumented

        # Uninstrument
        handler.uninstrument()
        assert not handler._instrumented

    def test_lock_thread_safety(self):
        """Test that lock prevents concurrent timestamp updates."""
        handler = AutoGenProfilerHandler()

        def update_timestamp():
            with handler._lock:
                time.sleep(0.01)
                handler.last_call_ts = time.time()

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(update_timestamp) for _ in range(10)]
            concurrent.futures.wait(futures)

        # Should complete without errors
        assert handler.last_call_ts > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
