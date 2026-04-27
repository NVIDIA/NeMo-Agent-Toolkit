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
"""Test AG2 Callback Handler."""

import concurrent.futures
import threading
import time
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from nat.plugins.ag2.callback_handler import AG2ProfilerHandler


class TestAG2ProfilerHandlerInit:
    """Test AG2ProfilerHandler initialization."""

    def test_init_creates_lock(self):
        """Test handler creates a threading lock."""
        handler = AG2ProfilerHandler()
        assert isinstance(handler._lock, type(threading.Lock()))

    def test_init_sets_timestamp(self):
        """Test handler initializes last_call_ts."""
        handler = AG2ProfilerHandler()
        assert isinstance(handler.last_call_ts, float)
        assert handler.last_call_ts > 0

    def test_init_not_instrumented(self):
        """Test handler starts not instrumented."""
        handler = AG2ProfilerHandler()
        assert handler._instrumented is False

    def test_init_no_original_create(self):
        """Test handler starts with no original_create."""
        handler = AG2ProfilerHandler()
        assert handler._original_create is None

    @patch('nat.plugins.ag2.callback_handler.Context.get')
    def test_init_gets_step_manager(self, mock_get):
        """Test handler gets step_manager from context."""
        mock_context = Mock()
        mock_step_manager = Mock()
        mock_context.intermediate_step_manager = mock_step_manager
        mock_get.return_value = mock_context

        handler = AG2ProfilerHandler()
        assert handler.step_manager is mock_step_manager


class TestInstrument:
    """Test instrument() method."""

    def test_instrument_skips_if_already_instrumented(self):
        """Test instrument() skips if already instrumented."""
        handler = AG2ProfilerHandler()
        handler._instrumented = True

        with patch('nat.plugins.ag2.callback_handler.logger') as mock_logger:
            handler.instrument()
            mock_logger.debug.assert_any_call("AG2ProfilerHandler already instrumented; skipping.")

    def test_instrument_handles_missing_import(self):
        """Test instrument() handles missing autogen.oai.client."""
        handler = AG2ProfilerHandler()

        with patch('nat.plugins.ag2.callback_handler.logger') as mock_logger:
            with patch.dict('sys.modules', {'autogen': None, 'autogen.oai': None, 'autogen.oai.client': None}):
                with patch('builtins.__import__', side_effect=ImportError("No module")):
                    handler.instrument()
                    mock_logger.debug.assert_any_call("autogen.oai.client not available; skipping AG2 instrumentation")

        assert handler._instrumented is False

    def test_instrument_patches_create(self):
        """Test instrument() patches OpenAIWrapper.create."""
        handler = AG2ProfilerHandler()

        mock_openai_wrapper = Mock()
        original_create = Mock()
        mock_openai_wrapper.create = original_create

        with patch('nat.plugins.ag2.callback_handler.logger'):
            with patch.dict(
                    'sys.modules',
                {
                    'autogen': Mock(),
                    'autogen.oai': Mock(),
                    'autogen.oai.client': Mock(OpenAIWrapper=mock_openai_wrapper),
                },
            ):
                handler.instrument()
                assert handler._instrumented is True
                assert handler._original_create is original_create
                # create should now be the wrapper, not the original
                assert mock_openai_wrapper.create is not original_create

                handler.uninstrument()

    def test_instrument_sets_instrumented_flag(self):
        """Test instrument() sets _instrumented to True."""
        handler = AG2ProfilerHandler()

        with patch('nat.plugins.ag2.callback_handler.logger'):
            handler.instrument()
            assert handler._instrumented is True
            handler.uninstrument()

    def test_legacy_patch_alias(self):
        """Test that patch() is an alias for instrument()."""
        handler = AG2ProfilerHandler()
        assert handler.patch == handler.instrument

    def test_legacy_unpatch_alias(self):
        """Test that unpatch() is an alias for uninstrument()."""
        handler = AG2ProfilerHandler()
        assert handler.unpatch == handler.uninstrument


class TestUninstrument:
    """Test uninstrument() method."""

    def test_uninstrument_resets_state(self):
        """Test uninstrument() resets handler state."""
        handler = AG2ProfilerHandler()
        handler._instrumented = True

        mock_openai_wrapper = Mock()
        original_create = Mock()
        handler._original_create = original_create

        with patch('nat.plugins.ag2.callback_handler.logger'):
            with patch.dict(
                    'sys.modules',
                {
                    'autogen': Mock(),
                    'autogen.oai': Mock(),
                    'autogen.oai.client': Mock(OpenAIWrapper=mock_openai_wrapper),
                },
            ):
                handler.uninstrument()

        assert handler._instrumented is False
        assert handler._original_create is None
        assert mock_openai_wrapper.create is original_create

    def test_uninstrument_handles_import_errors(self):
        """Test uninstrument() handles import errors gracefully."""
        handler = AG2ProfilerHandler()
        handler._instrumented = True
        handler._original_create = Mock()

        with patch('nat.plugins.ag2.callback_handler.logger') as mock_logger:
            with patch('builtins.__import__', side_effect=ImportError("No module")):
                handler.uninstrument()
                mock_logger.exception.assert_called_with("Failed to uninstrument AG2ProfilerHandler")

    def test_uninstrument_noop_when_not_instrumented(self):
        """Test uninstrument() is a no-op when original_create is None."""
        handler = AG2ProfilerHandler()
        handler._original_create = None

        with patch('nat.plugins.ag2.callback_handler.logger'):
            handler.uninstrument()

        assert handler._instrumented is False


class TestHelperMethods:
    """Test helper extraction methods."""

    def test_extract_model_name(self):
        """Test _extract_model_name extracts model from kwargs."""
        handler = AG2ProfilerHandler()
        result = handler._extract_model_name({"model": "gpt-4-turbo"})
        assert result == "gpt-4-turbo"

    def test_extract_model_name_missing(self):
        """Test _extract_model_name returns empty string when missing."""
        handler = AG2ProfilerHandler()
        result = handler._extract_model_name({})
        assert result == ""

    def test_extract_input_text(self):
        """Test _extract_input_text returns sanitized input string."""
        handler = AG2ProfilerHandler()
        result = handler._extract_input_text({
            "model": "gpt-4",
            "messages": [{
                "role": "user", "content": "hello"
            }],
        })
        assert "gpt-4" in result
        assert "1" in result  # message_count

    def test_extract_input_text_empty_messages(self):
        """Test _extract_input_text with no messages."""
        handler = AG2ProfilerHandler()
        result = handler._extract_input_text({"model": "test"})
        assert "0" in result  # message_count = 0


class TestLLMWrapper:
    """Test _create_llm_wrapper functionality."""

    @patch('nat.plugins.ag2.callback_handler.Context.get')
    def test_llm_wrapper_pushes_start_and_end_events(self, mock_get):
        """Test LLM wrapper pushes START and END events."""
        mock_context = Mock()
        mock_step_manager = Mock()
        mock_context.intermediate_step_manager = mock_step_manager
        mock_get.return_value = mock_context

        handler = AG2ProfilerHandler()

        original_func = Mock(return_value="llm_response")
        wrapped = handler._create_llm_wrapper(original_func)

        wrapper_self = Mock()
        wrapped(wrapper_self, model="test-model", messages=[{"content": "Hello"}])

        # Verify both events pushed
        assert mock_step_manager.push_intermediate_step.call_count == 2

        calls = mock_step_manager.push_intermediate_step.call_args_list
        assert calls[0][0][0].event_type.value == "LLM_START"
        assert calls[1][0][0].event_type.value == "LLM_END"

    @patch('nat.plugins.ag2.callback_handler.Context.get')
    def test_llm_wrapper_returns_original_result(self, mock_get):
        """Test LLM wrapper returns the original function's result."""
        mock_context = Mock()
        mock_context.intermediate_step_manager = Mock()
        mock_get.return_value = mock_context

        handler = AG2ProfilerHandler()

        original_func = Mock(return_value="expected_result")
        wrapped = handler._create_llm_wrapper(original_func)

        result = wrapped(Mock(), model="test")
        assert result == "expected_result"

    @patch('nat.plugins.ag2.callback_handler.Context.get')
    def test_llm_wrapper_handles_exception(self, mock_get):
        """Test LLM wrapper handles exceptions correctly."""
        mock_context = Mock()
        mock_step_manager = Mock()
        mock_context.intermediate_step_manager = mock_step_manager
        mock_get.return_value = mock_context

        handler = AG2ProfilerHandler()

        original_func = Mock(side_effect=ValueError("LLM Error"))
        wrapped = handler._create_llm_wrapper(original_func)

        with pytest.raises(ValueError, match="LLM Error"):
            wrapped(Mock(), model="test-model", messages=[])

        # Should have START and error END
        assert mock_step_manager.push_intermediate_step.call_count == 2
        error_call = mock_step_manager.push_intermediate_step.call_args_list[1][0][0]
        assert "error: ValueError" in error_call.data.output

    @patch('nat.plugins.ag2.callback_handler.Context.get')
    def test_llm_wrapper_uuid_consistency(self, mock_get):
        """Test that START and END events share the same UUID."""
        mock_context = Mock()
        mock_step_manager = Mock()
        mock_context.intermediate_step_manager = mock_step_manager
        mock_get.return_value = mock_context

        handler = AG2ProfilerHandler()

        original_func = Mock(return_value="response")
        wrapped = handler._create_llm_wrapper(original_func)

        wrapped(Mock(), model="test")

        calls = mock_step_manager.push_intermediate_step.call_args_list
        start_uuid = calls[0][0][0].UUID
        end_uuid = calls[1][0][0].UUID
        assert start_uuid == end_uuid

    @patch('nat.plugins.ag2.callback_handler.Context.get')
    def test_llm_wrapper_tracks_seconds_between_calls(self, mock_get):
        """Test that wrapper tracks time between calls."""
        mock_context = Mock()
        mock_step_manager = Mock()
        mock_context.intermediate_step_manager = mock_step_manager
        mock_get.return_value = mock_context

        handler = AG2ProfilerHandler()
        handler.last_call_ts = time.time() - 5  # 5 seconds ago

        original_func = Mock(return_value="response")
        wrapped = handler._create_llm_wrapper(original_func)

        wrapped(Mock(), model="test")

        start_call = mock_step_manager.push_intermediate_step.call_args_list[0][0][0]
        assert start_call.usage_info.seconds_between_calls >= 5

    @patch('nat.plugins.ag2.callback_handler.Context.get')
    def test_llm_wrapper_updates_last_call_ts(self, mock_get):
        """Test that wrapper updates last_call_ts after success."""
        mock_context = Mock()
        mock_context.intermediate_step_manager = Mock()
        mock_get.return_value = mock_context

        handler = AG2ProfilerHandler()
        old_ts = handler.last_call_ts

        original_func = Mock(return_value="response")
        wrapped = handler._create_llm_wrapper(original_func)

        time.sleep(0.01)
        wrapped(Mock(), model="test")

        assert handler.last_call_ts > old_ts

    @patch('nat.plugins.ag2.callback_handler.Context.get')
    def test_llm_wrapper_uses_ag2_framework(self, mock_get):
        """Test that events use AG2 framework identifier."""
        mock_context = Mock()
        mock_step_manager = Mock()
        mock_context.intermediate_step_manager = mock_step_manager
        mock_get.return_value = mock_context

        handler = AG2ProfilerHandler()

        original_func = Mock(return_value="response")
        wrapped = handler._create_llm_wrapper(original_func)

        wrapped(Mock(), model="test")

        calls = mock_step_manager.push_intermediate_step.call_args_list
        assert calls[0][0][0].framework.value == "ag2"
        assert calls[1][0][0].framework.value == "ag2"


class TestIntegration:
    """Integration tests for full workflow."""

    def test_full_instrument_uninstrument_cycle(self):
        """Test complete instrument/uninstrument cycle."""
        handler = AG2ProfilerHandler()

        assert not handler._instrumented

        handler.instrument()
        assert handler._instrumented

        handler.uninstrument()
        assert not handler._instrumented

    def test_lock_thread_safety(self):
        """Test that lock prevents concurrent timestamp updates."""
        handler = AG2ProfilerHandler()

        def update_timestamp():
            with handler._lock:
                time.sleep(0.01)
                handler.last_call_ts = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(update_timestamp) for _ in range(10)]
            concurrent.futures.wait(futures)

        assert handler.last_call_ts > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
