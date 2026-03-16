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
"""
Tests for model output extraction in CrewAIProfilerHandler._llm_call_monkey_patch.

Verifies that choice.message.content (crewai >= 1.1.0) and
choice.model_extra["message"] (older versions) are both handled correctly.
"""

import time
from types import SimpleNamespace
from unittest.mock import patch

from nat.plugins.crewai.crewai_callback_handler import CrewAIProfilerHandler
from nat.utils.reactive.subject import Subject


class _Message:
    """Mimics a proper Message attribute (crewai >= 1.1.0)."""

    def __init__(self, content, role="assistant"):
        self.content = content
        self.role = role


class _NewStyleChoice:
    """Choice with message as a proper attribute, empty model_extra."""

    def __init__(self, content):
        self.message = _Message(content)
        self.model_extra = {}


class _OldStyleChoice:
    """Choice with message only in model_extra (older crewai)."""

    def __init__(self, content):
        self.model_extra = {"message": {"content": content, "role": "assistant"}}


class _TokenUsage:

    def __init__(self):
        self.prompt_tokens = 10
        self.completion_tokens = 5
        self.total_tokens = 15


def _make_output(choices):
    """Build a minimal mock LLM output with the given choices."""
    return SimpleNamespace(
        choices=choices,
        model="test-model",
        usage=_TokenUsage(),
    )


def test_new_style_choice_message_attribute(reactive_stream: Subject):
    """choice.message.content is used when message is a proper attribute (crewai >= 1.1.0)."""
    handler = CrewAIProfilerHandler()
    handler._original_llm_call = lambda *a, **kw: _make_output([_NewStyleChoice("hello from new API")])

    with patch.object(time, "time", side_effect=[100.0, 103.0]):
        wrapped = handler._llm_call_monkey_patch()
        result = wrapped(model="test", messages=[{"content": "prompt"}])

    assert result.choices[0].message.content == "hello from new API"


def test_old_style_choice_model_extra(reactive_stream: Subject):
    """choice.model_extra['message'] is used when message lives in model_extra (older crewai)."""
    handler = CrewAIProfilerHandler()
    handler._original_llm_call = lambda *a, **kw: _make_output([_OldStyleChoice("hello from old API")])

    with patch.object(time, "time", side_effect=[100.0, 103.0]):
        wrapped = handler._llm_call_monkey_patch()
        result = wrapped(model="test", messages=[{"content": "prompt"}])

    assert result.choices[0].model_extra["message"]["content"] == "hello from old API"


def test_multiple_choices_mixed_styles(reactive_stream: Subject):
    """Multiple choices with different styles are all extracted."""
    choices = [_NewStyleChoice("first"), _OldStyleChoice("second")]
    handler = CrewAIProfilerHandler()
    handler._original_llm_call = lambda *a, **kw: _make_output(choices)

    results = []
    _ = reactive_stream.subscribe(results.append)

    with patch.object(time, "time", side_effect=[100.0, 103.0]):
        wrapped = handler._llm_call_monkey_patch()
        wrapped(model="test", messages=[{"content": "prompt"}])

    # The LLM_END event should contain the concatenated output
    llm_end_events = [r for r in results if r.event_type.value == "llm_end"]
    assert len(llm_end_events) == 1
    assert llm_end_events[0].data.output == "firstsecond"


def test_choice_with_none_content(reactive_stream: Subject):
    """None content is handled gracefully without raising."""
    handler = CrewAIProfilerHandler()
    handler._original_llm_call = lambda *a, **kw: _make_output([_NewStyleChoice(None)])

    results = []
    _ = reactive_stream.subscribe(results.append)

    with patch.object(time, "time", side_effect=[100.0, 103.0]):
        wrapped = handler._llm_call_monkey_patch()
        wrapped(model="test", messages=[{"content": "prompt"}])

    llm_end_events = [r for r in results if r.event_type.value == "llm_end"]
    assert len(llm_end_events) == 1
    assert llm_end_events[0].data.output == ""
