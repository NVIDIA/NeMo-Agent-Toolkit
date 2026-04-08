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

from types import SimpleNamespace
from unittest.mock import MagicMock

from nat.data_models.intermediate_step import IntermediateStepType
from nat.plugins.crewai.crewai_callback_handler import CrewAIProfilerHandler


class _Message:
    """Mimics a proper Message attribute (crewai >= 1.1.0)."""

    def __init__(self, content, role="assistant"):
        self.content = content
        self.role = role


class _NewStyleChoice:
    """Choice with message as a proper attribute, empty model_extra (crewai >= 1.1.0)."""

    def __init__(self, content):
        self.message = _Message(content)
        self.model_extra = {}

    def model_dump(self):
        return {"message": {"content": self.message.content, "role": self.message.role}}


class _OldStyleChoice:
    """Choice with message only in model_extra (older crewai)."""

    def __init__(self, content):
        self.model_extra = {"message": {"content": content, "role": "assistant"}}

    def model_dump(self):
        return {"message": self.model_extra["message"]}


class _TokenUsage:

    def __init__(self):
        self.prompt_tokens = 10
        self.completion_tokens = 5
        self.total_tokens = 15

    def model_dump(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }


def _make_output(choices):
    """Build a minimal mock LLM output with the given choices."""
    return SimpleNamespace(
        choices=choices,
        model="test-model",
        usage=_TokenUsage(),
        model_extra={"usage": _TokenUsage()},
    )


def _run_wrapped_call(choices):
    """Run the monkey-patched LLM call with the given choices, return captured payloads."""
    handler = CrewAIProfilerHandler()
    handler._original_llm_call = lambda *a, **kw: _make_output(choices)
    handler.step_manager = MagicMock()

    wrapped = handler._llm_call_monkey_patch()
    wrapped(model="test", messages=[{"content": "prompt"}])

    payloads = [call.args[0] for call in handler.step_manager.push_intermediate_step.call_args_list]
    llm_end = [p for p in payloads if p.event_type == IntermediateStepType.LLM_END]
    assert len(llm_end) == 1
    return llm_end[0]


def test_new_style_choice_message_attribute():
    """choice.message.content is used when message is a proper attribute (crewai >= 1.1.0)."""
    end_payload = _run_wrapped_call([_NewStyleChoice("hello from new API")])
    assert end_payload.data.output == "hello from new API"


def test_old_style_choice_model_extra():
    """choice.model_extra['message'] is used when message lives in model_extra (older crewai)."""
    end_payload = _run_wrapped_call([_OldStyleChoice("hello from old API")])
    assert end_payload.data.output == "hello from old API"


def test_multiple_choices_mixed_styles():
    """Multiple choices with different styles are all extracted."""
    end_payload = _run_wrapped_call([_NewStyleChoice("first"), _OldStyleChoice("second")])
    assert end_payload.data.output == "firstsecond"


def test_choice_with_none_content():
    """None content is handled gracefully without raising."""
    end_payload = _run_wrapped_call([_NewStyleChoice(None)])
    assert end_payload.data.output == ""
