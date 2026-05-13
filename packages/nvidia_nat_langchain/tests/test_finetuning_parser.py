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

from unittest.mock import MagicMock

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.intermediate_step import IntermediateStepState
from nat.data_models.intermediate_step import IntermediateStepType
from nat.finetuning.utils.parsers.base_parser import parse_to_openai_messages
from nat.test.observability import create_mock_step


def test_skip_non_relevant_event_types():
    """Test that non-LLM/TOOL events are skipped."""
    step = create_mock_step(IntermediateStepType.WORKFLOW_START,
                            IntermediateStepState.START,
                            framework=LLMFrameworkEnum.LANGCHAIN)

    result = parse_to_openai_messages([step])
    assert len(result) == 0


def test_skip_streaming_chunks():
    """Test that streaming chunks are skipped."""
    step = create_mock_step(
        IntermediateStepType.LLM_END,
        IntermediateStepState.CHUNK,  # Should be skipped
        framework=LLMFrameworkEnum.LANGCHAIN)

    result = parse_to_openai_messages([step])
    assert len(result) == 0


def test_skip_llm_start_after_tool_end():
    """Test that LLM_START after TOOL_END is skipped."""
    steps = [
        create_mock_step(IntermediateStepType.TOOL_END, IntermediateStepState.END,
                         framework=LLMFrameworkEnum.LANGCHAIN),
        create_mock_step(
            IntermediateStepType.LLM_START,  # Should be skipped
            IntermediateStepState.START,
            framework=LLMFrameworkEnum.LANGCHAIN),
    ]

    # Mock the data for tool_end
    steps[0].data = MagicMock()
    steps[0].data.output = "tool result"

    result = parse_to_openai_messages(steps)
    # Should only have tool message, no LLM_START
    assert len(result) == 1
