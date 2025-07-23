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

import pytest

from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.data_models.intermediate_step import IntermediateStep
from aiq.data_models.intermediate_step import IntermediateStepPayload
from aiq.data_models.intermediate_step import IntermediateStepType
from aiq.data_models.intermediate_step import StreamEventData
from aiq.data_models.invocation_node import InvocationNode
from aiq.eval.intermediate_step_adapter import IntermediateStepAdapter

# pylint: disable=redefined-outer-name


@pytest.fixture
def llm_name():
    return "mock_llm"


@pytest.fixture
def tool_name():
    return "mock_tool"


@pytest.fixture
def mock_intermediate_steps(llm_name, tool_name):
    """
    Fixture to generate a list of IntermediateStep objects with -
    1. LLM_START, LLM_NEW_TOKENs, LLM_END
    2. TOOL_START, and TOOL_END.
    """

    framework = LLMFrameworkEnum.LANGCHAIN
    token_cnt = 10
    user_input = "Question: What is AIQ Toolkit?"
    tool_input = "Tool query input"
    tool_output = "Tool output response"
    generated_output = "Final AI-generated response"

    def create_step(event_type, name=llm_name, input_data=None, output_data=None, chunk=None):
        """Helper to create an `IntermediateStep`."""
        return IntermediateStep(parent_id="root",
                                function_ancestry=InvocationNode(function_name=name, function_id="test-function-id"),
                                payload=IntermediateStepPayload(event_type=event_type,
                                                                framework=framework,
                                                                name=name,
                                                                data=StreamEventData(input=input_data,
                                                                                     output=output_data,
                                                                                     chunk=chunk)))

    return [
        create_step(IntermediateStepType.LLM_START, input_data=user_input),
        *[create_step(IntermediateStepType.LLM_NEW_TOKEN, chunk=f"Token {i}") for i in range(token_cnt)],
        create_step(IntermediateStepType.LLM_END, input_data=user_input, output_data=generated_output),
        create_step(IntermediateStepType.TOOL_START, name=tool_name, input_data=tool_input),
        create_step(IntermediateStepType.TOOL_END, name=tool_name, input_data=tool_input, output_data=tool_output),
    ]


@pytest.fixture
def intermediate_step_adapter():
    return IntermediateStepAdapter()


@pytest.fixture
def filter_events(intermediate_step_adapter):
    return {IntermediateStepType.LLM_END, IntermediateStepType.TOOL_END}


def test_filter_intermediate_steps(intermediate_step_adapter, mock_intermediate_steps, filter_events):
    """Test that filter_intermediate_steps only returns LLM_END and TOOL_END steps."""

    # Call actual method
    filtered_steps = intermediate_step_adapter.filter_intermediate_steps(mock_intermediate_steps,
                                                                         intermediate_step_adapter.DEFAULT_EVENT_FILTER)

    assert len(filtered_steps) == len(filter_events), f"Expected {len(filter_events)} steps, got {len(filtered_steps)}"
    assert all(step.event_type in filter_events for step in filtered_steps), "Only LLM_END & TOOL_END should remain"


def test_get_agent_actions(intermediate_step_adapter, mock_intermediate_steps, filter_events, llm_name, tool_name):
    """
    Test that get_agent_actions returns the correct number of steps and the correct action and output.
    Only tool_end is present in the adapted steps
    """

    # Call actual method
    adapted_steps = intermediate_step_adapter.get_agent_actions(mock_intermediate_steps,
                                                                intermediate_step_adapter.DEFAULT_EVENT_FILTER)

    assert adapted_steps, "Adapted steps are empty"
    # Find tool and LLM steps by their names
    tool_step = next((step for step in adapted_steps if step[0].tool == tool_name), None)
    llm_step = next((step for step in adapted_steps if step[0].tool == llm_name), None)

    assert tool_step is not None, "Tool step not found"
    assert llm_step is not None, "LLM step not found"

    tool_action, tool_output = tool_step
    llm_action, llm_output = llm_step

    assert tool_output == "Tool output response", "Tool output mismatch"
    assert llm_output == "Final AI-generated response", "LLM output mismatch"
