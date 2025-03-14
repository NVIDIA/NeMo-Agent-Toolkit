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
from aiq.eval.evaluator.evaluator_model import EvalInput
from aiq.eval.evaluator.evaluator_model import EvalInputItem
from aiq.eval.intermediate_step_adapter import IntermediateStepAdapter

# pylint: disable=redefined-outer-name


@pytest.fixture
def rag_user_inputs() -> list[str]:
    """Fixture providing multiple user inputs."""
    return ["What is ML?", "What is NLP?"]


@pytest.fixture
def rag_expected_outputs() -> list[str]:
    """Fixture providing expected outputs corresponding to user inputs."""
    return ["Machine Learning", "Natural Language Processing"]


@pytest.fixture
def rag_generated_outputs() -> list[str]:
    """Fixture providing workflow generated outputs corresponding to user inputs."""
    return ["ML is the abbreviation for Machine Learning", "NLP stands for Natural Language Processing"]


@pytest.fixture
def intermediate_step_adapter() -> IntermediateStepAdapter:
    return IntermediateStepAdapter()


@pytest.fixture
def rag_intermediate_steps(rag_user_inputs,
                           rag_generated_outputs) -> tuple[list[IntermediateStep], list[IntermediateStep]]:
    """
    Fixture to generate separate lists of IntermediateStep objects for each user input.

    Each list includes:
    1. LLM_START, LLM_NEW_TOKENs, LLM_END
    2. TOOL_START, and TOOL_END.

    Returns:
        (list for user_input_1, list for user_input_2)
    """
    framework = LLMFrameworkEnum.LANGCHAIN
    token_cnt = 10
    llm_name = "mock_llm"
    tool_name = "mock_tool"

    def create_step(event_type, name=llm_name, input_data=None, output_data=None, chunk=None):
        """Helper to create an `IntermediateStep`."""
        return IntermediateStep(
            payload=IntermediateStepPayload(event_type=event_type,
                                            framework=framework,
                                            name=name,
                                            data=StreamEventData(input=input_data, output=output_data, chunk=chunk)))

    step_lists = []  # Store separate lists

    for user_input, generated_ouput in zip(rag_user_inputs, rag_generated_outputs):
        tool_input = f"Get me the documents for {user_input}"
        tool_output = f"Here is information I have on {user_input}"
        generated_output = generated_ouput

        steps = [
            create_step(IntermediateStepType.LLM_START, input_data=user_input),
            *[
                create_step(IntermediateStepType.LLM_NEW_TOKEN, chunk=f"Token {i} for {user_input}")
                for i in range(token_cnt)
            ],
            create_step(IntermediateStepType.LLM_END, input_data=user_input, output_data=generated_output),
            create_step(IntermediateStepType.TOOL_START, name=tool_name, input_data=tool_input),
            create_step(IntermediateStepType.TOOL_END, name=tool_name, input_data=tool_input, output_data=tool_output),
        ]

        step_lists.append(steps)  # Append separate list for each user input

    return tuple(step_lists)  # Return as two separate lists


@pytest.fixture
def rag_eval_input(rag_user_inputs, rag_expected_outputs, rag_generated_outputs, rag_intermediate_steps) -> EvalInput:
    """Fixture to create a mock EvalInput with multiple items."""

    # Unpack intermediate steps
    steps_1, steps_2 = rag_intermediate_steps
    intermediate_steps_map = [steps_1, steps_2]

    eval_items = [
        EvalInputItem(
            id=index + 1,  # Ensure unique IDs (1, 2, ...)
            input_obj=user_input,
            expected_output_obj=expected_output,
            output_obj=generated_output,
            expected_trajectory=[],  # Modify if needed
            trajectory=intermediate_steps_map[index]  # Ensure correct step assignment
        ) for index, (user_input, expected_output,
                      generated_output) in enumerate(zip(rag_user_inputs, rag_expected_outputs, rag_generated_outputs))
    ]

    return EvalInput(eval_input_items=eval_items)
