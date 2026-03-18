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

import typing

import pytest

if typing.TYPE_CHECKING:
    from nat.data_models.evaluator import EvalInput
    from nat.plugins.eval.utils.intermediate_step_adapter import IntermediateStepAdapter


@pytest.fixture(name="rag_expected_outputs")
def rag_expected_outputs_fixture() -> list[str]:
    """Fixture providing expected outputs corresponding to user inputs."""
    return ["Machine Learning", "Natural Language Processing"]


@pytest.fixture(name="intermediate_step_adapter")
def intermediate_step_adapter_fixture() -> "IntermediateStepAdapter":
    from nat.plugins.eval.utils.intermediate_step_adapter import IntermediateStepAdapter
    return IntermediateStepAdapter()


@pytest.fixture(name="rag_eval_input")
def rag_eval_input_fixture(
    rag_user_inputs,
    rag_expected_outputs,
    rag_generated_outputs,
    rag_intermediate_steps,
) -> "EvalInput":
    """Build EvalInput items used by RAGAS evaluator tests."""
    from nat.data_models.evaluator import EvalInput
    from nat.data_models.evaluator import EvalInputItem

    eval_items = [
        EvalInputItem(
            id=index + 1,
            input_obj=user_input,
            expected_output_obj=expected_output,
            output_obj=generated_output,
            expected_trajectory=[],
            trajectory=rag_intermediate_steps[index],
            full_dataset_entry={
                "id": index + 1,
                "question": user_input,
                "answer": expected_output,
                "generated_answer": generated_output,
            },
        ) for index, (user_input, expected_output,
                      generated_output) in enumerate(zip(rag_user_inputs, rag_expected_outputs, rag_generated_outputs))
    ]

    return EvalInput(eval_input_items=eval_items)
