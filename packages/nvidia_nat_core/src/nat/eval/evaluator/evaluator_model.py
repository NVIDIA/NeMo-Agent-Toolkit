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

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import SerializeAsAny

from nat.data_models.intermediate_step import IntermediateStep


class EvalInputItem(BaseModel):
    """A single input item for evaluation."""

    id: typing.Any = Field(description="Unique identifier for this evaluation item.")
    input_obj: typing.Any = Field(description="The input to the workflow (e.g., user question).")
    expected_output_obj: typing.Any = Field(description="The expected/ground truth output.")
    output_obj: typing.Any = Field(default=None, description="The actual workflow output. Populated during evaluation.")
    expected_trajectory: list[IntermediateStep] = Field(
        default_factory=list,
        description="Expected intermediate steps for trajectory evaluation.",
    )
    trajectory: list[IntermediateStep] = Field(
        default_factory=list,
        description="Actual intermediate steps from workflow execution. Populated during evaluation.",
    )
    full_dataset_entry: typing.Any = Field(description="The complete original dataset entry.")

    def copy_with_updates(self, **updates) -> "EvalInputItem":
        """Copy EvalInputItem with optional field updates."""
        item_data = self.model_dump()
        item_data.update(updates)
        return EvalInputItem(**item_data)


class EvalInput(BaseModel):
    """Container for evaluation input items."""

    eval_input_items: list[EvalInputItem] = Field(description="List of items to evaluate.")


class EvalOutputItem(BaseModel):
    """A single output item from evaluation."""

    model_config = ConfigDict(exclude_none=True)

    id: typing.Any = Field(description="Identifier matching the corresponding EvalInputItem.")
    score: typing.Any = Field(description="Evaluation score (typically float, may be NaN on failure).")
    reasoning: typing.Any = Field(description="Evaluation context and LLM judge explanation.")
    error: str | None = Field(default=None, description="Evaluation error message if this item failed.")


class EvalOutput(BaseModel):
    """Container for evaluation output items."""

    average_score: typing.Any = Field(description="Average score across all evaluated items.")
    eval_output_items: list[SerializeAsAny[EvalOutputItem]] = Field(description="List of evaluation results.")
