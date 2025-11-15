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

from __future__ import annotations

from pydantic import BaseModel
from pydantic import Field

from nat.data_models.intermediate_step import IntermediateStep
from nat.eval.evaluator.evaluator_model import EvalOutputItem


class ConditionEvaluationResult(BaseModel):
    """
    Evaluation results for a single IntermediateStep that meets the filtering conditions.

    Contains the filter condition used, the score, the intermediate steps that contributed
    to it based on the reduction strategy, and the evaluations for each step.
    """
    score: float = Field(description="Final score for this condition after reduction")
    reduction_strategy: str = Field(
        description=(
            "Reduction strategy used: mean, max, or last. "
            "This is used when more than one intermediate step meets the filtering conditions."
        )
    )
    intermediate_steps: list[IntermediateStep] = Field(
        description="Filtered IntermediateSteps that contributed to the score (based on reduction strategy)"
    )
    step_evaluations: list[EvalOutputItem] = Field(
        description="Score and reasoning for each evaluated step"
    )


class RedTeamingEvalOutputItem(EvalOutputItem):
    """
    Extended evaluation output item for red teaming evaluations.

    Organizes results by filter condition name, with each condition containing
    its score, the intermediate steps that contributed, and their evaluations.

    Attributes:
        id: Identifier from the input item
        score: Average score across all filter conditions
        reasoning: Summary information for compatibility
        condition_results: Map from condition name to evaluation results
    """
    results_by_condition: dict[str, ConditionEvaluationResult] = Field(
        description="Results organized by filter condition name"
    )
