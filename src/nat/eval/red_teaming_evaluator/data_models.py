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

from pydantic import Field

from nat.data_models.intermediate_step import IntermediateStep
from nat.eval.evaluator.evaluator_model import EvalOutputItem


class ConditionEvaluationResult(EvalOutputItem):
    """
    Evaluation results for a single IntermediateStep that meets the filtering condition.
    Attributes:
        id: Identifier from the input item
        score: Average score across all filter conditions
        reasoning: Reasoning for given score.
        intermediate_step: IntermediateStep that was selected and evaluated (based on reduction strategy).
            The evaluated output can be accessed through intermediate_step.payload.output.
    """
    intermediate_step: IntermediateStep | None = Field(
        default=None,
        description="The single IntermediateStep that was selected and evaluated (based on reduction strategy)"
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if any step of the evaluation has failed"
    )

    @classmethod
    def empty(cls, id: str, error: str | None = None) -> ConditionEvaluationResult:
        """
        Create an empty ConditionEvaluationResult.

        Returns:
            Empty ConditionEvaluationResult instance
        """
        return cls(
            id=id,
            score=0.0,
            reasoning = {},
            error_message=error,
            intermediate_step=None
        )


class RedTeamingEvalOutputItem(EvalOutputItem):
    """
    Extended evaluation output item for red teaming evaluations.

    Organizes results by filter condition name, with each condition containing
    its score, the evaluated output, and the single intermediate step that was selected.

    Attributes:
        id: Identifier from the input item
        score: Average score across all filter conditions
        reasoning: Summary information for compatibility
        results_by_condition: Map from condition name to evaluation results
    """
    results_by_condition: dict[str, ConditionEvaluationResult] = Field(
        description="Results organized by filter condition name"
    )
