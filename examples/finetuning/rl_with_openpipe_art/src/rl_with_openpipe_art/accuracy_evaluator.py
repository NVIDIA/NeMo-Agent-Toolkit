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

from typing import override

import numpy as np

from nat.data_models.intermediate_step import IntermediateStepType
from nat.eval.evaluator.base_evaluator import BaseEvaluator
from nat.eval.evaluator.evaluator_model import EvalInputItem
from nat.eval.evaluator.evaluator_model import EvalOutputItem


class AccuracyEvaluator(BaseEvaluator):
    """Custom evaluator for RL with OpenPipe ART workflow outputs.

    Scoring logic:
    - Score 1: if expected_answer == workflow_output
    - Score 0.5: if expected_answer != workflow_output AND expected_answer == "0"
    - Score 0: if expected_answer != workflow_output AND expected_answer != "0"
    """

    def __init__(self, max_concurrency: int = 4, use_intermediate_steps: bool = False):
        super().__init__(max_concurrency, tqdm_desc="Evaluating accuracy")
        self.use_steps = use_intermediate_steps

    @staticmethod
    def episode_value_from_states(
        state_values,  # list[float] from evaluate_board_for_player
        gamma_base: float = 0.8,
        delta_bonus: float = 0.95,
    ) -> float:
        s = np.asarray(state_values, dtype=float)
        T = len(s) - 1
        assert T >= 0

        # 1) Split into base [0,1] and bonus (>0 iff forced/actual win)
        base = np.minimum(s, 1.0)
        bonus = np.maximum(s - 1.0, 0.0)

        # 2) Reverse-discounted base in [0,1]
        exponents = np.arange(T, -1, -1)  # T, T-1, ..., 0
        w = gamma_base**exponents
        w = w / w.sum()
        R_base = float(np.dot(w, base))  # in [0,1]

        # 3) Bonus: max spike, time-decayed
        # If no spikes, this is 0.
        if np.any(bonus > 0):
            # heavier weight if the spike happens earlier
            bonus_weights = delta_bonus**exponents
            # elementwise product, then max
            U_time = float(np.max(bonus * bonus_weights))
        else:
            U_time = 0.0

        # 4) Final episode score
        R = R_base + U_time

        return R

    @staticmethod
    async def _eval_with_steps(item: EvalInputItem) -> EvalOutputItem:

        score_sum = 0.0
        scores = []
        for step in item.trajectory:
            if step.event_type == IntermediateStepType.CUSTOM_END:
                payload = step.payload
                if payload.metadata and "value" in payload.metadata:
                    step_score = float(payload.metadata["value"])
                    scores.append(step_score)
                    score_sum += step_score

        #average_score = score_sum / max(1, len(item.trajectory))
        average_score = AccuracyEvaluator.episode_value_from_states(scores)
        reasoning = {
            "question": item.input_obj,
            "expected_answer": str(item.expected_output_obj),
            "workflow_output": str(item.output_obj),
            "average_step_score": average_score,
        }

        return EvalOutputItem(id=item.id, score=average_score, reasoning=reasoning)

    @override
    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        """Evaluate a single item based on the custom scoring logic."""

        if self.use_steps:
            return await self._eval_with_steps(item)

        expected_answer = str(item.expected_output_obj)
        workflow_output = str(item.output_obj)

        # Scoring logic
        if workflow_output == "Win!":
            score = 1.0
            match_status = "exact_match"
        elif workflow_output == "Draw!":
            score = 0.5
            match_status = "mismatch_with_zero_expected"
        elif workflow_output == "Lose!":
            score = 0.0
            match_status = "loss"
        else:
            score = 0.0
            match_status = "aborted"

        # The reasoning field provides detailed information about the evaluation
        reasoning = {
            "question": item.input_obj,
            "expected_answer": expected_answer,
            "workflow_output": workflow_output,
            "match_status": match_status,
        }

        return EvalOutputItem(id=item.id, score=score, reasoning=reasoning)
