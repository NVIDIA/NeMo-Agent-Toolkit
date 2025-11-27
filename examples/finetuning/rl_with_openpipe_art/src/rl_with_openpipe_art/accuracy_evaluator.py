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

from typing import override

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

    def __init__(self, max_concurrency: int = 4):
        super().__init__(max_concurrency, tqdm_desc="Evaluating accuracy")

    @override
    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        """Evaluate a single item based on the custom scoring logic."""
        expected_answer = str(item.expected_output_obj)
        workflow_output = str(item.output_obj)

        # Scoring logic
        if expected_answer == workflow_output:
            score = 1.0
            match_status = "exact_match"
        elif expected_answer == "0":
            score = 0.5
            match_status = "mismatch_with_zero_expected"
        else:
            score = 0.0
            match_status = "mismatch"

        # The reasoning field provides detailed information about the evaluation
        reasoning = {
            "question": item.input_obj,
            "expected_answer": expected_answer,
            "workflow_output": workflow_output,
            "match_status": match_status,
        }

        return EvalOutputItem(id=item.id, score=score, reasoning=reasoning)
