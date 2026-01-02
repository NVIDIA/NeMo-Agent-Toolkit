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
Evaluators for the DPO Tic-Tac-Toe workflow.

This module provides evaluators for scoring game outcomes and collecting
intermediate step data for DPO preference dataset construction.
"""

from typing import override

from nat.eval.evaluator.base_evaluator import BaseEvaluator
from nat.eval.evaluator.evaluator_model import EvalInputItem
from nat.eval.evaluator.evaluator_model import EvalOutputItem


class GameOutcomeEvaluator(BaseEvaluator):
    """
    Simple evaluator for game outcomes.

    Scoring logic:
    - Win: 1.0
    - Draw: 0.5
    - Lose: 0.0
    """

    def __init__(self, max_concurrency: int = 4):
        super().__init__(max_concurrency, tqdm_desc="Evaluating game outcomes")

    @override
    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        """Evaluate a single game based on the outcome."""
        workflow_output = str(item.output_obj)

        # Scoring logic
        if workflow_output == "Win!":
            score = 1.0
            status = "win"
        elif workflow_output == "Draw!":
            score = 0.5
            status = "draw"
        elif workflow_output == "Lose!":
            score = 0.0
            status = "loss"
        else:
            score = 0.0
            status = "unknown"

        reasoning = {
            "question": item.input_obj,
            "expected_answer": str(item.expected_output_obj),
            "workflow_output": workflow_output,
            "status": status,
        }

        return EvalOutputItem(id=item.id, score=score, reasoning=reasoning)


class DPODataCollectorEvaluator(BaseEvaluator):
    """
    Evaluator that collects DPO preference data from intermediate steps.

    This evaluator processes the 'dpo_candidate_move' intermediate steps
    recorded during gameplay to extract preference pairs for DPO training.

    For each turn with multiple candidates, it identifies:
    - Chosen response: The move that was selected (is_selected=True)
    - Rejected responses: Other moves with lower scores
    """

    def __init__(self, max_concurrency: int = 4):
        super().__init__(max_concurrency, tqdm_desc="Collecting DPO data")

    @override
    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        """
        Process intermediate steps to extract DPO preference data.

        Returns evaluation output with reasoning containing:
        - game_outcome: Win/Lose/Draw
        - num_turns: Number of turns played
        - dpo_pairs: List of preference pairs per turn
        """
        from nat.data_models.intermediate_step import IntermediateStepType

        workflow_output = str(item.output_obj)

        # Score based on outcome (same as GameOutcomeEvaluator)
        if workflow_output == "Win!":
            score = 1.0
        elif workflow_output == "Draw!":
            score = 0.5
        else:
            score = 0.0

        # Collect all dpo_candidate_move steps
        moves_by_turn: dict[str, list[dict]] = {}

        for step in item.trajectory:
            if (step.event_type == IntermediateStepType.CUSTOM_END and step.payload.name == "dpo_candidate_move"):
                metadata = step.payload.metadata
                if metadata:
                    turn_id = metadata.get("turn_id")
                    if turn_id:
                        if turn_id not in moves_by_turn:
                            moves_by_turn[turn_id] = []
                        moves_by_turn[turn_id].append(metadata)

        # Build DPO pairs for each turn
        dpo_pairs = []
        for turn_id, moves in moves_by_turn.items():
            # Sort by score descending
            sorted_moves = sorted(moves, key=lambda m: m.get("score", 0), reverse=True)

            # Find chosen (selected) move
            chosen = next((m for m in sorted_moves if m.get("is_selected")), None)

            # All non-selected moves are potential rejected responses
            rejected = [m for m in sorted_moves if not m.get("is_selected")]

            if chosen and rejected:
                # Create preference pair with the highest-scoring rejected move
                # (more challenging comparison)
                best_rejected = rejected[0]

                dpo_pairs.append({
                    "turn_id": turn_id,
                    "turn_index": chosen.get("turn_index"),
                    "board_state": chosen.get("board_state_before"),
                    "chosen": {
                        "move": chosen.get("move"),
                        "response": chosen.get("raw_llm_response"),
                        "score": chosen.get("score"),
                    },
                    "rejected": {
                        "move": best_rejected.get("move"),
                        "response": best_rejected.get("raw_llm_response"),
                        "score": best_rejected.get("score"),
                    },
                    "score_diff": chosen.get("score", 0) - best_rejected.get("score", 0),
                })

        reasoning = {
            "question": item.input_obj,
            "game_outcome": workflow_output,
            "num_turns": len(moves_by_turn),
            "num_dpo_pairs": len(dpo_pairs),
            "dpo_pairs": dpo_pairs,
        }

        return EvalOutputItem(id=item.id, score=score, reasoning=reasoning)
