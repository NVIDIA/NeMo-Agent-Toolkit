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
Custom TTC Scorer for Tic-Tac-Toe that uses game-theoretic position evaluation.

This scorer evaluates moves using the `evaluate_board_for_player` function,
which combines heuristic evaluation with alpha-beta minimax search to provide
accurate position scores. This is faster and more accurate than LLM-based scoring
for this domain.
"""

import logging

import numpy as np

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_ttc_strategy
from nat.data_models.ttc_strategy import TTCStrategyBaseConfig
from nat.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from nat.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from nat.experimental.test_time_compute.models.strategy_base import StrategyBase
from nat.experimental.test_time_compute.models.ttc_item import TTCItem

from .core import evaluate_board_for_player

logger = logging.getLogger(__name__)


class BoardPositionScorerConfig(TTCStrategyBaseConfig, name="board_position_scorer"):
    """
    Configuration for scoring moves using game-theoretic evaluation.

    This scorer uses the `evaluate_board_for_player` function to score each
    candidate move based on the resulting board position. No additional
    configuration is needed since it uses deterministic game-theoretic evaluation.
    """

    pass


class BoardPositionScorer(StrategyBase):
    """
    Custom TTC Scorer that evaluates moves using game-theoretic position evaluation.

    This scorer expects TTCItem objects with the following structure:
    - item.output: ChooseMoveOutput with 'row', 'col', 'raw_response'
    - item.metadata: dict containing 'board' (list[list[int]]) and 'player_value' (int)

    The scorer applies each move to a copy of the board and evaluates the
    resulting position using `evaluate_board_for_player`.
    """

    def __init__(self, config: BoardPositionScorerConfig):
        super().__init__(config)

    async def build_components(self, builder: Builder) -> None:
        """No external components needed - uses deterministic evaluation."""
        pass

    def supported_pipeline_types(self) -> list[PipelineTypeEnum]:
        """Support agent execution and custom pipeline types."""
        return [PipelineTypeEnum.AGENT_EXECUTION, PipelineTypeEnum.CUSTOM]

    def stage_type(self) -> StageTypeEnum:
        """This is a scoring strategy."""
        return StageTypeEnum.SCORING

    async def ainvoke(
        self,
        items: list[TTCItem],
        original_prompt: str | None = None,
        agent_context: str | None = None,
        **kwargs,
    ) -> list[TTCItem]:
        """
        Score each candidate move using game-theoretic position evaluation.

        For each TTCItem:
        1. Extract the move (row, col) from item.output
        2. Extract the board state and player_value from item.metadata
        3. Apply the move to a copy of the board
        4. Evaluate the resulting position with evaluate_board_for_player
        5. Set item.score to the evaluation result

        Args:
            items: List of TTCItems containing candidate moves
            original_prompt: Not used (kept for interface compatibility)
            agent_context: Not used (kept for interface compatibility)

        Returns:
            The same list of TTCItems with .score set on each
        """
        for item in items:
            try:
                # Extract move from output
                move_output = item.output
                if hasattr(move_output, "row"):
                    # Pydantic model
                    row, col = move_output.row, move_output.col
                else:
                    # Dict
                    row, col = move_output["row"], move_output["col"]

                # Extract board and player value from metadata
                board_list = item.metadata["board"]
                player_value = item.metadata["player_value"]

                # Convert board to numpy array
                board = np.array(board_list, dtype=int)

                # Apply move to a copy of the board
                board_after_move = board.copy()
                board_after_move[row, col] = player_value

                # Evaluate the resulting position
                score = evaluate_board_for_player(board_after_move, player_value)
                item.score = float(score)

                logger.debug(f"Scored move ({row}, {col}) for player {player_value}: {score:.3f}")

            except Exception as e:
                logger.error(f"Error scoring item: {e}")
                # Set a low score on error so the move is deprioritized
                item.score = 0.0

        return items


@register_ttc_strategy(config_type=BoardPositionScorerConfig)
async def register_board_position_scorer(config: BoardPositionScorerConfig, builder: Builder):
    """Register the custom board position scorer strategy."""
    scorer = BoardPositionScorer(config)
    await scorer.build_components(builder)
    yield scorer
