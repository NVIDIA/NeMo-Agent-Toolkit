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
DPO Tic-Tac-Toe Workflow

This workflow demonstrates how to use NAT's Test Time Compute (TTC) harness
to generate preference data for Direct Preference Optimization (DPO) finetuning.

For EACH turn (both trained player and opponent), it calls a ttc_move_selector
function which:
1. Generates N candidate moves using a TTC search strategy
2. Scores each move using a TTC scoring strategy
3. Selects the best move using a TTC selection strategy
4. Records ALL candidate moves as intermediate steps for DPO data collection

This enables DPO data collection from ALL game turns, not just the trained
player's turns. The opponent can use either an LLM or random move generation
(configured via the opponent's choose_move function).
"""

import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef
from nat.data_models.function import FunctionBaseConfig

from .core import board_to_list
from .core import board_to_str
from .core import check_winner
from .core import is_draw
from .core import new_board
from .ttc_move_selector_function import TTCMoveSelectorInput

logger = logging.getLogger(__name__)


class DPOTicTacToeConfig(FunctionBaseConfig, name="dpo_tic_tac_toe"):
    """
    Configuration for the DPO Tic-Tac-Toe workflow.

    Both players use TTC pipelines, enabling DPO data collection from all turns.
    The trained player typically uses an LLM-based choose_move function, while
    the opponent can use either LLM or random move generation.
    """

    trained_ttc_move_selector_fn: FunctionRef = Field(description="TTC move selector for trained player (uses LLM)")
    opponent_ttc_move_selector_fn: FunctionRef = Field(
        description="TTC move selector for opponent (can use LLM or random)")


@register_function(config_type=DPOTicTacToeConfig)
async def dpo_tic_tac_toe_workflow(config: DPOTicTacToeConfig, builder: Builder):
    """
    DPO Tic-Tac-Toe workflow that generates preference data for finetuning.

    Both players use TTC pipelines for move selection. Each ttc_move_selector:
    1. Generates N candidate moves (via search strategy)
    2. Scores each candidate (via scoring strategy)
    3. Selects the best move (via selection strategy)
    4. Records ALL candidates as intermediate steps for DPO data

    This enables DPO data collection from ALL turns, not just trained player.

    Args:
        config: Workflow configuration
        builder: NAT builder for loading components

    Yields:
        FunctionInfo wrapping the game play function
    """
    # Get TTC move selectors for both players
    trained_move_selector = await builder.get_function(config.trained_ttc_move_selector_fn)
    opponent_move_selector = await builder.get_function(config.opponent_ttc_move_selector_fn)

    async def _play_game(role: str) -> str:
        """
        Play a game of Tic-Tac-Toe with DPO data collection.

        Both players use TTC pipelines - the trained player uses an LLM-based
        pipeline while the opponent uses random (or LLM) based pipeline.
        All candidate moves from both players are recorded for DPO data.

        Args:
            role: "X" or "O" - which side the trained player plays

        Returns:
            Game outcome: "Win!", "Lose!", or "Draw!"
        """
        if role not in ["X", "O"]:
            raise ValueError("Role must be either 'X' or 'O'.")

        board = new_board()
        trained_symbol = role
        trained_value = 1 if role == "X" else -1

        current_symbol = "X"  # X always starts
        turn_index = 0

        logger.debug("=== Starting DPO Tic-Tac-Toe Game ===")
        logger.debug(f"Trained player: {trained_symbol}")
        logger.debug("Initial board:")
        logger.debug("\n" + board_to_str(board))

        while True:
            current_value = 1 if current_symbol == "X" else -1
            is_trained_turn = current_symbol == trained_symbol

            logger.debug(f"\n--- Turn {turn_index + 1}: {current_symbol} ---")
            logger.debug("Current board:")
            logger.debug("\n" + board_to_str(board))

            # Select the appropriate TTC move selector
            move_selector = (trained_move_selector if is_trained_turn else opponent_move_selector)
            player_type = "Trained" if is_trained_turn else "Opponent"

            try:
                # Call TTC move selector (search → score → select)
                # This records ALL candidates as intermediate steps
                input_dict = {
                    "board": board_to_list(board),
                    "player_symbol": current_symbol,
                    "turn_index": turn_index,
                }
                move_result = await move_selector.ainvoke(TTCMoveSelectorInput(**input_dict))

                # Extract selected move
                if hasattr(move_result, "row"):
                    row, col = move_result.row, move_result.col
                else:
                    row, col = move_result["row"], move_result["col"]

                board[row, col] = current_value
                logger.debug(f"{player_type} plays at ({row + 1}, {col + 1})")

            except RuntimeError as e:
                logger.error(f"{player_type} move selector failed: {e}")
                # If trained player fails, they lose; if opponent fails, trained wins
                return "Lose!" if is_trained_turn else "Win!"

            # Check game end conditions
            logger.debug("Board after move:")
            logger.debug("\n" + board_to_str(board))

            winner = check_winner(board)
            if winner != 0:
                winner_symbol = "X" if winner == 1 else "O"
                logger.debug(f"*** Game over! {winner_symbol} wins. ***")
                return "Win!" if winner == trained_value else "Lose!"

            if is_draw(board):
                logger.debug("*** Game over! It's a draw. ***")
                return "Draw!"

            # Switch to next player
            current_symbol = "O" if current_symbol == "X" else "X"
            turn_index += 1

    yield FunctionInfo.from_fn(
        _play_game,
        description="Play Tic-Tac-Toe with DPO data collection from all turns.",
    )
