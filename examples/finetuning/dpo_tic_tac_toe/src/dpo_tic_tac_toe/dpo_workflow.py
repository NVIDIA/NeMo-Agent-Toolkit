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

"""
DPO Tic-Tac-Toe Workflow

This workflow demonstrates how to use NAT's Test Time Compute (TTC) harness
to generate preference data for Direct Preference Optimization (DPO) finetuning.

For each turn of the trained player, it calls the ttc_move_selector function which:
1. Generates N candidate moves using a TTC search strategy
2. Scores each move using a TTC scoring strategy
3. Selects the best move using a TTC selection strategy
4. Records ALL candidate moves as intermediate steps for DPO data collection

The intermediate steps include full metadata for each candidate move, enabling
a custom trajectory builder to construct DPO preference pairs in Phase 2.
"""

import logging
import random

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

from .core import available_moves
from .core import board_to_list
from .core import board_to_str
from .core import check_winner
from .core import is_draw
from .core import new_board
from .llm_agents import build_player_chain
from .llm_agents import parse_move_any

logger = logging.getLogger(__name__)


class DPOTicTacToeConfig(FunctionBaseConfig, name="dpo_tic_tac_toe"):
    """Configuration for the DPO Tic-Tac-Toe workflow."""

    opponent_llm: LLMRef | None = Field(default=None, description="LLM for opponent (None=random)")
    ttc_move_selector_fn: FunctionRef = Field(
        description="Reference to the ttc_move_selector NAT Function that wraps search/score/select"
    )
    max_parser_retries: int = Field(default=2, description="Max retries for opponent LLM parsing")


@register_function(config_type=DPOTicTacToeConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def dpo_tic_tac_toe_workflow(config: DPOTicTacToeConfig, builder: Builder):
    """
    DPO Tic-Tac-Toe workflow that generates preference data for finetuning.

    This workflow uses the ttc_move_selector function to handle the complete
    TTC pipeline (search → score → select) for choosing moves. The
    ttc_move_selector function also handles recording intermediate steps
    for DPO data collection.

    Args:
        config: Workflow configuration
        builder: NAT builder for loading components

    Yields:
        FunctionInfo wrapping the game play function
    """
    # Get the TTC move selector function (wraps search/score/select)
    ttc_move_selector = await builder.get_function(config.ttc_move_selector_fn)

    # Get LLM for opponent (if specified)
    opponent_llm = None
    if config.opponent_llm:
        opponent_llm = await builder.get_llm(config.opponent_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    async def _play_game(role: str) -> str:
        """
        Play a game of Tic-Tac-Toe with DPO data collection.

        For each turn of the trained player, calls the ttc_move_selector
        function which handles generating candidates, scoring, selecting,
        and recording intermediate steps.

        Args:
            role: "X" or "O" - which side the trained player plays

        Returns:
            Game outcome: "Win!", "Lose!", or "Draw!"
        """
        if role not in ["X", "O"]:
            raise ValueError("Role must be either 'X' or 'O'.")

        board = new_board()

        # Determine player values
        trained_symbol = role
        trained_value = 1 if role == "X" else -1

        # Build opponent chain if using LLM opponent
        opponent_chain = None
        if opponent_llm is not None:
            opponent_symbol = "O" if role == "X" else "X"
            opponent_chain = build_player_chain(opponent_llm, opponent_symbol)

        # Game loop
        current_symbol = "X"  # X always starts
        turn_index = 0

        logger.debug("=== Starting DPO Tic-Tac-Toe Game ===")
        logger.debug(f"Trained player: {trained_symbol}")
        logger.debug("Initial board:")
        logger.debug("\n" + board_to_str(board))

        while True:
            current_value = 1 if current_symbol == "X" else -1
            is_trained_player_turn = current_symbol == trained_symbol

            logger.debug(f"\n--- Turn {turn_index + 1}: {current_symbol} ---")
            logger.debug("Current board:")
            logger.debug("\n" + board_to_str(board))

            if is_trained_player_turn:
                # === Trained player's turn: use TTC move selector ===
                try:
                    # Call the TTC move selector function
                    # This handles: search → score → select → record intermediate steps
                    move_result = await ttc_move_selector.ainvoke({
                        "board": board_to_list(board),
                        "player_symbol": current_symbol,
                        "turn_index": turn_index,
                    })

                    # Extract selected move
                    if hasattr(move_result, "row"):
                        selected_row, selected_col = move_result.row, move_result.col
                    else:
                        selected_row, selected_col = move_result["row"], move_result["col"]

                    board[selected_row, selected_col] = current_value
                    logger.debug(f"Trained player plays at ({selected_row + 1}, {selected_col + 1})")

                except RuntimeError as e:
                    logger.error(f"TTC move selector failed: {e}")
                    return "Lose!"

            else:
                # === Opponent's turn ===
                if opponent_chain is None:
                    # Random opponent
                    legal_moves = available_moves(board)
                    if not legal_moves:
                        logger.error("No legal moves for opponent!")
                        break
                    row, col = random.choice(legal_moves)
                    logger.debug(f"Random opponent plays at ({row + 1}, {col + 1})")
                else:
                    # LLM opponent
                    opponent_messages = []
                    legal_moves = available_moves(board)

                    for attempt in range(config.max_parser_retries + 1):
                        board_str = board_to_str(board)
                        from langchain_core.messages import AIMessage
                        from langchain_core.messages import HumanMessage

                        if attempt > 0:
                            opponent_messages.append(
                                HumanMessage(
                                    content=f"Invalid move. Available: "
                                    f"{', '.join(f'({r+1},{c+1})' for r,c in legal_moves)}.\n"
                                    f"Board:\n{board_str}"
                                )
                            )
                        else:
                            opponent_messages.append(HumanMessage(content=board_str))

                        response = await opponent_chain.ainvoke({"messages": opponent_messages})
                        text = str(response)
                        opponent_messages.append(AIMessage(content=text))

                        move = parse_move_any(text)
                        if move and move in legal_moves:
                            row, col = move
                            break
                    else:
                        # Fallback to random on failure
                        row, col = random.choice(legal_moves)

                    logger.debug(f"LLM opponent plays at ({row + 1}, {col + 1})")

                board[row, col] = current_value

            # Check game end conditions
            logger.debug("Board after move:")
            logger.debug("\n" + board_to_str(board))

            winner = check_winner(board)
            if winner != 0:
                winner_symbol = "X" if winner == 1 else "O"
                logger.debug(f"*** Game over! {winner_symbol} wins. ***")

                if winner == trained_value:
                    return "Win!"
                else:
                    return "Lose!"

            if is_draw(board):
                logger.debug("*** Game over! It's a draw. ***")
                return "Draw!"

            # Switch to next player
            current_symbol = "O" if current_symbol == "X" else "X"
            turn_index += 1

    yield FunctionInfo.from_fn(
        _play_game,
        description="Play a game of Tic-Tac-Toe with DPO preference data collection.",
    )
