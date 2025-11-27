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

import logging
from dataclasses import dataclass

import numpy as np
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

from .core import board_to_str
from .core import check_winner
from .core import evaluate_board_for_player
from .core import is_draw
from .core import new_board
from .llm_agents import LLMTicTacToePlayer
from .llm_agents import build_player_chain

logger = logging.getLogger(__name__)

# ---------- Game data structures ----------


@dataclass
class MoveRecord:
    turn_index: int
    player_name: str
    symbol: str
    row: int  # 0-based
    col: int  # 0-based
    score: float
    raw_llm_output: str


@dataclass
class TicTacToeGame:
    player_x: LLMTicTacToePlayer
    player_o: LLMTicTacToePlayer
    board: np.ndarray
    history: list[MoveRecord]

    def __init__(self, player_x: LLMTicTacToePlayer, player_o: LLMTicTacToePlayer):
        self.player_x = player_x
        self.player_o = player_o
        self.board = new_board()
        self.history = []

    def play(self) -> int:
        """Run the full game loop until win or draw."""

        current_player = self.player_x
        turn_index = 0

        logger.debug("=== Starting LLM vs LLM Tic-Tac-Toe (XML moves) ===")
        logger.debug("Initial board:")
        logger.debug("\n" + board_to_str(self.board))

        while True:
            logger.debug(f"\n--- Turn {turn_index + 1}: {current_player.name} ({current_player.symbol}) ---")
            logger.debug("Current board:")
            logger.debug("\n" + board_to_str(self.board))

            # Ask LLM for a move (with retries)
            row, col, raw = current_player.choose_move(self.board)

            # Apply move
            self.board[row, col] = current_player.value

            # Heuristic score *after* move, from current player's perspective
            score = evaluate_board_for_player(self.board, current_player.value)

            self.history.append(
                MoveRecord(
                    turn_index=turn_index,
                    player_name=current_player.name,
                    symbol=current_player.symbol,
                    row=row,
                    col=col,
                    score=score,
                    raw_llm_output=raw,
                ))

            logger.debug(f"{current_player.name} plays at (row={row+1}, col={col+1}).")
            logger.debug(f"Heuristic score for this move (from {current_player.symbol}'s perspective): {score:.2f}")
            logger.debug("Board after move:")
            logger.debug("\n" + board_to_str(self.board))

            # Check game termination
            winner_val = check_winner(self.board)
            if winner_val != 0:
                winner_symbol = "X" if winner_val == 1 else "O"
                winner_name = (self.player_x.name if winner_symbol == "X" else self.player_o.name)
                logger.debug(f"*** Game over! {winner_name} ({winner_symbol}) wins. ***")
                return winner_val

            if is_draw(self.board):
                logger.debug("*** Game over! It's a draw. ***")
                return 0  # Draw

            # Swap players
            current_player = self.player_o if current_player is self.player_x else self.player_x
            turn_index += 1


class RlWithOpenpipeArtFunctionConfig(FunctionBaseConfig, name="rl_with_openpipe_art"):
    """
    NAT function template. Please update the description.
    """
    larger_model: LLMRef = Field(description="LLMRef for the larger model to use.")
    smaller_model: LLMRef = Field(description="LLMRef for the smaller model to use.")
    max_parser_retries: int = Field(default=3, description="Maximum number of retries for parsing LLM output.")
    play_larger_random: bool = Field(
        default=False, description="If true, the larger model will play randomly instead of using the LLM chain.")


@register_function(config_type=RlWithOpenpipeArtFunctionConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def rl_with_openpipe_art_function(config: RlWithOpenpipeArtFunctionConfig, builder: Builder):
    """
    Registers a function (addressable via `rl_with_openpipe_art` in the configuration).
    This registration ensures a static mapping of the function type, `rl_with_openpipe_art`, to the
    `RlWithOpenpipeArtFunctionConfig` configuration object.

    Args:
        config (RlWithOpenpipeArtFunctionConfig): The configuration for the function.
        builder (Builder): The builder object.

    Returns:
        FunctionInfo: The function info object for the function.
    """

    smaller_model = await builder.get_llm(config.smaller_model, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    larger_model = await builder.get_llm(config.larger_model, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    max_retries = config.max_parser_retries

    # Define the function that will be registered.
    async def _echo(role: str) -> str:
        """
        Takes a text input and echoes back with a pre-defined prefix.

        Args:
            role (str): If smaller model will be X or O

        Returns:
            str: The text with the prefix.
        """

        if role not in ["X", "O"]:
            raise ValueError("Role must be either 'X' or 'O'.")

        if role == "X":
            player_x = LLMTicTacToePlayer(
                name="Smaller Model",
                symbol="X",
                value=1,
                chain=build_player_chain(smaller_model, "X"),
                max_retries=max_retries,
            )
            player_o = LLMTicTacToePlayer(
                name="Larger Model",
                symbol="O",
                value=-1,
                chain=build_player_chain(larger_model, "O"),
                max_retries=max_retries,
                choose_random=config.play_larger_random,
            )
        else:
            player_o = LLMTicTacToePlayer(
                name="Smaller Model",
                symbol="O",
                value=-1,
                chain=build_player_chain(smaller_model, "O"),
                max_retries=max_retries,
            )
            player_x = LLMTicTacToePlayer(
                name="Larger Model",
                symbol="X",
                value=1,
                chain=build_player_chain(larger_model, "X"),
                max_retries=max_retries,
                choose_random=config.play_larger_random,
            )

        game = TicTacToeGame(player_x=player_x, player_o=player_o)
        winner = game.play()
        return str(winner)

    # The callable is wrapped in a FunctionInfo object.
    # The description parameter is used to describe the function.
    yield FunctionInfo.from_fn(_echo, description=_echo.__doc__)
