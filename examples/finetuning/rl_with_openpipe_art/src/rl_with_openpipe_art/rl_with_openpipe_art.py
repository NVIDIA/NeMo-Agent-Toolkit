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

import logging
import uuid
from dataclasses import dataclass

import numpy as np
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.builder.intermediate_step_manager import IntermediateStepManager
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType

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
    raw_llm_output: str


@dataclass
class TicTacToeGame:
    player_x: LLMTicTacToePlayer
    player_o: LLMTicTacToePlayer
    board: np.ndarray
    history: list[MoveRecord]

    def __init__(self, player_x: LLMTicTacToePlayer, player_o: LLMTicTacToePlayer, role: str):
        self.player_x = player_x
        self.player_o = player_o
        self.board = new_board()

        if role == "X":
            self.role = player_x.name
        else:
            self.role = player_o.name

        self.history = []
        self.step_manager: IntermediateStepManager = Context.get().intermediate_step_manager

    def play(self) -> int:
        """Run the full game loop until win or draw."""

        current_player = self.player_x
        turn_index = 0

        logger.debug("=== Starting LLM vs LLM Tic-Tac-Toe (XML moves) ===")
        logger.debug("Initial board:")
        logger.debug("\n" + board_to_str(self.board))

        try:
            while True:
                logger.debug(f"\n--- Turn {turn_index + 1}: {current_player.name} ({current_player.symbol}) ---")
                logger.debug("Current board:")
                logger.debug("\n" + board_to_str(self.board))

                # Ask LLM for a move (with retries)
                row, col, raw = current_player.choose_move(self.board)

                # Apply move
                self.board[row, col] = current_player.value

                # Create an intermediate step for the value of the current agent move - better evaluations
                if current_player.name == self.role:
                    uuid_str = str(uuid.uuid4())[:8]

                    start_payload = IntermediateStepPayload(event_type=IntermediateStepType.CUSTOM_START,
                                                            name="agent_move",
                                                            metadata={
                                                                "agent_name": current_player.name,
                                                                "step": turn_index,
                                                                "symbol": current_player.symbol,
                                                            },
                                                            UUID=uuid_str)

                    self.step_manager.push_intermediate_step(start_payload)

                    end_payload = IntermediateStepPayload(
                        event_type=IntermediateStepType.CUSTOM_END,
                        name="agent_move",
                        metadata={
                            "agent_name": current_player.name,
                            "step": turn_index,
                            "symbol": current_player.symbol,
                            "value": evaluate_board_for_player(self.board, current_player.value)
                        },
                        UUID=uuid_str)

                    self.step_manager.push_intermediate_step(end_payload)

                self.history.append(
                    MoveRecord(
                        turn_index=turn_index,
                        player_name=current_player.name,
                        symbol=current_player.symbol,
                        row=row,
                        col=col,
                        raw_llm_output=raw,
                    ))

                logger.debug(f"{current_player.name} plays at (row={row+1}, col={col+1}).")
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

        except RuntimeError as _:
            logger.debug("*** Game aborted due to too many invalid moves. ***")
            return current_player.steps


class RlWithOpenpipeArtFunctionConfig(FunctionBaseConfig, name="rl_with_openpipe_art"):
    """
    NAT function template. Please update the description.
    """
    player_model: LLMRef = Field(description="LLMRef for the player model to use.")
    opponent_model: LLMRef | None = Field(description="LLMRef for the opponent model to use.", default=None)
    max_parser_retries: int = Field(default=0, description="Maximum number of retries for parsing LLM output.")


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

    player_model = await builder.get_llm(config.player_model, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    opponent_model = await builder.get_llm(
        config.opponent_model, wrapper_type=LLMFrameworkEnum.LANGCHAIN) if config.opponent_model else player_model
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
                chain=build_player_chain(player_model, "X"),
                max_retries=max_retries,
            )
            player_o = LLMTicTacToePlayer(
                name="Larger Model",
                symbol="O",
                value=-1,
                chain=build_player_chain(opponent_model, "O"),
                max_retries=max_retries,
                choose_random=True if config.opponent_model is None else False,
            )
        else:
            player_o = LLMTicTacToePlayer(
                name="Smaller Model",
                symbol="O",
                value=-1,
                chain=build_player_chain(player_model, "O"),
                max_retries=max_retries,
            )
            player_x = LLMTicTacToePlayer(
                name="Larger Model",
                symbol="X",
                value=1,
                chain=build_player_chain(opponent_model, "X"),
                max_retries=max_retries,
                choose_random=True if config.opponent_model is None else False,
            )

        game = TicTacToeGame(player_x=player_x, player_o=player_o, role=role)
        winner = game.play()

        if role == "X":
            if winner == 1:
                return "Win!"
            elif winner == -1:
                return "Lose!"
            elif winner == 0:
                return "Draw!"
            else:
                return str(winner)
        elif winner == -1:
            return "Win!"
        elif winner == 1:
            return "Lose!"
        elif winner == 0:
            return "Draw!"
        else:
            return str(winner)

    # The callable is wrapped in a FunctionInfo object.
    # The description parameter is used to describe the function.
    yield FunctionInfo.from_fn(_echo, description=_echo.__doc__)
