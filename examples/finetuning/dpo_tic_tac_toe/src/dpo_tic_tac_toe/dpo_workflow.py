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

For each turn of the trained player:
1. Generate N candidate moves using the choose_move NAT Function
2. Score each move using the board_position_scorer TTC strategy
3. Select the best move using BestOfN selection
4. Record ALL candidate moves as intermediate steps for DPO data collection

The intermediate steps include full metadata for each candidate move, enabling
a custom trajectory builder to construct DPO preference pairs in Phase 2.
"""

import logging
import random
import uuid

import numpy as np
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.component_ref import TTCStrategyRef
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from nat.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from nat.experimental.test_time_compute.models.ttc_item import TTCItem

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

    player_llm: LLMRef = Field(description="LLM for the trained player")
    opponent_llm: LLMRef | None = Field(default=None, description="LLM for opponent (None=random)")
    choose_move_fn: FunctionRef = Field(description="Reference to the choose_move NAT Function")
    scorer: TTCStrategyRef = Field(description="TTC scorer strategy (board_position_scorer)")
    selector: TTCStrategyRef = Field(description="TTC selector strategy (best_of_n_selection)")
    num_candidates: int = Field(default=3, ge=1, description="Number of candidate moves per turn")
    max_parser_retries: int = Field(default=2, description="Max retries for opponent LLM parsing")


@register_function(config_type=DPOTicTacToeConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def dpo_tic_tac_toe_workflow(config: DPOTicTacToeConfig, builder: Builder):
    """
    DPO Tic-Tac-Toe workflow that generates preference data for finetuning.

    This workflow:
    1. Generates multiple candidate moves per turn using TTC
    2. Scores each move using game-theoretic evaluation
    3. Selects the best move to play
    4. Records ALL candidate moves as intermediate steps for DPO data collection

    The recorded intermediate steps can be collected by a custom trajectory
    builder to create DPO preference pairs.

    Args:
        config: Workflow configuration
        builder: NAT builder for loading components

    Yields:
        FunctionInfo wrapping the game play function
    """
    # Get the choose_move function
    choose_move_fn = await builder.get_function(config.choose_move_fn)

    # Get TTC strategies
    scorer = await builder.get_ttc_strategy(
        strategy_name=config.scorer,
        pipeline_type=PipelineTypeEnum.AGENT_EXECUTION,
        stage_type=StageTypeEnum.SCORING,
    )
    selector = await builder.get_ttc_strategy(
        strategy_name=config.selector,
        pipeline_type=PipelineTypeEnum.AGENT_EXECUTION,
        stage_type=StageTypeEnum.SELECTION,
    )

    # Get LLMs for opponent (if specified)
    opponent_llm = None
    if config.opponent_llm:
        opponent_llm = await builder.get_llm(config.opponent_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    async def _play_game(role: str) -> str:
        """
        Play a game of Tic-Tac-Toe with DPO data collection.

        For each turn of the trained player, multiple candidate moves are
        generated, scored, and recorded as intermediate steps. The best
        move is selected and played.

        Args:
            role: "X" or "O" - which side the trained player plays

        Returns:
            Game outcome: "Win!", "Lose!", or "Draw!"
        """
        if role not in ["X", "O"]:
            raise ValueError("Role must be either 'X' or 'O'.")

        step_manager = Context.get().intermediate_step_manager
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
                # === TTC Pipeline for trained player ===
                turn_id = f"turn_{turn_index}_{uuid.uuid4().hex[:8]}"

                # Generate N candidate moves
                candidate_items: list[TTCItem] = []
                for candidate_idx in range(config.num_candidates):
                    try:
                        # Call choose_move function
                        move_result = await choose_move_fn.ainvoke({
                            "board": board_to_list(board),
                            "player_symbol": current_symbol,
                        })

                        # Wrap in TTCItem with metadata for scoring
                        item = TTCItem(
                            input={"board": board_to_list(board), "player_symbol": current_symbol},
                            output=move_result,
                            metadata={
                                "board": board_to_list(board),
                                "player_value": current_value,
                                "candidate_idx": candidate_idx,
                            },
                        )
                        candidate_items.append(item)

                    except RuntimeError as e:
                        logger.warning(f"Failed to generate candidate {candidate_idx}: {e}")
                        continue

                if not candidate_items:
                    logger.error("No valid candidate moves generated!")
                    return "Lose!"

                # Score all candidates using TTC scorer
                scored_items = await scorer.ainvoke(candidate_items)

                # Select best move using TTC selector
                selected_items = await selector.ainvoke(scored_items)
                selected_item = selected_items[0]

                # === Write intermediate steps for ALL candidates ===
                for idx, item in enumerate(scored_items):
                    move_id = f"{turn_id}_move_{idx}"
                    is_selected = item is selected_item

                    # Get move data
                    move_output = item.output
                    if hasattr(move_output, "row"):
                        row, col = move_output.row, move_output.col
                        raw_response = move_output.raw_response
                    else:
                        row, col = move_output["row"], move_output["col"]
                        raw_response = move_output["raw_response"]

                    step_uuid = str(uuid.uuid4())[:8]

                    # Write CUSTOM_START
                    step_manager.push_intermediate_step(
                        IntermediateStepPayload(
                            event_type=IntermediateStepType.CUSTOM_START,
                            name="dpo_candidate_move",
                            metadata={
                                "turn_id": turn_id,
                                "move_id": move_id,
                                "turn_index": turn_index,
                                "candidate_index": idx,
                            },
                            UUID=step_uuid,
                        )
                    )

                    # Write CUSTOM_END with full move data
                    step_manager.push_intermediate_step(
                        IntermediateStepPayload(
                            event_type=IntermediateStepType.CUSTOM_END,
                            name="dpo_candidate_move",
                            metadata={
                                "turn_id": turn_id,
                                "move_id": move_id,
                                "turn_index": turn_index,
                                "candidate_index": idx,
                                "board_state_before": board_to_list(board),
                                "move": {"row": row, "col": col},
                                "raw_llm_response": raw_response,
                                "score": item.score,
                                "is_selected": is_selected,
                                "player_symbol": current_symbol,
                                "player_value": current_value,
                            },
                            UUID=step_uuid,
                        )
                    )

                # Apply the selected move
                selected_output = selected_item.output
                if hasattr(selected_output, "row"):
                    selected_row, selected_col = selected_output.row, selected_output.col
                else:
                    selected_row, selected_col = selected_output["row"], selected_output["col"]

                board[selected_row, selected_col] = current_value
                logger.debug(f"Trained player plays at ({selected_row + 1}, {selected_col + 1})")

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
