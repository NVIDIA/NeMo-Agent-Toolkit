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
NAT Function that wraps TTC search → score → select pipeline for move selection.

This function encapsulates the entire TTC pipeline for choosing a move:
1. SEARCH: Generate N candidate moves using MultiCandidateMoveSearcher
2. SCORE: Evaluate each move using BoardPositionScorer
3. SELECT: Choose the best move using BestOfN selector

It also records all candidates as intermediate steps for DPO data collection.
"""

import logging
import uuid

from pydantic import BaseModel
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import TTCStrategyRef
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import TTCEventData
from nat.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from nat.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from nat.experimental.test_time_compute.models.ttc_item import TTCItem

from .choose_move_function import ChooseMoveOutput

logger = logging.getLogger(__name__)


class TTCMoveSelectorInput(BaseModel):
    """Input schema for the TTC move selector function."""

    board: list[list[int]] = Field(description="3x3 board state as nested list (0=empty, 1=X, -1=O)")
    player_symbol: str = Field(description="Player symbol: 'X' or 'O'")
    turn_index: int = Field(description="Current turn index for tracking")


class TTCMoveSelectorOutput(BaseModel):
    """Output schema for the TTC move selector function."""

    row: int = Field(description="0-based row index of the selected move")
    col: int = Field(description="0-based column index of the selected move")
    raw_response: str = Field(description="Raw LLM response of the selected move")
    score: float = Field(description="Score of the selected move")
    num_candidates: int = Field(description="Number of candidates that were evaluated")


class TTCMoveSelectorConfig(FunctionBaseConfig, name="ttc_move_selector"):
    """
    Configuration for the TTC move selector function.

    This function wraps the complete TTC pipeline:
    - search: Generates multiple candidate moves
    - scorer: Evaluates each candidate using game-theoretic scoring
    - selector: Selects the best move
    """

    search: TTCStrategyRef = Field(description="TTC search strategy for generating candidates")
    scorer: TTCStrategyRef = Field(description="TTC scoring strategy for evaluating moves")
    selector: TTCStrategyRef = Field(description="TTC selection strategy for choosing best move")


@register_function(config_type=TTCMoveSelectorConfig)
async def ttc_move_selector_function(config: TTCMoveSelectorConfig, builder: Builder):
    """
    NAT Function that wraps TTC search → score → select for move selection.

    This function:
    1. Generates N candidate moves using the search strategy
    2. Scores each candidate using the scorer strategy
    3. Selects the best move using the selector strategy
    4. Records ALL candidates as intermediate steps for DPO data collection

    Args:
        config: Configuration with references to TTC strategies
        builder: NAT builder for loading components

    Yields:
        FunctionInfo wrapping the move selection function
    """
    # Get TTC strategies
    searcher = await builder.get_ttc_strategy(
        strategy_name=config.search,
        pipeline_type=PipelineTypeEnum.AGENT_EXECUTION,
        stage_type=StageTypeEnum.SEARCH,
    )
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

    async def _select_move(input_data: TTCMoveSelectorInput) -> TTCMoveSelectorOutput:
        """
        Select the best move using the TTC pipeline.

        Args:
            input_data: Board state, player symbol, and turn index

        Returns:
            TTCMoveSelectorOutput with the selected move and metadata
        """
        step_manager = Context.get().intermediate_step_manager

        board = input_data.board
        player_symbol = input_data.player_symbol
        turn_index = input_data.turn_index

        player_value = 1 if player_symbol == "X" else -1
        turn_id = f"turn_{turn_index}_{uuid.uuid4().hex[:8]}"

        # Create initial TTCItem for the search strategy
        initial_item = TTCItem(
            input={
                "board": board, "player_symbol": player_symbol
            },
            metadata={"turn_index": turn_index},
        )

        # === TTC Pipeline ===

        # 1. SEARCH: Generate N candidate moves
        candidate_items = await searcher.ainvoke([initial_item])

        if not candidate_items:
            raise RuntimeError("No valid candidate moves generated!")

        # 2. SCORE: Evaluate each candidate
        scored_items = await scorer.ainvoke(candidate_items)

        # 3. SELECT: Choose the best move
        selected_items = await selector.ainvoke(scored_items)
        selected_item = selected_items[0]

        # === Record intermediate steps for ALL candidates ===
        for idx, item in enumerate(scored_items):
            move_id = f"{turn_id}_move_{idx}"
            is_selected = item is selected_item

            # Extract move data including messages
            move_output = item.output

            if not isinstance(move_output, ChooseMoveOutput):
                # Attempt to cast or raise error
                if isinstance(move_output, dict):
                    move_output = ChooseMoveOutput(**move_output)
                else:
                    raise TypeError(f"Expected ChooseMoveOutput, got {type(move_output)}")

            row, col = move_output.row, move_output.col
            raw_response = move_output.raw_response

            step_uuid = str(uuid.uuid4())[:8]

            # Write CUSTOM_START
            step_manager.push_intermediate_step(
                IntermediateStepPayload(
                    event_type=IntermediateStepType.TTC_START,
                    name="dpo_candidate_move",
                    data=TTCEventData(
                        turn_id=turn_id,
                        turn_index=turn_index,
                        candidate_index=idx,
                    ),
                    metadata={
                        "move_id": move_id,
                    },
                    UUID=step_uuid,
                ))

            # Write CUSTOM_END with full move data including messages
            step_manager.push_intermediate_step(
                IntermediateStepPayload(
                    event_type=IntermediateStepType.TTC_END,
                    name="dpo_candidate_move",
                    data=TTCEventData(
                        turn_id=turn_id,
                        turn_index=turn_index,
                        candidate_index=idx,
                        input=move_output.messages,
                        output=raw_response,
                        score=item.score,
                    ),
                    metadata={
                        "move_id": move_id,
                        "turn_index": turn_index,
                        "candidate_index": idx,
                        "board_state_before": board,
                        "move": {
                            "row": row, "col": col
                        },
                        "raw_llm_response": raw_response,
                        "is_selected": is_selected,
                        "player_symbol": player_symbol,
                        "player_value": player_value,
                    },
                    UUID=step_uuid,
                ))

        # Extract selected move data
        selected_output = selected_item.output
        selected_row = selected_output.row
        selected_col = selected_output.col
        selected_raw = selected_output.raw_response

        return TTCMoveSelectorOutput(
            row=selected_row,
            col=selected_col,
            raw_response=selected_raw,
            score=selected_item.score,
            num_candidates=len(scored_items),
        )

    yield FunctionInfo.from_fn(
        _select_move,
        description="Select the best Tic-Tac-Toe move using TTC search, score, and select pipeline.",
    )
