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
TTC Search Strategy for generating multiple candidate moves in Tic-Tac-Toe.

This strategy generates N candidate moves by invoking the choose_move function
multiple times, wrapping each result in a TTCItem for downstream scoring and selection.
"""

import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_ttc_strategy
from nat.data_models.component_ref import FunctionRef
from nat.data_models.ttc_strategy import TTCStrategyBaseConfig
from nat.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from nat.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from nat.experimental.test_time_compute.models.strategy_base import StrategyBase
from nat.experimental.test_time_compute.models.ttc_item import TTCItem

from .choose_move_function import ChooseMoveInput

logger = logging.getLogger(__name__)


class MultiCandidateMoveSearchConfig(TTCStrategyBaseConfig, name="multi_candidate_move_search"):
    """
    Configuration for generating multiple candidate moves.

    This search strategy invokes a move generation function multiple times
    to produce N candidate moves that can then be scored and selected.
    """

    choose_move_fn: FunctionRef = Field(description="Reference to the choose_move NAT Function")
    num_candidates: int = Field(default=3, ge=1, description="Number of candidate moves to generate")


class MultiCandidateMoveSearcher(StrategyBase):
    """
    TTC Search Strategy that generates multiple candidate moves.

    This strategy expects input TTCItems with:
    - item.input: dict with 'board' (list[list[int]]) and 'player_symbol' (str)

    It produces output TTCItems with:
    - item.input: The original input
    - item.output: ChooseMoveOutput with 'row', 'col', 'raw_response'
    - item.metadata: Contains board, player_value, candidate_idx for scoring
    """

    def __init__(self, config: MultiCandidateMoveSearchConfig):
        super().__init__(config)
        self.choose_move_fn = None
        self.num_candidates = config.num_candidates

    async def build_components(self, builder: Builder) -> None:
        """Load the choose_move function from the builder."""
        self.choose_move_fn = await builder.get_function(self.config.choose_move_fn)

    def supported_pipeline_types(self) -> list[PipelineTypeEnum]:
        """Support agent execution and custom pipeline types."""
        return [PipelineTypeEnum.AGENT_EXECUTION, PipelineTypeEnum.CUSTOM]

    def stage_type(self) -> StageTypeEnum:
        """This is a search strategy."""
        return StageTypeEnum.SEARCH

    async def ainvoke(
        self,
        items: list[TTCItem],
        original_prompt: str | None = None,
        agent_context: str | None = None,
        **kwargs,
    ) -> list[TTCItem]:
        """
        Generate multiple candidate moves for each input item.

        For each input TTCItem, generates num_candidates moves by invoking
        the choose_move function multiple times.

        Args:
            items: List of TTCItems containing board state and player info
            original_prompt: Not used
            agent_context: Not used

        Returns:
            List of TTCItems, one per candidate move generated
        """
        output_items: list[TTCItem] = []

        for item in items:
            # Extract input data
            input_data = item.input
            if isinstance(input_data, dict):
                board = input_data["board"]
                player_symbol = input_data["player_symbol"]
            else:
                board = input_data.board
                player_symbol = input_data.player_symbol

            # Determine player value from symbol
            player_value = 1 if player_symbol == "X" else -1

            # Generate N candidate moves
            for candidate_idx in range(self.num_candidates):
                try:
                    # Call choose_move function
                    move_result = await self.choose_move_fn.ainvoke(
                        ChooseMoveInput(
                            board=board,
                            player_symbol=player_symbol,
                        ))

                    # Wrap in TTCItem with metadata for scoring
                    candidate_item = TTCItem(
                        input=input_data,
                        output=move_result,
                        metadata={
                            "board": board,
                            "player_value": player_value,
                            "player_symbol": player_symbol,
                            "candidate_idx": candidate_idx,
                        },
                    )
                    output_items.append(candidate_item)

                    logger.debug(f"Generated candidate {candidate_idx}: "
                                 f"row={move_result.row if hasattr(move_result, 'row') else move_result['row']}, "
                                 f"col={move_result.col if hasattr(move_result, 'col') else move_result['col']}")

                except RuntimeError as e:
                    logger.warning(f"Failed to generate candidate {candidate_idx}: {e}")
                    continue

        if not output_items:
            logger.error("No valid candidate moves generated!")

        return output_items


@register_ttc_strategy(config_type=MultiCandidateMoveSearchConfig)
async def register_multi_candidate_move_search(config: MultiCandidateMoveSearchConfig, builder: Builder):
    """Register the multi-candidate move search strategy."""
    searcher = MultiCandidateMoveSearcher(config)
    await searcher.build_components(builder)
    yield searcher
