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
NAT Function for choosing a move in Tic-Tac-Toe.

This function is designed to be invoked multiple times by the TTC harness
to generate candidate moves that can then be scored and selected.

Supports both LLM-based and random move generation:
- If an LLM is configured, uses the LLM to generate moves
- If no LLM is configured (llm=None), generates random moves

This allows the TTC pipeline to be used for both trained players (LLM)
and opponents (random), enabling DPO data collection from all turns.
"""

import logging

import numpy as np
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.finetuning import OpenAIMessage
from nat.data_models.function import FunctionBaseConfig

from .core import available_moves
from .core import board_to_str
from .llm_agents import build_player_chain
from .llm_agents import get_system_prompt
from .llm_agents import make_random_move
from .llm_agents import parse_move_any

logger = logging.getLogger(__name__)


class ChooseMoveInput(BaseModel):
    """Input schema for the choose_move function."""

    board: list[list[int]] = Field(description="3x3 board state as nested list (0=empty, 1=X, -1=O)")
    player_symbol: str = Field(description="Player symbol: 'X' or 'O'")


class ChooseMoveOutput(BaseModel):
    """Output schema for the choose_move function."""

    row: int = Field(description="0-based row index of the move")
    col: int = Field(description="0-based column index of the move")
    raw_response: str = Field(description="Raw LLM response text")
    messages: list[OpenAIMessage] = Field(
        description="Full conversation history (system, user, assistant messages) that produced this response")


class ChooseMoveConfig(FunctionBaseConfig, name="choose_move"):
    """
    Configuration for the choose_move NAT Function.

    If llm is None, the function generates random moves. This enables
    the TTC pipeline to be used for both LLM-based and random players,
    allowing DPO data collection from all game turns.
    """

    llm: LLMRef | None = Field(default=None,
                               description="LLM to use for move generation. If None, generates random moves.")
    max_retries: int = Field(default=2, description="Maximum number of parsing retries")


@register_function(config_type=ChooseMoveConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def choose_move_function(config: ChooseMoveConfig, builder: Builder):
    """
    NAT Function that generates a single move for a given board state.

    This function is designed to be called multiple times by the TTC harness
    to generate candidate moves. Each invocation produces one move suggestion.

    Supports two modes:
    - LLM mode (llm is configured): Uses the LLM to generate moves
    - Random mode (llm is None): Generates random legal moves

    Args:
        config: Configuration specifying the LLM and retry settings
        builder: NAT builder for loading LLM models

    Yields:
        FunctionInfo wrapping the move generation function
    """
    # Load LLM if configured, otherwise use random mode
    llm = None
    if config.llm is not None:
        llm = await builder.get_llm(config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    max_retries = config.max_retries
    use_random = llm is None

    def _get_message_content(msg) -> str:
        """Extract string content from a LangChain message."""
        content = msg.content
        if isinstance(content, str):
            return content
        # Handle list content (multi-part messages)
        if isinstance(content, list):
            return " ".join(str(part) for part in content)
        return str(content)

    def _build_openai_messages(
        player_symbol: str,
        langchain_messages: list,
    ) -> list[OpenAIMessage]:
        """
        Convert LangChain messages to OpenAIMessage format with system prompt.

        Args:
            player_symbol: The player symbol ('X' or 'O')
            langchain_messages: List of LangChain messages (HumanMessage, AIMessage)

        Returns:
            List of OpenAIMessage objects including system prompt
        """
        result = [OpenAIMessage(role="system", content=get_system_prompt(player_symbol))]

        for msg in langchain_messages:
            content = _get_message_content(msg)
            if isinstance(msg, HumanMessage):
                result.append(OpenAIMessage(role="user", content=content))
            elif isinstance(msg, AIMessage):
                result.append(OpenAIMessage(role="assistant", content=content))

        return result

    async def _choose_move(input_data: ChooseMoveInput) -> ChooseMoveOutput:
        """
        Generate a single move for the given board state.

        Args:
            input_data: Board state and player symbol

        Returns:
            ChooseMoveOutput with row, col, raw_response, and messages
        """

        board_list = input_data.board
        player_symbol = input_data.player_symbol

        # Convert to numpy array
        board = np.array(board_list, dtype=int)
        board_str = board_to_str(board)

        # === Random mode: generate a random legal move ===
        if use_random:
            row, col, raw_response = make_random_move(board)
            # Build messages list with system prompt and user board state
            openai_messages = [
                OpenAIMessage(role="system", content=get_system_prompt(player_symbol)),
                OpenAIMessage(role="user", content=board_str),
            ]
            return ChooseMoveOutput(row=row, col=col, raw_response=raw_response, messages=openai_messages)

        # === LLM mode: use the LLM to generate a move ===
        # Build chain for this player symbol
        chain = build_player_chain(llm, player_symbol)

        # Get available moves
        legal_moves = available_moves(board)
        if not legal_moves:
            raise RuntimeError("No available moves; game should be over.")

        # Conversation history for retries (LangChain format)
        langchain_messages: list = []

        for attempt in range(max_retries + 1):
            current_board_str = board_to_str(board)

            if attempt > 0:
                # Add retry message with available moves hint
                langchain_messages.append(
                    HumanMessage(content=f"You made an invalid move. You have "
                                 f"{max_retries - attempt + 1} attempts left.\n"
                                 f"Available moves are: "
                                 f"{', '.join(f'({r+1},{c+1})' for r, c in legal_moves)}\n"
                                 f"Current board:\n{current_board_str}"))
            else:
                langchain_messages.append(HumanMessage(content=current_board_str))

            # Invoke the LLM
            raw_response = await chain.ainvoke({"messages": langchain_messages})
            text = str(raw_response)

            # Add AI response to history
            langchain_messages.append(AIMessage(content=text))

            # Parse the move
            move = parse_move_any(text)

            if move is not None and move in legal_moves:
                # Convert to OpenAIMessage format
                openai_messages = _build_openai_messages(player_symbol, langchain_messages)
                return ChooseMoveOutput(
                    row=move[0],
                    col=move[1],
                    raw_response=text,
                    messages=openai_messages,
                )

            logger.debug(f"[WARN] Invalid move on attempt {attempt + 1}: '{text}'. "
                         f"Legal moves: {legal_moves}. Retrying...")

        raise RuntimeError(f"Failed to produce a valid move after {max_retries + 1} attempts")

    yield FunctionInfo.from_fn(
        _choose_move,
        description="Generate a single Tic-Tac-Toe move for the given board state and player symbol.",
    )
