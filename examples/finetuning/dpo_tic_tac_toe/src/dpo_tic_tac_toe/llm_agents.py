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
LLM agent utilities for Tic-Tac-Toe.

This module provides XML parsing, random move generation, and LangChain chain
construction for LLM-based Tic-Tac-Toe players. The actual choose_move logic
is moved to a separate NAT Function (choose_move_function.py) to enable proper
TTC integration.
"""

import random
import re
from typing import Any

import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

# ---------- XML move parsing ----------

XML_ROW_REGEX = re.compile(r"<row>\s*([1-3])\s*</row>", re.IGNORECASE)
XML_COL_REGEX = re.compile(r"<col>\s*([1-3])\s*</col>", re.IGNORECASE)


def parse_move_xml(text: str) -> tuple[int, int] | None:
    """
    Parse move from XML:

        <move>
          <row>1</row>
          <col>3</col>
        </move>

    Returns 0-based (row, col).
    """
    row_match = XML_ROW_REGEX.search(text)
    col_match = XML_COL_REGEX.search(text)
    if not row_match or not col_match:
        return None
    row = int(row_match.group(1)) - 1
    col = int(col_match.group(1)) - 1
    if not (0 <= row < 3 and 0 <= col < 3):
        return None
    return row, col


def parse_move_any(text: str) -> tuple[int, int] | None:
    """Try XML parsing for move extraction."""
    mv = parse_move_xml(text)
    return mv


# ---------- Random move generation ----------


def make_random_move(board: np.ndarray) -> tuple[int, int, str]:
    """
    Generate a random legal move with a proper XML raw_response.

    This is used for random opponents when no LLM is specified. The raw_response
    is formatted consistently with LLM responses for proper history tracking.

    Args:
        board: 3x3 numpy array board state (0=empty, 1=X, -1=O)

    Returns:
        Tuple of (row, col, raw_response) where row/col are 0-based indices
        and raw_response is the XML-formatted move string.

    Raises:
        RuntimeError: If no legal moves are available.
    """
    # Find all empty positions
    legal_moves: list[tuple[int, int]] = []
    for r in range(3):
        for c in range(3):
            if board[r, c] == 0:
                legal_moves.append((r, c))

    if not legal_moves:
        raise RuntimeError("No available moves; game should be over.")

    # Pick a random move
    row, col = random.choice(legal_moves)

    # Generate XML response consistent with LLM format
    raw_response = f"<move>\n  <row>{row + 1}</row>\n  <col>{col + 1}</col>\n</move>"

    return row, col, raw_response


# ---------- Prompt construction ----------

SYSTEM_TEMPLATE = """
You are an expert Tic-Tac-Toe player.

You are playing as '{symbol}' on a 3x3 board.

Rules:
- The board uses 'X' and 'O' markers.
- The goal is to get 3 of your marks in a row, column, or diagonal.
- You must choose ONLY among the available empty positions.
- Rows and columns are numbered 1 to 3.
- Illegal moves (placing on an occupied square or out of range) are forbidden.

You MUST respond ONLY with a single XML snippet of this exact shape:

<move>
  <row>R</row>
  <col>C</col>
</move>

Where R and C are integers in [1, 3].

No explanation, no comments, no markdown, nothing else besides that XML.
"""


def get_system_prompt(player_symbol: str) -> str:
    """Get the formatted system prompt for a player symbol."""
    return SYSTEM_TEMPLATE.format(symbol=player_symbol)


def format_prompt_for_dpo(
    player_symbol: str,
    messages: list,
) -> str:
    """
    Format the full prompt as a string for DPO training.

    Returns the prompt as a simple string with each message on its own line:
        system: <content>
        user: <content>
        assistant: <content>
        ...

    Args:
        player_symbol: The player symbol ('X' or 'O')
        messages: List of LangChain messages (HumanMessage, AIMessage)

    Returns:
        Formatted prompt string
    """
    from langchain_core.messages import AIMessage
    from langchain_core.messages import HumanMessage

    lines = [f"system: {get_system_prompt(player_symbol)}"]

    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"user: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"assistant: {msg.content}")

    return "\n".join(lines)


def build_player_chain(model, player_symbol: str) -> Any:
    """
    Build a LangChain Runnable for a Tic-Tac-Toe player:
      (prompt -> model -> StrOutputParser)
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE),
        MessagesPlaceholder(variable_name="messages"),
    ]).partial(symbol=player_symbol)
    parser = StrOutputParser()
    chain = prompt | model | parser
    return chain
