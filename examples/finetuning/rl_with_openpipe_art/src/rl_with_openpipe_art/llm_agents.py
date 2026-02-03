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
import random
import re
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

from .core import available_moves
from .core import board_to_str

logger = logging.getLogger(__name__)

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
    """Try XML first, then legacy 'row,col'."""
    mv = parse_move_xml(text)
    return mv


# ---------- LLM player wrapper ----------


@dataclass
class LLMTicTacToePlayer:
    name: str
    symbol: str  # 'X' or 'O'
    value: int  # 1 for X, -1 for O
    chain: Any  # LangChain Runnable: prompt | model | StrOutputParser
    max_retries: int = 0
    choose_random: bool = False
    messages: list = field(default_factory=list)
    steps = 0

    def choose_move(self, board) -> tuple[int, int, str]:
        """
        Ask the LLM for a move and return (row, col, raw_response_text).

        - Tries up to `max_retries` times to parse a valid move.
        - If still invalid, falls back to a random legal move.
        """
        board_str = board_to_str(board)
        moves: list[tuple[int, int]] = available_moves(board)

        if self.choose_random:
            fallback_move = random.choice(moves)
            raw_response = f"<move>\n  <row>{fallback_move[0]+1}</row>\n  <col>{fallback_move[1]+1}</col>\n</move>"
            return fallback_move[0], fallback_move[1], raw_response

        if not moves:
            raise RuntimeError("No available moves; game should be over.")

        # ruff
        for attempt in range(0, self.max_retries + 1):
            # Provide all user and LLM messages + current board
            self.steps += 1

            if attempt > 0:
                self.messages.append(
                    HumanMessage(content=f"You made an invalid move. You have "
                                 f"{self.max_retries - attempt + 1} attempts left.\n"
                                 f"Available moves are: "
                                 f"{', '.join(f'({r+1},{c+1})' for r,c in moves)}\n. "
                                 f"Current board:\n{board_str}"))
            else:
                self.messages.append(HumanMessage(content=board_str))

            raw_response = self.chain.invoke({
                "messages": self.messages,
            })

            text = str(raw_response)

            move = parse_move_any(text)

            self.messages.append(AIMessage(content=text))

            if move is not None and move in moves:
                return move[0], move[1], text

            logger.debug(f"[WARN] {self.name} produced invalid move on attempt {attempt}: "
                         f"'{text}'. Retrying...")

        raise RuntimeError(f"{self.name} failed to produce a valid move")


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
