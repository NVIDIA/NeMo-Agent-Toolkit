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
LLM agent utilities for Tic-Tac-Toe.

This module provides XML parsing and LangChain chain construction for LLM-based
Tic-Tac-Toe players. The actual choose_move logic is moved to a separate NAT
Function (choose_move_function.py) to enable proper TTC integration.
"""

import re
from typing import Any

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
