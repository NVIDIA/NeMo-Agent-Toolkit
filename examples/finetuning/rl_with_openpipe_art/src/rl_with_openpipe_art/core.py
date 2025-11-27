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

import numpy as np

# ---------- Board / game primitives ----------

# Board encoding:
#   0 -> empty
#   1 -> 'X'
#  -1 -> 'O'

BOARD_SHAPE = (3, 3)

# Precompute all 8 lines (3 rows, 3 cols, 2 diags) for vectorized scoring
LINE_INDICES = np.array(
    [
        # rows
        [[0, 0], [0, 1], [0, 2]],
        [[1, 0], [1, 1], [1, 2]],
        [[2, 0], [2, 1], [2, 2]],
        # cols
        [[0, 0], [1, 0], [2, 0]],
        [[0, 1], [1, 1], [2, 1]],
        [[0, 2], [1, 2], [2, 2]],
        # diagonals
        [[0, 0], [1, 1], [2, 2]],
        [[0, 2], [1, 1], [2, 0]],
    ],
    dtype=int,
)


def new_board() -> np.ndarray:
    return np.zeros(BOARD_SHAPE, dtype=int)


def board_to_str(board: np.ndarray) -> str:
    """Pretty ASCII board for prompts / logging."""
    mapping = {1: "X", -1: "O", 0: "."}
    rows = [" ".join(mapping[int(x)] for x in row) for row in board]
    return "\n".join(rows)


def available_moves(board: np.ndarray) -> list[tuple[int, int]]:
    """Return list of available (row, col) indices (0-based)."""
    empties = np.argwhere(board == 0)
    return [tuple(map(int, idx)) for idx in empties]


def check_winner(board: np.ndarray) -> int:
    """
    Return:
      1  -> X wins
     -1  -> O wins
      0  -> no winner yet
    """
    # Rows and columns
    for i in range(3):
        row_sum = int(board[i, :].sum())
        if row_sum == 3:
            return 1
        if row_sum == -3:
            return -1

        col_sum = int(board[:, i].sum())
        if col_sum == 3:
            return 1
        if col_sum == -3:
            return -1

    # Diagonals
    diag1 = int(np.trace(board))
    if diag1 == 3:
        return 1
    if diag1 == -3:
        return -1

    diag2 = int(np.fliplr(board).trace())
    if diag2 == 3:
        return 1
    if diag2 == -3:
        return -1

    return 0


def is_draw(board: np.ndarray) -> bool:
    return (board == 0).sum() == 0 and check_winner(board) == 0
