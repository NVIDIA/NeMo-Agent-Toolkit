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


# ---------- Heuristic evaluation (Alpha-Beta-style evaluation function) ----------


def evaluate_board_for_player(board: np.ndarray, player_val: int) -> float:
    """
    Heuristic evaluation from the perspective of `player_val` (1 for X, -1 for O).

    Alpha-Beta-style evaluation, but used *post-facto* (no search), based on:
      - Terminal states (win/lose/draw)
      - Line control (open lines, two-in-a-row, blocks)
      - Center / corner / edge control
      - Forks (two simultaneous threats)
      - Immediate opponent threats

    Higher score = better for `player_val`.
    """

    assert player_val in (1, -1), "player_val must be 1 (X) or -1 (O)"

    winner = check_winner(board)
    if winner == player_val:
        return 1e6  # immediate win
    elif winner == -player_val:
        return -1e6  # immediate loss
    elif is_draw(board):
        return 0.0

    # Perspective transform: player_val pieces -> +1, opponent -> -1
    b = board * player_val

    # Extract all lines at once: shape (8, 3)
    line_vals = b[LINE_INDICES[..., 0], LINE_INDICES[..., 1]]

    player_counts = (line_vals == 1).sum(axis=1)
    opp_counts = (line_vals == -1).sum(axis=1)
    empty_counts = (line_vals == 0).sum(axis=1)

    # Offensive / defensive line scores
    line_scores = np.zeros(8, dtype=float)

    offensive_weights = np.array([0.0, 1.0, 4.0, 0.0])  # index = num_player_marks
    defensive_weights = np.array([0.0, 1.2, 6.0, 0.0])  # index = num_opp_marks

    pure_off = (opp_counts == 0) & (player_counts > 0)
    pure_def = (player_counts == 0) & (opp_counts > 0)

    line_scores[pure_off] += offensive_weights[player_counts[pure_off]]
    line_scores[pure_def] -= defensive_weights[opp_counts[pure_def]]

    # Fork potential: lines with 2 of ours + 1 empty
    fork_lines = (player_counts == 2) & (opp_counts == 0) & (empty_counts == 1)
    num_forks = int(fork_lines.sum())
    fork_score = 0.0
    if num_forks >= 2:
        fork_score += 8.0 * (num_forks - 1)

    # Center / corner / edge control
    center = int(b[1, 1])
    center_score = 3.0 * center

    corners = np.array([b[0, 0], b[0, 2], b[2, 0], b[2, 2]], dtype=int)
    corner_score = 1.5 * int(corners.sum())

    edges = np.array([b[0, 1], b[1, 0], b[1, 2], b[2, 1]], dtype=int)
    edge_score = 0.5 * int(edges.sum())

    # Opponent immediate threats: lines with 2 opponent marks + 1 empty
    opp_threats = (player_counts == 0) & (opp_counts == 2) & (empty_counts == 1)
    opp_threat_score = -7.0 * int(opp_threats.sum())

    total_score = (float(line_scores.sum()) + fork_score + center_score + corner_score + edge_score + opp_threat_score)

    return total_score
