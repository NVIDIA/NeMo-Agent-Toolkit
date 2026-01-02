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
    mapping = {1: "X", -1: "O", 0: "_"}
    # Return a string representation of tic tac toe board with rows and columns
    # Also add numbered rows and columns for easier reading, starting at index 1
    rows = []
    for i in range(3):
        row_str = " ".join(mapping[board[i, j]] for j in range(3))
        rows.append(f"{i + 1} {row_str}")
    header = "  1 2 3"
    return "\n".join([header] + rows)


def board_to_list(board: np.ndarray) -> list[list[int]]:
    """Convert board to nested list for JSON serialization."""
    return [[int(board[i, j]) for j in range(3)] for i in range(3)]


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


def evaluate_board_for_player(board: np.ndarray, player_val: int) -> float:
    """
    Evaluate the position from the perspective of `player_val` (1 for X, -1 for O).

    Output:
      - For *non-guaranteed* states (no forced win/loss under perfect play):
          value in [0, 1], continuous.
      - For states where `player_val` has a *forced future win* (but not yet won):
          value in (1, 11]  ≈  base ∈ [0,1]  + 10.
      - For states where `player_val` has an *immediate win* on the board:
          value in (1, 16]  ≈  base ∈ [0,1]  + 15.
      - For states where `player_val` has *already lost* or is in a
        *forced future loss*:
          value = 0.0

    This is suitable as a state-value / reward signal for RL.
    """

    assert player_val in (1, -1), "player_val must be 1 (X) or -1 (O)"

    # -------- persistent cache for solved outcomes (board, side_to_move) -> {-1, 0, 1} --------
    # This dictionary survives across function calls.
    outcome_cache: dict[tuple[tuple[int, ...], int], float] = getattr(evaluate_board_for_player, "_outcome_cache", None)
    if outcome_cache is None:
        outcome_cache = {}
        setattr(evaluate_board_for_player, "_outcome_cache", outcome_cache)

    # -------- continuous static evaluation in [-1, 1] (no search) --------

    def static_eval(b: np.ndarray) -> float:
        """
        Heuristic position evaluation from the perspective of `player_val`.
        Returns a continuous value in [-1, 1].
        """
        winner = check_winner(b)
        if winner == player_val:
            return 1.0
        elif winner == -player_val:
            return -1.0
        elif is_draw(b):
            return 0.0

        # Perspective transform: player_val pieces -> +1, opponent -> -1
        pb = b * player_val

        # All 8 lines (rows, cols, diags)
        line_vals = pb[LINE_INDICES[..., 0], LINE_INDICES[..., 1]]
        player_counts = (line_vals == 1).sum(axis=1)
        opp_counts = (line_vals == -1).sum(axis=1)

        my_two_open = np.sum((player_counts == 2) & (opp_counts == 0))
        opp_two_open = np.sum((player_counts == 0) & (opp_counts == 2))
        my_one_open = np.sum((player_counts == 1) & (opp_counts == 0))
        opp_one_open = np.sum((player_counts == 0) & (opp_counts == 1))

        center = int(pb[1, 1])
        corners = np.array(
            [pb[0, 0], pb[0, 2], pb[2, 0], pb[2, 2]],
            dtype=int,
        )
        edges = np.array(
            [pb[0, 1], pb[1, 0], pb[1, 2], pb[2, 1]],
            dtype=int,
        )

        # Hand-crafted features: "how good is this board for player_val?"
        score_raw = (
            4.0 * (my_two_open - opp_two_open)  # strong threats
            + 1.5 * (my_one_open - opp_one_open)  # influence / potential
            + 1.5 * center  # center control
            + 0.75 * int(corners.sum())  # corners
            + 0.25 * int(edges.sum())  # edges
        )

        # Squash to [-1, 1] for stability
        return float(np.tanh(score_raw / 5.0))

    # -------- alpha-beta outcome solver: forced win / loss / draw  --------

    def solve_outcome(b: np.ndarray, side_to_move: int, alpha: float = -1.0, beta: float = 1.0) -> float:
        """
        Game-theoretic outcome from the perspective of `player_val`:

          +1  -> `player_val` can force a win from this state
           0  -> perfect play leads to a draw
          -1  -> `player_val` will lose with best play from both sides

        Uses full-depth search with alpha-beta and memoization.
        """
        key = (tuple(int(x) for x in b.flatten()), int(side_to_move))
        if key in outcome_cache:
            return outcome_cache[key]

        winner = check_winner(b)
        if winner == player_val:
            v = 1.0
        elif winner == -player_val:
            v = -1.0
        elif is_draw(b):
            v = 0.0
        else:
            moves = available_moves(b)
            if not moves:
                # No moves, no winner (shouldn't really happen): treat as draw
                v = 0.0
            elif side_to_move == player_val:
                # Maximizing for player_val
                best = -1.0
                for r, c in moves:
                    b[r, c] = side_to_move
                    child_val = solve_outcome(b, -side_to_move, alpha, beta)
                    b[r, c] = 0

                    best = max(best, child_val)
                    alpha = max(alpha, best)
                    if alpha >= beta:
                        break  # beta cut-off

                v = best
            else:
                # Minimizing for opponent
                best = 1.0
                for r, c in moves:
                    b[r, c] = side_to_move
                    child_val = solve_outcome(b, -side_to_move, alpha, beta)
                    b[r, c] = 0

                    best = min(best, child_val)
                    beta = min(beta, best)
                    if alpha >= beta:
                        break  # alpha cut-off

                v = best

        outcome_cache[key] = v
        return v

    # -------- continuous base value in [0, 1] --------

    static_score = static_eval(board)  # in [-1, 1]
    base_value = 0.5 * (static_score + 1.0)  # map [-1, 1] -> [0, 1]
    base_value = float(min(1.0, max(0.0, base_value)))  # clamp numerically

    # -------- handle immediate terminal states first --------

    winner_now = check_winner(board)
    if winner_now == player_val:
        # Immediate win:  base in [0,1] plus 15-point bonus
        return base_value + 15.0
    elif winner_now == -player_val:
        # Already lost: reward is exactly 0
        return 0.0
    elif is_draw(board):
        # True draw state: not a forced win or loss, just neutral -> [0,1]
        return base_value

    # -------- non-terminal: check forced outcome under perfect play --------

    # Figure out whose turn it is: X always starts.
    x_count = int(np.count_nonzero(board == 1))
    o_count = int(np.count_nonzero(board == -1))
    side_to_move = 1 if x_count == o_count else -1

    outcome = solve_outcome(board.copy(), side_to_move)

    if outcome > 0.0:
        # Forced future win for player_val (but not already winning on board)
        # Base in [0,1], plus +10 bonus to make it strictly > 1.
        return base_value + 10.0
    elif outcome < 0.0:
        # Forced future loss for player_val
        return 0.0
    else:
        # Game-theoretic draw (with perfect play) -> use smooth base in [0,1].
        return base_value
