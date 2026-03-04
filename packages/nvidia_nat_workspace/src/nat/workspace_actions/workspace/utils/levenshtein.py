# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Levenshtein distance utilities for fuzzy file-name matching."""

from __future__ import annotations

import os
from pathlib import Path


def levenshtein_distance(a: str, b: str) -> int:
    """Return the Levenshtein edit distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    matrix = [[0] * (len(a) + 1) for _ in range(len(b) + 1)]

    for i in range(len(b) + 1):
        matrix[i][0] = i
    for j in range(len(a) + 1):
        matrix[0][j] = j

    for i in range(1, len(b) + 1):
        for j in range(1, len(a) + 1):
            if b[i - 1] == a[j - 1]:
                matrix[i][j] = matrix[i - 1][j - 1]
            else:
                matrix[i][j] = min(
                    matrix[i - 1][j - 1] + 1,  # substitution
                    matrix[i][j - 1] + 1,  # insertion
                    matrix[i - 1][j] + 1,  # deletion
                )

    return matrix[len(b)][len(a)]


def find_similar_file(file_path: str) -> str | None:
    """Find the best similar file in the same directory.

    Strategy:
    1) Case-insensitive exact basename match.
    2) Fuzzy stem match with Levenshtein distance < 3.
    """
    path = Path(file_path)
    directory = path.parent
    basename = path.name.lower()

    try:
        files = os.listdir(directory)
    except OSError:
        return None

    for file_name in files:
        if file_name.lower() == basename and file_name != path.name:
            return str(directory / file_name)

    target_stem = path.stem.lower()
    best_match: str | None = None
    best_distance = float("inf")

    for file_name in files:
        distance = levenshtein_distance(target_stem, Path(file_name).stem.lower())
        if distance < 3 and distance < best_distance:
            best_distance = distance
            best_match = str(directory / file_name)

    return best_match


def find_similar_files(file_path: str, max_suggestions: int = 3) -> list[str]:
    """Find multiple similar file candidates sorted by distance."""
    if max_suggestions <= 0:
        return []

    path = Path(file_path)
    directory = path.parent
    target_stem = path.stem.lower()

    try:
        files = os.listdir(directory)
    except OSError:
        return []

    distances: list[tuple[int, str]] = []
    for file_name in files:
        distance = levenshtein_distance(target_stem, Path(file_name).stem.lower())
        if distance < 3:
            distances.append((distance, str(directory / file_name)))

    distances.sort(key=lambda item: item[0])
    return [path for _, path in distances[:max_suggestions]]
