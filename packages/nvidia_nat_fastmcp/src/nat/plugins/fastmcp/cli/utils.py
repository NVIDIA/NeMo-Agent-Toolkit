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
"""CLI helper utilities for FastMCP commands."""

from __future__ import annotations

from fnmatch import fnmatch
from collections.abc import Iterable
from collections.abc import Iterator
from pathlib import Path

from watchfiles import Change
from watchfiles import watch

DEFAULT_RELOAD_EXCLUDE_GLOBS: tuple[str, ...] = (
    "*.log",
    "*.tmp",
    "*.temp",
    "*.swp",
    "*.pyc",
    "*.pyo",
    "*__pycache__",
    "*__pycache__/*",
)


def _glob_matches(path: str, pattern: str) -> bool:
    normalized_path = path.replace("\\", "/")
    normalized_pattern = pattern.replace("\\", "/")
    return fnmatch(normalized_path, normalized_pattern) or fnmatch(Path(normalized_path).name, normalized_pattern)


def _filter_change_set(
    changes: set[tuple[Change, str]],
    include_globs: tuple[str, ...],
    exclude_globs: tuple[str, ...],
) -> set[tuple[Change, str]]:
    filtered_changes: set[tuple[Change, str]] = set()
    for change_type, changed_path in changes:
        if include_globs and not any(_glob_matches(changed_path, pattern) for pattern in include_globs):
            continue
        if exclude_globs and any(_glob_matches(changed_path, pattern) for pattern in exclude_globs):
            continue
        filtered_changes.add((change_type, changed_path))
    return filtered_changes


def iter_file_changes(
    paths: Iterable[Path],
    debounce_ms: int = 750,
    include_globs: Iterable[str] = (),
    exclude_globs: Iterable[str] = (),
) -> Iterator[set[tuple[Change, str]]]:
    """Yield filtered file change sets using watchfiles with debounce."""
    watch_paths = [str(path) for path in paths]
    include_patterns = tuple(pattern.strip() for pattern in include_globs if pattern.strip())
    exclude_patterns = DEFAULT_RELOAD_EXCLUDE_GLOBS + tuple(pattern.strip() for pattern in exclude_globs
                                                            if pattern.strip())
    for changes in watch(*watch_paths, debounce=debounce_ms):
        filtered_changes = _filter_change_set(changes, include_patterns, exclude_patterns)
        if filtered_changes:
            yield filtered_changes
