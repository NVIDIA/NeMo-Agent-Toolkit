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

from collections.abc import Iterable
from collections.abc import Iterator
from fnmatch import fnmatch
from pathlib import Path

from watchfiles import Change
from watchfiles import watch

# `watchfiles.watch()` already uses `DefaultFilter`, which ignores common
# artifacts such as `__pycache__`, `*.pyc`, `*.pyo`, and `*.swp`.
# These are additional noisy patterns for dev workflows.
DEFAULT_RELOAD_EXCLUDE_GLOBS: tuple[str, ...] = (
    "*.log",
    "*.tmp",
    "*.temp",
)


def _glob_matches(path: str, pattern: str) -> bool:
    """Return True when a path matches a glob pattern.

    Matching is performed against both the normalized full path and basename
    so patterns like `*.py` work regardless of directory depth.
    """
    normalized_path = path.replace("\\", "/")
    normalized_pattern = pattern.replace("\\", "/")
    return fnmatch(normalized_path, normalized_pattern) or fnmatch(Path(normalized_path).name, normalized_pattern)


def _filter_change_set(
    changes: set[tuple[Change, str]],
    include_globs: tuple[str, ...],
    exclude_globs: tuple[str, ...],
) -> set[tuple[Change, str]]:
    """Filter change events using include and exclude glob rules."""
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
    """Yield filtered file change sets using watchfiles with debounce.

    :param paths: File or directory paths to watch for changes.
    :param debounce_ms: Debounce interval in milliseconds passed to `watchfiles`.
    :param include_globs: Optional include patterns. When provided, only matching
        paths trigger reload checks.
    :param exclude_globs: Optional exclude patterns. These are merged with
        `DEFAULT_RELOAD_EXCLUDE_GLOBS` only when include patterns are not provided.
    :returns: Iterator yielding sets of `(Change, path)` tuples that pass
        include/exclude filtering.
    """
    watch_paths = [str(path) for path in paths]
    include_patterns = tuple(pattern.strip() for pattern in include_globs if pattern.strip())
    user_exclude_patterns = tuple(pattern.strip() for pattern in exclude_globs if pattern.strip())
    if include_patterns:
        # Explicit include patterns should not be blocked by default excludes.
        exclude_patterns = user_exclude_patterns
    else:
        exclude_patterns = DEFAULT_RELOAD_EXCLUDE_GLOBS + user_exclude_patterns
    for changes in watch(*watch_paths, debounce=debounce_ms):
        filtered_changes = _filter_change_set(changes, include_patterns, exclude_patterns)
        if filtered_changes:
            yield filtered_changes
