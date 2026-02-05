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
from pathlib import Path

from watchfiles import Change
from watchfiles import watch


def iter_file_changes(
    paths: Iterable[Path],
    debounce_ms: int = 750,
) -> Iterator[set[tuple[Change, str]]]:
    """Yield file change sets using watchfiles with debounce."""
    watch_paths = [str(path) for path in paths]
    return watch(*watch_paths, debounce=debounce_ms)
