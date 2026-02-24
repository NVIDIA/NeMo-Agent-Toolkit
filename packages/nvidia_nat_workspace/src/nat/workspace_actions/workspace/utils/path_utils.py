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

"""Shared path resolution for workspace actions."""

from __future__ import annotations

from pathlib import Path


def resolve_workspace_path(file_path: str, root_path: Path) -> Path:
    """Resolve a file path relative to the workspace root."""
    path = Path(file_path)
    if not path.is_absolute():
        path = root_path / path
    return path.resolve()


def validate_within_root(resolved: Path, root_path: Path) -> None:
    """Raise ValueError if resolved path escapes the workspace root."""
    resolved_root = root_path.resolve()
    if resolved_root != resolved and resolved_root not in resolved.parents:
        raise ValueError(f"Path escapes workspace root: {resolved}")
