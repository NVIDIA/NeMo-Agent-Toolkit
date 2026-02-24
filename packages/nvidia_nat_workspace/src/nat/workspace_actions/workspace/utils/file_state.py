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

"""Session-scoped file read-state tracking for read-before-edit validation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileReadState:
    """State recorded when a file is read."""

    content: str
    timestamp: float


_session_states: dict[str, dict[str, FileReadState]] = {}


def _to_abs_path(file_path: str) -> str:
    return str(Path(file_path).resolve())


def register_file_read(session_id: str, file_path: str, content: str) -> None:
    """Record that a file was read, including its content and timestamp."""
    states = _session_states.setdefault(session_id, {})
    states[_to_abs_path(file_path)] = FileReadState(
        content=content,
        timestamp=time.time(),
    )


def get_file_read_state(session_id: str, file_path: str) -> FileReadState | None:
    """Get previously recorded read state for a file path."""
    states = _session_states.get(session_id, {})
    return states.get(_to_abs_path(file_path))


def clear_file_read_state(session_id: str, file_path: str) -> None:
    """Clear any read state associated with a file path."""
    states = _session_states.get(session_id)
    if states is not None:
        states.pop(_to_abs_path(file_path), None)


def clear_session_state(session_id: str) -> None:
    """Clear all file read state for a session."""
    _session_states.pop(session_id, None)
