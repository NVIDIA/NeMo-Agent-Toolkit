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

"""Backward-compatible local mode policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nat_harbor.environments.local import LocalEnvironment


def is_shell_profile_write(command: str) -> bool:
    """Detect writes targeting shell profile files."""
    return LocalEnvironment.is_shell_profile_write(command)


@dataclass(frozen=True)
class LocalWritePolicy:
    """Allowed write roots for local-mode operations."""

    allowed_write_roots: tuple[Path, ...]

    @staticmethod
    def is_within(path: Path, root: Path) -> bool:
        """Return True if path is under root."""
        try:
            path.resolve().relative_to(root.resolve())
            return True
        except ValueError:
            return False

    def assert_allowed_write_path(self, path: Path, operation: str) -> None:
        """Raise PermissionError when write target is outside allowed roots."""
        resolved = path.resolve()
        if any(self.is_within(resolved, root) for root in self.allowed_write_roots):
            return
        roots = ", ".join(str(root) for root in self.allowed_write_roots)
        raise PermissionError(
            f"Local mode policy violation during {operation}: write path '{resolved}' "
            f"is outside allowed roots [{roots}]"
        )


__all__ = ["LocalWritePolicy", "is_shell_profile_write"]

