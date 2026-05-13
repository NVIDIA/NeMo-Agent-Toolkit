# SPDX-FileCopyrightText: Copyright (c) 2025, Harbor Framework Contributors (https://github.com/harbor-framework/harbor)
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
"""Subagent trajectory reference model for ATIF trajectories."""

from __future__ import annotations

from typing import Any
from typing import Self

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator


class SubagentTrajectoryRef(BaseModel):
    """Reference to a delegated subagent trajectory (ATIF v1.7).

    A ref MUST be resolvable via at least one of two mechanisms:

    - **Embedded form** — ``trajectory_id`` matches the ``trajectory_id`` of
      an entry in the parent ``Trajectory.subagent_trajectories`` array.
    - **File-ref form** — ``trajectory_path`` references an external file
      (path, S3 URL, database identifier, etc.).

    A ref MUST set at least one of ``trajectory_id`` or ``trajectory_path``;
    setting both is permitted (an embedded ref MAY also record the original
    file path for debug). ``session_id`` is run-scoped and informational
    only — it is NOT a valid resolution key (two sibling subagents MAY
    legitimately share a ``session_id``).

    **Breaking vs. v1.6:** in v1.6 ``session_id`` was required on the ref
    and served as the resolution key. Under v1.7 a ref of the shape
    ``{"session_id": "..."}`` (no ``trajectory_id`` and no
    ``trajectory_path``) no longer validates. Producers MUST migrate by
    setting ``trajectory_id`` for embedded refs or ``trajectory_path`` for
    external-file refs. Pre-v1.7 refs that already set ``trajectory_path``
    remain valid.
    """

    trajectory_id: str | None = Field(
        default=None,
        description=("Canonical identifier of the delegated subagent trajectory, "
                     "used to resolve embedded references. Matches "
                     "``Trajectory.trajectory_id`` of an entry in the parent's "
                     "``subagent_trajectories`` array. At least one of "
                     "``trajectory_id`` or ``trajectory_path`` MUST be set so the "
                     "ref is resolvable. Added in ATIF v1.7."),
    )
    trajectory_path: str | None = Field(
        default=None,
        description=("Location of the complete subagent trajectory as an external "
                     "file (file path, S3 URL, database reference, etc.), used to "
                     "resolve file-ref references. At least one of ``trajectory_id`` "
                     "or ``trajectory_path`` MUST be set so the ref is resolvable."),
    )
    session_id: str | None = Field(
        default=None,
        description=("Run identity of the delegated subagent trajectory. "
                     "**Informational only** — recorded so consumers can correlate "
                     "this ref back to the subagent's run for debug / search / "
                     "display purposes. Run-scoped (see ``Trajectory.session_id``) "
                     "and therefore NOT a valid resolution key; consumers MUST NOT "
                     "use ``session_id`` alone to resolve a ref. Required in v1.6 "
                     "and earlier; relaxed to Optional + informational in v1.7."),
    )
    extra: dict[str, Any] | None = Field(
        default=None,
        description="Custom metadata about the subagent execution",
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_resolution_key_present(self) -> Self:
        # Spec §SubagentTrajectoryRefSchema: a ref MUST set at least one of
        # trajectory_id (embedded form) or trajectory_path (file-ref form).
        # session_id alone is insufficient — it's informational.
        if self.trajectory_id is None and self.trajectory_path is None:
            raise ValueError(
                "SubagentTrajectoryRef MUST set at least one of "
                "`trajectory_id` (embedded form) or `trajectory_path` "
                "(file-ref form); `session_id` alone is informational and "
                "not a valid resolution key (ATIF v1.7).", )
        return self
