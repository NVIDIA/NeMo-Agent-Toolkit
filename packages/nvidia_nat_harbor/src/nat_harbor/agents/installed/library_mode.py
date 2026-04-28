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
"""Inline library-mode interface contracts for Nemo agent execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Protocol

from pydantic import BaseModel
from pydantic import ConfigDict


class NemoInlineRunnerInput(BaseModel):
    """Inputs required for phase-1 inline Nemo workflow execution."""
    model_config = ConfigDict(frozen=True)

    instruction: str
    config_file: str
    artifact_dir: Path
    env: dict[str, str]


class NemoInlineRunnerResult(BaseModel):
    """Outputs produced by a phase-1 inline Nemo workflow execution."""
    model_config = ConfigDict(frozen=True)

    output_text: str
    trajectory_path: Path
    steps_count: int
    runner_details: dict[str, Any]


class NemoInlineRunner(Protocol):
    """Protocol for inline Nemo execution used by library mode."""

    async def run(self, request: NemoInlineRunnerInput) -> NemoInlineRunnerResult:
        """Execute a workflow in-process and return inline-runner outputs."""
