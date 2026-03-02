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
"""
CompilationStage protocol: the unit of work in a pipelined compiler.

Stages can perform extraction, validation, analysis, optimization,
or any other transformation on the compilation context.
"""

from __future__ import annotations

from typing import Any
from typing import Protocol
from typing import TypeVar
from typing import runtime_checkable

from nat_app.compiler.compilation_context import CompilationContext

_CompiledArtifactType = TypeVar("_CompiledArtifactType")


@runtime_checkable
class CompilationStage(Protocol[_CompiledArtifactType]):
    """A single step in a pipelined compiler.

    Each stage receives a CompilationContext (compiled artifact + shared
    metadata), applies its work, and returns the updated context.
    Stages can read metadata from previous stages and write their own.
    """

    @property
    def name(self) -> str:
        """Human-readable name for logging / identification."""
        ...

    def apply(
        self,
        context: CompilationContext[_CompiledArtifactType],
        **kwargs: Any,
    ) -> CompilationContext[_CompiledArtifactType]:
        """Apply this stage and return the updated context.

        Args:
            context: The current compilation context.

        Returns:
            The updated compilation context after this stage's work.
        """
        ...
