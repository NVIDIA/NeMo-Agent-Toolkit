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
AbstractPipelinedCompiler: compiler that chains CompilationStage instances.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any
from typing import TypeVar

from nat_app.compiler.compilation_context import CompilationContext
from nat_app.compiler.compilation_stage import CompilationStage
from nat_app.compiler.compiler import AbstractCompiler

_SourceArtifactType = TypeVar("_SourceArtifactType")
_CompiledArtifactType = TypeVar("_CompiledArtifactType")

logger = logging.getLogger(__name__)


class AbstractPipelinedCompiler(
        AbstractCompiler[_SourceArtifactType, _CompiledArtifactType], ):
    """Compiler that runs an ordered sequence of CompilationStage instances.

    Each stage receives a ``CompilationContext`` containing the compiled
    artifact and a shared metadata dict for inter-stage communication.

    Subclasses must implement:
    - ``default_stages()`` — return the stages used when none are supplied.
    - ``prepare()``        — normalize the source into the compiled type.

    Optionally override:
    - ``finalize()``       — post-stage hook (e.g. apply an executor).
    - ``seed_context()``   — inject initial metadata before stages run.
    """

    def __init__(
        self,
        stages: Sequence[CompilationStage[_CompiledArtifactType]] | None = None,
    ) -> None:
        self._stages: tuple[CompilationStage[_CompiledArtifactType],
                            ...] = (tuple(stages) if stages is not None else tuple(self.default_stages()))
        self._last_context: CompilationContext[_CompiledArtifactType] | None = None

    @abstractmethod
    def default_stages(self) -> Sequence[CompilationStage[_CompiledArtifactType]]:
        """Return the default optimization stages for this compiler.

        Returns:
            Ordered sequence of stages that form the default pipeline.
        """
        ...

    @abstractmethod
    def prepare(
        self,
        source: _SourceArtifactType,
        **kwargs: Any,
    ) -> _CompiledArtifactType:
        """Normalize the source artifact into the compiled type.

        Called once before stages run.

        Args:
            source: The raw source artifact from the caller.

        Returns:
            The initial compiled artifact to seed the pipeline.
        """
        ...

    def seed_context(
        self,
        context: CompilationContext[_CompiledArtifactType],
    ) -> None:
        """Inject initial metadata into the context before stages run.

        Override to pre-populate ``context.metadata``.
        Default implementation is a no-op.

        Args:
            context: The freshly created context to seed.
        """

    def finalize(
        self,
        context: CompilationContext[_CompiledArtifactType],
        **kwargs: Any,
    ) -> _CompiledArtifactType:
        """Post-processing after all stages have run.

        Default implementation returns ``context.compiled`` unchanged.

        Args:
            context: The context after all stages have run.

        Returns:
            The final compiled artifact.
        """
        return context.compiled

    @property
    def stages(self) -> tuple[CompilationStage[_CompiledArtifactType], ...]:
        """The immutable sequence of optimization stages."""
        return self._stages

    @property
    def last_context(self) -> CompilationContext[_CompiledArtifactType] | None:
        """The context from the most recent compile() call, or None."""
        return self._last_context

    def compile(
        self,
        source: _SourceArtifactType,
        **kwargs: Any,
    ) -> _CompiledArtifactType:
        """Prepare source, run stages in order, then finalize.

        Args:
            source: The source artifact to compile.

        Returns:
            The finalized compiled artifact.
        """
        compiled = self.prepare(source, **kwargs)
        context = CompilationContext(compiled=compiled)
        self.seed_context(context)

        for stage in self._stages:
            logger.debug("Running compilation stage: %s", stage.name)
            context = stage.apply(context, **kwargs)

        self._last_context = context
        return self.finalize(context, **kwargs)
