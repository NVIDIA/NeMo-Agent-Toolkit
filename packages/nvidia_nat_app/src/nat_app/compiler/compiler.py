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
Generic compiler abstraction for agent optimization.

Provides a framework-agnostic base that compiles any source artifact into any
compiled artifact. Not limited to graphs -- subclass ``AbstractCompiler``
for tool-calling agents, reasoning pipelines, or any other agent topology.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any
from typing import Generic
from typing import TypeVar

_SourceArtifactType = TypeVar("_SourceArtifactType")
_CompiledArtifactType = TypeVar("_CompiledArtifactType")


class UnsupportedSourceError(ValueError):
    """
    Raised when a compiler does not support the given source artifact.

    Use this in validate() to provide a reason instead of returning False,
    so callers can report why compilation was rejected.
    """


class AbstractCompiler(ABC, Generic[_SourceArtifactType, _CompiledArtifactType]):
    """
    Abstract base for framework-specific compilers.

    Subclasses implement compile() to turn a source artifact into an optimized
    executable for that framework. Optional hooks: validate() before compile,
    export() for persistence.
    """

    @abstractmethod
    def compile(self, source: _SourceArtifactType, **kwargs: Any) -> _CompiledArtifactType:
        """Compile a source artifact into an optimized compiled artifact.

        Args:
            source: The source artifact to compile.

        Returns:
            The compiled and optimized artifact.
        """
        ...

    def validate(self, source: _SourceArtifactType) -> bool:
        """Return whether this compiler can compile the given source artifact.

        Override to add checks (schema, node types, framework, etc.). Default: True.
        To give a reason when unsupported, raise UnsupportedSourceError(reason)
        instead of returning False.

        Args:
            source: The source artifact to validate.

        Returns:
            True if this compiler supports the given source.
        """
        return True

    def export(self, compiled: _CompiledArtifactType, path: str | Path, **kwargs: Any) -> None:
        """Persist the compiled artifact to disk (e.g. for deployment).

        Override to implement serialization. Default: raises NotImplementedError.

        Args:
            compiled: The compiled artifact to persist.
            path: Filesystem path to write to.
        """
        raise NotImplementedError("export is not implemented for this compiler")


def compile_with(
    source: _SourceArtifactType,
    compiler: AbstractCompiler[_SourceArtifactType, _CompiledArtifactType],
    **kwargs: Any,
) -> _CompiledArtifactType:
    """Compile a source artifact using the given compiler (validate then compile).

    Args:
        source: The source artifact to compile.
        compiler: The compiler instance to use.

    Returns:
        The compiled artifact.

    Raises:
        UnsupportedSourceError: If compiler.validate(source) is False or
            the compiler raised UnsupportedSourceError with a reason.
    """
    if not compiler.validate(source):
        raise UnsupportedSourceError("Compiler does not support this source")
    return compiler.compile(source, **kwargs)
