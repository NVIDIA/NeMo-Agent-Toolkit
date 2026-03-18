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
"""ValidateStage: validate the extracted Graph structure."""

from __future__ import annotations

from typing import Any

from nat_app.compiler.compilation_context import CompilationContext
from nat_app.compiler.errors import GraphValidationError


class ValidateStage:
    """Validate the Graph produced by the adapter.

    Reads: ``graph``
    Raises: ``GraphValidationError`` if issues are found.
    """

    @property
    def name(self) -> str:
        return "validate"

    def apply(self, context: CompilationContext, **kwargs: Any) -> CompilationContext:
        """Validate the graph structure and raise if issues are found.

        Args:
            context: Current compilation context with ``graph`` in metadata.
            **kwargs: Additional arguments (reserved for future use).

        Returns:
            The context unchanged if validation passes.

        Raises:
            GraphValidationError: If structural issues are found.
        """
        graph = context.metadata["graph"]
        issues = graph.validate()
        if issues:
            raise GraphValidationError(issues)
        return context
