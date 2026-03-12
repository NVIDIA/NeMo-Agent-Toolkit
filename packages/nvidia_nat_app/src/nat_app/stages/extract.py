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
"""ExtractStage: extract a Graph from the source via the adapter."""

from __future__ import annotations

import logging
from typing import Any

from nat_app.compiler.compilation_context import CompilationContext
from nat_app.graph.adapter import AbstractFrameworkAdapter

logger = logging.getLogger(__name__)


class ExtractStage:
    """Extract an abstract Graph from the framework source artifact.

    Reads: ``context.compiled`` (the framework source)
    Writes: ``graph``, ``reducer_fields``, ``all_schema_fields``, ``state_schema``
    """

    def __init__(self, adapter: AbstractFrameworkAdapter) -> None:
        self._adapter = adapter

    @property
    def name(self) -> str:
        return "extract"

    def apply(self, context: CompilationContext, **kwargs: Any) -> CompilationContext:
        """Extract a Graph from the framework source and populate context metadata.

        Args:
            context: Current compilation context with ``compiled`` (the source).
            **kwargs: Additional arguments (reserved for future use).

        Returns:
            The updated context with ``graph``, ``reducer_fields``,
            ``all_schema_fields``, and ``state_schema`` in metadata.
        """
        graph = self._adapter.extract(context.compiled)
        logger.info("Extracted %d nodes, %d edges", graph.node_count, graph.edge_count)

        context.metadata["graph"] = graph
        context.metadata["reducer_fields"] = self._adapter.get_reducer_fields()
        context.metadata["all_schema_fields"] = self._adapter.get_all_schema_fields()
        context.metadata["state_schema"] = self._adapter.get_state_schema()

        reducer_fields = context.metadata["reducer_fields"]
        if reducer_fields:
            logger.info("Reducer fields (parallel-safe writes): %s", reducer_fields)

        return context
