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
Result type dispatch for graph node execution.

Determines whether a node's return value should be merged into state,
based on its type (dict, list, None, callable, framework-specific command, etc.).

The command-object check is pluggable so framework packages can inject
their own detection logic.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class ResultHandler:
    """Handles different result types from node execution.

    Uses a dispatch pattern instead of complex nested conditionals.
    The ``command_checker`` is pluggable so framework packages can
    inject their own command-object detection.
    """

    def __init__(self, command_checker: Callable[[Any], bool] | None = None) -> None:
        """Initialize the result handler.

        Args:
            command_checker: Optional predicate that returns True for
                framework-specific command objects.
        """
        self._is_command = command_checker or (lambda _: False)

    def should_merge(self, result: Any) -> tuple[bool, str]:
        """Determine if a result should be merged into state.

        Args:
            result: The value returned by a node execution.

        Returns:
            Tuple of (should_merge, result_type_description).
        """
        if result is None:
            return False, "None"

        if callable(result) and not isinstance(result, (dict, list)):
            return False, f"callable:{type(result).__name__}"

        if isinstance(result, dict):
            return True, "dict"

        if isinstance(result, list):
            return True, "list"

        if self._is_command(result):
            return True, f"command:{type(result).__name__}"

        return False, f"unknown:{type(result).__name__}"

    def log_result(self, node_name: str, result: Any, should_merge: bool, type_desc: str) -> None:
        """Log a node result at the appropriate level.

        Args:
            node_name: Name of the node that produced the result.
            result: The raw result value.
            should_merge: Whether the result will be merged into state.
            type_desc: Short description of the result type.
        """
        if type_desc == "None":
            logger.debug("Node '%s' returned None (no state update)", node_name)
        elif type_desc.startswith("callable"):
            logger.warning("Node '%s' returned %s. Skipping state merge.", node_name, type_desc)
        elif type_desc == "dict":
            keys = list(result.keys()) if result else []
            logger.debug("Node '%s' returned dict with keys: %s", node_name, keys)
        elif type_desc == "list":
            logger.debug("Node '%s' returned list with %s updates", node_name, len(result))
        elif type_desc.startswith("command:"):
            logger.debug("Node '%s' returned framework command (%s)", node_name, type_desc)
        else:
            logger.warning("Node '%s' returned unexpected type '%s'. Skipping.", node_name, type_desc)
