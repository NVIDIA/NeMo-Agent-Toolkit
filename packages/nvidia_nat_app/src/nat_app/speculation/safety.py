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
Speculation safety system and router description.

Framework-agnostic primitives for controlling speculative execution:

- ``@speculation_unsafe`` decorator -- marks nodes as unsafe for speculation.
- ``SpeculationSafetyConfig`` -- per-node safe/unsafe overrides.
- ``RouterDescriptor`` -- framework-agnostic description of a router node.

No framework imports -- uses only Python stdlib.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import TypeVar

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def speculation_unsafe(cls_or_func: T) -> T:
    """Mark a node or middleware as unsafe for speculative execution.

    Use this when a node:
    - Modifies tool_calls (e.g., filtering, blocking)
    - Blocks for human input or external approval
    - Redacts/transforms content that downstream nodes depend on

    Args:
        cls_or_func: Class or function to mark as speculation-unsafe.

    Returns:
        The original class or function, annotated with the unsafe marker.

    Example:

        @speculation_unsafe
        class HumanApprovalMiddleware:
            def after_model(self, state, runtime):
                ...

        @speculation_unsafe
        def my_blocking_node(state):
            ...
    """
    cls_or_func._speculation_unsafe = True  # type: ignore[attr-defined]
    return cls_or_func


def is_marked_speculation_unsafe(obj: Any) -> bool:
    """Check if an object has been marked as speculation-unsafe via decorator.

    Args:
        obj: Object to inspect for the speculation-unsafe marker.

    Returns:
        ``True`` if the object was decorated with ``@speculation_unsafe``.
    """
    return getattr(obj, "_speculation_unsafe", False)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SpeculationSafetyConfig:
    """
    Configuration for speculation safety checks.

    Use ``unsafe_nodes`` to opt out specific nodes with side effects.
    Use ``safe_overrides`` to force-enable nodes on the built-in unsafe list.
    """

    unsafe_nodes: set[str] = field(default_factory=set)
    """Nodes that should block speculation (side effects, human-in-the-loop, etc.)."""

    safe_overrides: set[str] = field(default_factory=set)
    """Force-enable speculation for specific nodes (overrides unsafe_nodes and decorators)."""


# ---------------------------------------------------------------------------
# Router descriptor
# ---------------------------------------------------------------------------


@dataclass
class RouterDescriptor:
    """Framework-agnostic description of a router for speculative execution.

    Bridges compile-time router detection (``topology.RouterInfo``) and
    executor-level speculation.  The ``decision_fn`` is optional because
    not all frameworks expose an explicit decision function:

    - **Agno**: ``step.router_fn``
    - **CrewAI**: inferred from Flow return values (no explicit function)
    - **LangGraph**: conditional edges evaluated internally
    """

    name: str
    """Router node name."""

    possible_targets: list[str]
    """Names of all nodes this router can route to."""

    decision_fn: Callable[[dict[str, Any]], str] | None = None
    """Optional callable ``(state) -> chosen_target_name``."""
