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
"""Node constraint decorators."""

from __future__ import annotations

from collections.abc import Callable

from nat_app.constraints.models import NodeConstraints


def _get_or_create_constraints(func: Callable) -> NodeConstraints:
    existing = getattr(func, "_optimization_constraints", None)
    return existing if existing is not None else NodeConstraints(name=func.__name__)


def sequential(reason: str | None = None) -> Callable:
    """Mark a node as requiring sequential execution.

    Use this for nodes with side effects that cannot be parallelized:
    database writes, external API calls with state, file system operations, etc.

    Args:
        reason: Human-readable explanation of why sequential ordering is needed.

    Returns:
        A decorator that marks the function as requiring sequential execution.

    Example:

        @sequential(reason="Writes to database")
        async def save_results(state):
            await db.insert(state["results"])
            return {"saved": True}
    """

    def decorator(func: Callable) -> Callable:
        constraints = _get_or_create_constraints(func)
        constraints.force_sequential = True
        constraints.has_side_effects = True
        constraints.reason = reason or "Marked as sequential"
        func._optimization_constraints = constraints
        return func

    return decorator


def depends_on(*node_names: str, reason: str | None = None) -> Callable:
    """Explicitly declare that this node depends on specific other nodes.

    Use when the dependency isn't visible in state (side effects)
    or you want to enforce ordering regardless of analysis.

    Args:
        *node_names: Names of nodes this node depends on.
        reason: Human-readable explanation for the dependency.

    Returns:
        A decorator that adds the dependency constraints to the function.

    Example:

        @depends_on("fetch_data", "validate_input", reason="Needs both complete")
        async def process(state):
            ...
    """

    def decorator(func: Callable) -> Callable:
        constraints = _get_or_create_constraints(func)
        constraints.depends_on.update(node_names)
        if reason:
            constraints.reason = reason
        func._optimization_constraints = constraints
        return func

    return decorator


def has_side_effects(reason: str | None = None) -> Callable:
    """Mark a node as having side effects (but potentially parallelizable).

    Different from @sequential:
    - @sequential = MUST be sequential, never parallelize
    - @has_side_effects = Has side effects, be careful, warn user

    Args:
        reason: Human-readable description of the side effect.

    Returns:
        A decorator that marks the function as having side effects.

    Example:

        @has_side_effects(reason="Sends HTTP request to external API")
        async def call_external_api(state):
            response = await http.post(...)
            return {"response": response}
    """

    def decorator(func: Callable) -> Callable:
        constraints = _get_or_create_constraints(func)
        constraints.has_side_effects = True
        constraints.reason = reason or "Has side effects"
        func._optimization_constraints = constraints
        return func

    return decorator
