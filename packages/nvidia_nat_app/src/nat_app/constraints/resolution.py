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
"""Constraint resolution: combine decorators, config, and heuristics."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from nat_app.constraints.models import NodeConstraints
from nat_app.constraints.models import OptimizationConfig
from nat_app.constraints.models import ResolvedConstraints

if TYPE_CHECKING:
    from nat_app.graph.analysis import NodeAnalysis


def get_constraints(func: Callable) -> NodeConstraints | None:
    """Get constraints registered for a function (from its decorators).

    Args:
        func: The decorated function to inspect.

    Returns:
        The ``NodeConstraints`` attached by decorators, or None.
    """
    return getattr(func, "_optimization_constraints", None)


def resolve_constraints(
    node_name: str,
    node_func: Callable | None,
    config: OptimizationConfig,
) -> ResolvedConstraints:
    """Resolve constraints for a node from all sources.

    Priority order:
    1. Decorators (highest -- developer explicitly marked)
    2. Config (explicit overrides)
    3. Heuristics (keyword-based detection)

    Args:
        node_name: The graph node name.
        node_func: The callable for the node, or None if unavailable.
        config: Optimization configuration with overrides and keywords.

    Returns:
        Combined constraints from all applicable sources.
    """
    result = ResolvedConstraints(name=node_name)

    if node_func is not None:
        decorator_constraints = get_constraints(node_func)
        if decorator_constraints:
            result.source = "decorator"
            if decorator_constraints.force_sequential:
                result.force_sequential = True
                result.reasons.append(f"@sequential: {decorator_constraints.reason}")
            if decorator_constraints.depends_on:
                result.explicit_dependencies.update(decorator_constraints.depends_on)
                result.reasons.append(f"@depends_on: {decorator_constraints.depends_on}")
            if decorator_constraints.has_side_effects:
                result.has_side_effects = True

    if node_name in config.force_sequential:
        result.force_sequential = True
        result.reasons.append("Config: force_sequential")
        if result.source == "analysis":
            result.source = "config"

    if node_name in config.explicit_dependencies:
        result.explicit_dependencies.update(config.explicit_dependencies[node_name])
        result.reasons.append("Config: explicit_dependencies")
        if result.source == "analysis":
            result.source = "config"

    if node_name in config.side_effect_nodes:
        result.has_side_effects = True
        if result.source == "analysis":
            result.source = "config"

    if not config.trust_analysis and config.side_effect_keywords:
        name_lower = node_name.lower()
        for keyword in config.side_effect_keywords:
            if keyword in name_lower:
                result.has_side_effects = True
                result.reasons.append(f"Heuristic: contains '{keyword}'")
                if result.source == "analysis":
                    result.source = "heuristic"
                break

    if config.disable_parallelization:
        result.force_sequential = True
        result.reasons.append("Config: disable_parallelization=True")

    return result


def apply_constraints_to_analysis(
    node_analyses: dict[str, NodeAnalysis],
    node_funcs: dict[str, Callable],
    config: OptimizationConfig,
) -> tuple[dict[str, ResolvedConstraints], list[str]]:
    """Apply constraints to analysis results.

    Args:
        node_analyses: Per-node analysis results keyed by node name. Values are
            ``NodeAnalysis`` objects from static analysis.
        node_funcs: Mapping of node name to callable.
        config: Optimization configuration with constraint overrides.

    Returns:
        Tuple of (resolved constraints per node, list of warning messages).
    """
    constraints: dict[str, ResolvedConstraints] = {}
    warnings: list[str] = []

    for name in node_analyses:
        func = node_funcs.get(name)
        resolved = resolve_constraints(name, func, config)
        constraints[name] = resolved

        if resolved.has_side_effects and not resolved.force_sequential:
            warnings.append(f"Node '{name}' may have side effects ({resolved.source}). "
                            "Consider using @sequential if it must be ordered.")

    return constraints, warnings


def merge_dependencies(
    data_dependencies: dict[str, set[str]],
    constraints: dict[str, ResolvedConstraints],
) -> dict[str, set[str]]:
    """Merge automatic data dependencies with explicit constraint dependencies.

    Args:
        data_dependencies: Data-flow dependencies from static analysis.
        constraints: Resolved constraints containing explicit dependencies.

    Returns:
        Merged dependency mapping with both data and constraint edges.
    """
    merged = {name: deps.copy() for name, deps in data_dependencies.items()}

    for name, constraint in constraints.items():
        if name not in merged:
            merged[name] = set()
        merged[name].update(constraint.explicit_dependencies)

    return merged
