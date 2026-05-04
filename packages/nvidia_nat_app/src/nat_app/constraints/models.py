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
"""Constraint data models."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field


@dataclass
class NodeConstraints:
    """Constraints for a single node (stored on the function object)."""

    name: str
    force_sequential: bool = False
    depends_on: set[str] = field(default_factory=set)
    reason: str | None = None
    has_side_effects: bool = False


@dataclass
class ResolvedConstraints:
    """Combined constraints from all sources for a node."""

    name: str
    force_sequential: bool = False
    explicit_dependencies: set[str] = field(default_factory=set)
    has_side_effects: bool = False
    reasons: list[str] = field(default_factory=list)
    source: str = "analysis"


@dataclass
class OptimizationConfig:
    """
    Configuration for graph optimization.

    Allows overriding constraints without modifying node code.
    Useful for third-party graphs or when decorators aren't practical.
    """

    force_sequential: set[str] = field(default_factory=set)
    explicit_dependencies: dict[str, set[str]] = field(default_factory=dict)
    side_effect_nodes: set[str] = field(default_factory=set)

    side_effect_keywords: set[str] = field(
        default_factory=lambda: {
            "write",
            "save",
            "update",
            "delete",
            "remove",
            "send",
            "email",
            "notify",
            "publish",
            "insert",
            "create",
            "modify",
            "mutate",
            "payment",
            "charge",
            "transfer",
            "execute",
            "run",
            "trigger", })

    disable_parallelization: bool = False
    trust_analysis: bool = False

    max_recursion_depth: int = 5
    """Max call depth for AST analysis when following callees. Default 5."""

    @classmethod
    def conservative(cls) -> OptimizationConfig:
        """Create a conservative config that disables parallelization.

        Returns:
            Config with ``disable_parallelization=True``.
        """
        return cls(disable_parallelization=True)

    @classmethod
    def aggressive(cls) -> OptimizationConfig:
        """Create an aggressive config that trusts analysis fully.

        Returns:
            Config with ``trust_analysis=True`` and no side-effect keywords.
        """
        return cls(trust_analysis=True, side_effect_keywords=set())
