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
Node constraints for graph optimization.

Provides decorators and configuration to control how nodes are optimized:

- ``@sequential`` — force a node to never be parallelized.
- ``@depends_on`` — declare explicit dependencies on other nodes.
- ``@has_side_effects`` — mark a node as having side effects (warning-only).
- ``OptimizationConfig`` — configuration-based overrides (for third-party code).
"""

from nat_app.constraints.decorators import depends_on
from nat_app.constraints.decorators import has_side_effects
from nat_app.constraints.decorators import sequential
from nat_app.constraints.models import NodeConstraints
from nat_app.constraints.models import OptimizationConfig
from nat_app.constraints.models import ResolvedConstraints
from nat_app.constraints.resolution import apply_constraints_to_analysis
from nat_app.constraints.resolution import get_constraints
from nat_app.constraints.resolution import merge_dependencies
from nat_app.constraints.resolution import resolve_constraints

__all__ = [
    "apply_constraints_to_analysis",
    "depends_on",
    "get_constraints",
    "has_side_effects",
    "merge_dependencies",
    "NodeConstraints",
    "OptimizationConfig",
    "resolve_constraints",
    "ResolvedConstraints",
    "sequential",
]
