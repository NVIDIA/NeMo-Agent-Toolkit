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
NVIDIA Agent Performance Primitives (NAT-APP).

Graph analysis, optimization, and scheduling algorithms for accelerating
agentic AI applications. Framework-agnostic -- zero external dependencies.

Quick start for framework teams:

    from nat_app import quick_optimize
    stages = quick_optimize(nodes={"a": fn_a, "b": fn_b}, edges=[("a", "b")])

See ``nat_app.api`` for the full embeddable API surface.
"""

import warnings as _warnings


class ExperimentalWarning(UserWarning):
    """Issued once when importing an experimental nat_app package."""


_warnings.warn(
    "The nvidia-nat-app package is experimental and the API may change in future releases. "
    "Future versions may introduce breaking changes without notice.",
    ExperimentalWarning,
    stacklevel=2,
)

# ruff: noqa: E402
from nat_app.api import SpeculationPlan
from nat_app.api import analyze_function
from nat_app.api import benchmark
from nat_app.api import classify_edge
from nat_app.api import find_parallel_stages
from nat_app.api import partition_targets
from nat_app.api import plan_speculation
from nat_app.api import quick_optimize
from nat_app.api import speculative_opportunities
from nat_app.executors.runner import SpeculativeResult
from nat_app.executors.runner import run_speculation
from nat_app.speculation.planner import SpeculationPlanner
from nat_app.speculation.resolution import Resolution
from nat_app.speculation.resolution import ResolutionPolicy
from nat_app.speculation.strategies import RouterBranchResolution
from nat_app.speculation.strategies import RouterBranchStrategy

__all__ = [
    "ExperimentalWarning",
    "Resolution",
    "ResolutionPolicy",
    "RouterBranchResolution",
    "RouterBranchStrategy",
    "SpeculationPlan",
    "SpeculationPlanner",
    "SpeculativeResult",
    "analyze_function",
    "benchmark",
    "classify_edge",
    "find_parallel_stages",
    "partition_targets",
    "plan_speculation",
    "quick_optimize",
    "run_speculation",
    "speculative_opportunities",
]
