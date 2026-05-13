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
Speculation planning, resolution, and safety primitives.

Planning:
    - ``SpeculationPlan`` -- concrete per-decision-node speculation plan.
    - ``plan_speculation`` -- produce plans from graph data + safety config.
    - ``partition_targets`` -- split targets into immediate vs. deferred.

Resolution:
    - ``Resolution`` -- outcome dataclass (keep, cancel, rerun).
    - ``ResolutionPolicy`` -- protocol for resolving speculation outcomes.

Strategies:
    - ``SpeculationStrategy`` -- protocol for pluggable strategies.
    - ``SpeculationPlanner`` -- composes multiple strategies.
    - ``RouterBranchStrategy`` -- full-branch router speculation.
    - ``RouterBranchResolution`` -- router-branch resolution policy.

Safety:
    - ``@speculation_unsafe`` -- marks nodes as unsafe for speculation.
    - ``is_marked_speculation_unsafe`` -- checks the decorator mark.
    - ``SpeculationSafetyConfig`` -- per-node safe/unsafe overrides.

Router description:
    - ``RouterDescriptor`` -- framework-agnostic router description.
"""

from nat_app.speculation.plan import SpeculationPlan
from nat_app.speculation.plan import partition_targets
from nat_app.speculation.plan import plan_speculation
from nat_app.speculation.planner import SpeculationPlanner
from nat_app.speculation.resolution import Resolution
from nat_app.speculation.resolution import ResolutionPolicy
from nat_app.speculation.safety import RouterDescriptor
from nat_app.speculation.safety import SpeculationSafetyConfig
from nat_app.speculation.safety import is_marked_speculation_unsafe
from nat_app.speculation.safety import speculation_unsafe
from nat_app.speculation.strategies import RouterBranchResolution
from nat_app.speculation.strategies import RouterBranchStrategy
from nat_app.speculation.strategies import SpeculationOpportunity
from nat_app.speculation.strategies import SpeculationStrategy

__all__ = [
    "Resolution",
    "ResolutionPolicy",
    "RouterBranchResolution",
    "RouterBranchStrategy",
    "RouterDescriptor",
    "SpeculationOpportunity",
    "SpeculationPlan",
    "SpeculationPlanner",
    "SpeculationSafetyConfig",
    "SpeculationStrategy",
    "is_marked_speculation_unsafe",
    "partition_targets",
    "plan_speculation",
    "speculation_unsafe",
]
