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
Built-in compilation stages for the optimization pipeline.

These stages decompose the optimization into composable units that can be
reordered, extended, or replaced by framework-specific stages.
"""

from nat_app.graph.types import BranchGroup
from nat_app.graph.types import BranchGroupType
from nat_app.stages.edge_classification import EdgeClassificationStage
from nat_app.stages.extract import ExtractStage
from nat_app.stages.llm_analysis import LLMAnalysisStage
from nat_app.stages.node_analysis import NodeAnalysisStage
from nat_app.stages.priority_assignment import PriorityAssignmentStage
from nat_app.stages.priority_assignment import PriorityStrategy
from nat_app.stages.priority_assignment import SJFPriorityStrategy
from nat_app.stages.scheduling import SchedulingStage
from nat_app.stages.topology import TopologyStage
from nat_app.stages.validate import ValidateStage

__all__ = [
    "BranchGroup",
    "BranchGroupType",
    "EdgeClassificationStage",
    "ExtractStage",
    "LLMAnalysisStage",
    "NodeAnalysisStage",
    "PriorityAssignmentStage",
    "PriorityStrategy",
    "SchedulingStage",
    "SJFPriorityStrategy",
    "TopologyStage",
    "ValidateStage",
]
