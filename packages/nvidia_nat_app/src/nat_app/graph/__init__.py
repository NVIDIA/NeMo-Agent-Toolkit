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
Graph analysis, optimization, and scheduling primitives.

Primary entry points:
    - ``AbstractFrameworkAdapter`` -- abstract base class for framework integrations.
    - ``Graph`` -- the central interchange type for all algorithms.

For the one-call ``GraphOptimizer`` wrapper, see ``nat_app.compiler``.

Core types:
    - ``AccessSet`` -- multi-object, nested-path read/write tracking.
    - ``NodeAnalysis`` -- per-node read/write/mutation profile.

Adapter protocols (for framework packages):
    - ``GraphExtractor`` -- extract a Graph from framework artifacts.
    - ``NodeIntrospector`` -- extract node functions and schema info.
    - ``GraphBuilder`` -- build optimized framework artifacts.
    - ``LLMDetector`` -- identify LLM objects for priority analysis.

LLM detection:
    - ``LLMCallInfo`` -- per-node LLM call detection result.
"""

from nat_app.graph.access import AccessSet
from nat_app.graph.access import ReducerSet
from nat_app.graph.adapter import AbstractFrameworkAdapter
from nat_app.graph.analysis import GraphAnalysisResult
from nat_app.graph.analysis import NodeAnalysis
from nat_app.graph.llm_detection import LLMCallInfo
from nat_app.graph.models import BranchInfo
from nat_app.graph.models import CompilationResult
from nat_app.graph.models import EdgeAnalysis
from nat_app.graph.models import EdgeType
from nat_app.graph.models import TransformationResult
from nat_app.graph.protocols import GraphBuilder
from nat_app.graph.protocols import GraphExtractor
from nat_app.graph.protocols import LLMDetector
from nat_app.graph.protocols import NodeIntrospector
from nat_app.graph.types import BranchGroup
from nat_app.graph.types import BranchGroupType
from nat_app.graph.types import CostMetric
from nat_app.graph.types import Edge
from nat_app.graph.types import EdgeKind
from nat_app.graph.types import Graph
from nat_app.graph.types import NodeInfo
from nat_app.graph.types import PriorityLevel
from nat_app.graph.types import ProfiledNodeCost

__all__ = [
    "AccessSet",
    "AbstractFrameworkAdapter",
    "BranchGroup",
    "BranchGroupType",
    "BranchInfo",
    "CompilationResult",
    "CostMetric",
    "Edge",
    "EdgeAnalysis",
    "EdgeKind",
    "EdgeType",
    "Graph",
    "GraphAnalysisResult",
    "GraphBuilder",
    "GraphExtractor",
    "LLMCallInfo",
    "LLMDetector",
    "NodeAnalysis",
    "NodeInfo",
    "NodeIntrospector",
    "PriorityLevel",
    "ProfiledNodeCost",
    "ReducerSet",
    "TransformationResult",
]
