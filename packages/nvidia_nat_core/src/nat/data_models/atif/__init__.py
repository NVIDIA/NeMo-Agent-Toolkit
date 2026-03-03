# SPDX-FileCopyrightText: Copyright (c) 2025, Harbor Framework Contributors (https://github.com/harbor-framework/harbor)
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Pydantic models for Agent Trajectory Interchange Format (ATIF).

Models are derived from the Harbor reference implementation
(https://github.com/harbor-framework/harbor) and follow the ATIF RFC
(0001-trajectory-format).  NAT-specific relaxations are documented inline
in the individual model files.

Backward-compatible aliases (``ATIFStep``, ``ATIFTrajectory``, etc.) are
provided so that existing NAT code continues to work without import changes.
"""

from nat.data_models.atif.agent import Agent
from nat.data_models.atif.content import ContentPart
from nat.data_models.atif.content import ImageSource
from nat.data_models.atif.final_metrics import FinalMetrics
from nat.data_models.atif.metrics import Metrics
from nat.data_models.atif.observation import Observation
from nat.data_models.atif.observation_result import ObservationResult
from nat.data_models.atif.step import Step
from nat.data_models.atif.subagent_trajectory_ref import SubagentTrajectoryRef
from nat.data_models.atif.tool_call import ToolCall
from nat.data_models.atif.trajectory import ATIF_VERSION
from nat.data_models.atif.trajectory import Trajectory

# ---------------------------------------------------------------------------
# Backward-compatible aliases used by the converter, API server, and tests.
# Prefer the Harbor-aligned names for new code.
# ---------------------------------------------------------------------------
ATIFAgentConfig = Agent
ATIFContentPart = ContentPart
ATIFImageSource = ImageSource
ATIFFinalMetrics = FinalMetrics
ATIFStepMetrics = Metrics
ATIFObservation = Observation
ATIFObservationResult = ObservationResult
ATIFStep = Step
ATIFSubagentTrajectoryRef = SubagentTrajectoryRef
ATIFToolCall = ToolCall
ATIFTrajectory = Trajectory

__all__ = [
    "ATIF_VERSION",
    "ATIFAgentConfig",
    "ATIFContentPart",
    "ATIFFinalMetrics",
    "ATIFImageSource",
    "ATIFObservation",
    "ATIFObservationResult",
    "ATIFStep",
    "ATIFStepMetrics",
    "ATIFSubagentTrajectoryRef",
    "ATIFToolCall",
    "ATIFTrajectory",
    "Agent",
    "ContentPart",
    "FinalMetrics",
    "ImageSource",
    "Metrics",
    "Observation",
    "ObservationResult",
    "Step",
    "SubagentTrajectoryRef",
    "ToolCall",
    "Trajectory",
]
