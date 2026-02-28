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
"""Compatibility re-export of ATIF models from core."""

from nat.data_models.atif import ATIFAgentConfig
from nat.data_models.atif import ATIFFinalMetrics
from nat.data_models.atif import ContentPart
from nat.data_models.atif import ImageSource
from nat.data_models.atif import ATIFObservation
from nat.data_models.atif import ATIFObservationResult
from nat.data_models.atif import ATIFStep
from nat.data_models.atif import ATIFStepMetrics
from nat.data_models.atif import SubagentTrajectoryRef
from nat.data_models.atif import ATIFToolCall
from nat.data_models.atif import ATIFTrajectory
from nat.data_models.atif import ATIF_VERSION

__all__ = [
    "ATIF_VERSION",
    "ATIFAgentConfig",
    "ATIFFinalMetrics",
    "ContentPart",
    "ImageSource",
    "ATIFObservation",
    "ATIFObservationResult",
    "ATIFStep",
    "ATIFStepMetrics",
    "SubagentTrajectoryRef",
    "ATIFToolCall",
    "ATIFTrajectory",
]
