# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from enum import Enum


class PipelineTypeEnum(str, Enum):
    """
    Enum to represent the type of pipeline used in Inference Time Scaling.
    """
    PLANNING = "planning"
    TOOL_USE = "tool_use"
    AGENT_EXECUTION = "agent_execution"
    CUSTOM = "custom"

    def __str__(self) -> str:
        return self.value


class StageTypeEnum(str, Enum):
    """
    Enum to represent the type of stage in a pipeline.
    """
    SEARCH = "search"
    EDITING = "editing"
    SCORING = "scoring"
    SELECTION = "selection"
    CUSTOM = "custom"

    def __str__(self) -> str:
        return self.value
