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

from __future__ import annotations
from pydantic import BaseModel
from pydantic import Field

from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepType


class IntermediateStepsFilterCondition(BaseModel):
    """
    Filter conditions for selecting intermediate steps from a trajectory.

    This model encapsulates the filtering logic used to select specific intermediate
    steps for evaluation. Multiple filter conditions can be defined to evaluate
    different parts of a trajectory separately.
    """

    name: str = Field(description="Name for this filter condition (used for organizing results)")
    event_type: IntermediateStepType | str | None = Field(
        default=None,
        description="Filter steps by event_type (e.g., 'TOOL_END', 'LLM_END', 'FUNCTION_END')"
    )
    payload_name: str | None = Field(
        default=None,
        description="Filter steps by payload.name (e.g., specific tool or function name)"
    )

    def filter_trajectory(self, trajectory: list[IntermediateStep]) -> list[IntermediateStep]:
        """
        Filter a trajectory based on these conditions.

        Args:
            trajectory: List of intermediate steps to filter

        Returns:
            List of filtered intermediate steps matching the conditions
        """
        filtered_steps = trajectory

        # Convert string event_type to enum if needed
        event_type_filter = None
        if self.event_type is not None:
            if isinstance(self.event_type, str):
                event_type_filter = IntermediateStepType(self.event_type)
            else:
                event_type_filter = self.event_type

        # Filter by event_type if specified
        if event_type_filter is not None:
            filtered_steps = [step for step in filtered_steps if step.event_type == event_type_filter]

        # Filter by payload.name if specified
        if self.payload_name is not None:
            filtered_steps = [
                step for step in filtered_steps
                if step.payload.name is not None and step.payload.name == self.payload_name
            ]
        return filtered_steps
    @classmethod
    def default(cls) -> IntermediateStepsFilterCondition:
        # Get the default filter conditions that essentially perform no filtering.
        return cls(name="default", event_type=None, payload_name=None)
