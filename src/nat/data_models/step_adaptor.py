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

import logging
from enum import Enum

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from aiq.data_models.intermediate_step import IntermediateStepType

logger = logging.getLogger(__name__)


class StepAdaptorMode(str, Enum):
    DEFAULT = "default"
    CUSTOM = "custom"
    OFF = "off"


class StepAdaptorConfig(BaseModel):
    """
    Configures how intermediate steps are filtered and normalized by the StepAdaptor.

    Args:
        mode (StepAdaptorMode): One of:
            - 'current' => pass only LLM (all LLM_* events) + TOOL_END
            - 'end_events_only' => pass only LLM_END and TOOL_END
            - 'custom' => pass only the events in custom_event_types
        custom_event_types (list[IntermediateStepType]):
            If mode == 'custom', we only pass events whose event_type is in this list.
            Otherwise, this field is ignored.
    """
    mode: StepAdaptorMode = StepAdaptorMode.DEFAULT
    custom_event_types: list[IntermediateStepType] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_custom_event_types(self) -> "StepAdaptorConfig":
        """
        Validates custom configurations
        """
        if self.mode != StepAdaptorMode.CUSTOM and self.custom_event_types:
            logger.warning("Ignoring custom_event_types because mode is not 'custom'")
            self.custom_event_types = []
        elif self.mode == StepAdaptorMode.CUSTOM and not self.custom_event_types:
            logger.warning("No custom_event_types provided for custom mode. Defaulting to CUSTOM_START and CUSTOM_END")
            self.custom_event_types = [IntermediateStepType.CUSTOM_START, IntermediateStepType.CUSTOM_END]
        elif self.mode == StepAdaptorMode.OFF:
            logger.warning("StepAdaptor is disabled. Ignoring all intermediate event types")
            self.custom_event_types = []
        return self
