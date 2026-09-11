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

import logging
from enum import StrEnum

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from nat.data_models.intermediate_step import IntermediateStepType

logger = logging.getLogger(__name__)


class StepAdaptorMode(StrEnum):
    DEFAULT = "default"
    CUSTOM = "custom"
    OFF = "off"


class StepAdaptorConfig(BaseModel):
    """
    Configures how intermediate steps are filtered and normalized by the ``StepAdaptor``.

    Args:
        mode (StepAdaptorMode): Mode determining which events are emitted (``StepAdaptorMode.DEFAULT``,
            ``StepAdaptorMode.CUSTOM``, or ``StepAdaptorMode.OFF``).
        custom_event_types (list[IntermediateStepType]):
            If ``mode`` is ``StepAdaptorMode.CUSTOM``, only events whose ``event_type`` is in this list are passed.
            Otherwise, this field is ignored.
        stream_llm_tokens (bool): Whether to emit intermediate LLM token events
            (``IntermediateStepType.LLM_NEW_TOKEN``). When ``False``, only ``IntermediateStepType.LLM_START``
            and ``IntermediateStepType.LLM_END`` are emitted.
        max_input_length (int): Maximum character length for input fields in intermediate step payloads.
            Exceeding text will be truncated. Must be greater than or equal to 0.
        max_output_length (int): Maximum character length for output fields in intermediate step payloads.
            Exceeding text will be truncated. Must be greater than or equal to 0.
    """
    mode: StepAdaptorMode = StepAdaptorMode.DEFAULT
    custom_event_types: list[IntermediateStepType] = Field(default_factory=list)
    stream_llm_tokens: bool = Field(
        default=False,
        description=("Whether to emit intermediate LLM token events (LLM_NEW_TOKEN). "
                     "When False, only LLM_START and LLM_END are emitted."),
    )
    max_input_length: int = Field(
        default=4000,
        ge=0,
        description=("Maximum character length for input fields in intermediate step payloads. "
                     "Exceeding text will be truncated."),
    )
    max_output_length: int = Field(
        default=4000,
        ge=0,
        description=("Maximum character length for output fields in intermediate step payloads. "
                     "Exceeding text will be truncated."),
    )

    @model_validator(mode="after")
    def check_custom_event_types(self) -> "StepAdaptorConfig":
        """
        Validates ``StepAdaptorConfig`` when ``mode`` is ``StepAdaptorMode.CUSTOM``.

        Returns:
            StepAdaptorConfig: The validated configuration instance.
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
