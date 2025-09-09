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

from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import TokenUsageBaseModel


class IntermediatePropertyAdaptor(IntermediateStep):

    @classmethod
    def from_intermediate_step(cls, step: IntermediateStep) -> "IntermediatePropertyAdaptor":
        """
        Create an adaptor instance from an existing IntermediateStep.
        Uses the dict() representation of the instance to initialize the adaptor.
        """
        return cls(**step.model_dump())

    @property
    def token_usage(self) -> TokenUsageBaseModel:
        return self.payload.usage_info.token_usage if self.payload.usage_info else TokenUsageBaseModel()

    @property
    def seconds_between_calls(self) -> int:
        return self.payload.usage_info.seconds_between_calls if self.payload.usage_info else 0

    @property
    def llm_text_input(self) -> str:
        ret = ""
        if self.payload.data and self.event_type == IntermediateStepType.LLM_START:
            ret = self.payload.data.input
        return ret

    @property
    def llm_text_output(self) -> str:
        ret = ""
        if self.payload.data and self.event_type == IntermediateStepType.LLM_END:
            ret = self.payload.data.output
        return ret

    @property
    def llm_text_chunk(self) -> str:
        ret = ""
        if self.payload.data and self.event_type == IntermediateStepType.LLM_NEW_TOKEN:
            ret = self.payload.data.chunk
        return ret

    @property
    def tool_input(self) -> str:
        ret = ""
        if self.payload.data and self.event_type == IntermediateStepType.TOOL_START:
            ret = self.payload.data.input
        return ret

    @property
    def tool_output(self) -> str:
        ret = ""
        if self.payload.data and self.event_type == IntermediateStepType.TOOL_END:
            ret = self.payload.data.output
        return ret

    @property
    def llm_name(self) -> str:
        ret = ""
        if self.payload.name and self.event_type in [IntermediateStepType.LLM_START, IntermediateStepType.LLM_END]:
            ret = self.payload.name
        return ret

    @property
    def tool_name(self) -> str:
        ret = ""
        if self.payload.name and self.event_type in [IntermediateStepType.TOOL_START, IntermediateStepType.TOOL_END]:
            ret = self.payload.name
        return ret

    @property
    def function_name(self) -> str:
        return self.function_ancestry.function_name

    @property
    def function_id(self) -> str:
        return self.function_ancestry.function_id

    @property
    def parent_function_id(self) -> str:
        return self.function_ancestry.parent_id

    @property
    def parent_function_name(self) -> str:
        return self.function_ancestry.parent_name
