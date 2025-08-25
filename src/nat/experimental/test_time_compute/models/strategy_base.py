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

from abc import ABC
from abc import abstractmethod

from nat.builder.builder import Builder
from nat.data_models.ttc_strategy import TTCStrategyBaseConfig
from nat.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from nat.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from nat.experimental.test_time_compute.models.ttc_item import TTCItem


class StrategyBase(ABC):
    """
    Abstract base class for strategy implementations.

    This class defines the interface for strategies that can be used in the
    TTC framework. Concrete strategy classes should
    implement the methods defined in this class.
    """

    def __init__(self, config: TTCStrategyBaseConfig) -> None:
        self.config: TTCStrategyBaseConfig = config
        self.pipeline_type: PipelineTypeEnum | None = None

    @abstractmethod
    async def build_components(self, builder: Builder) -> None:
        """Build the components required for the selector."""
        pass

    @abstractmethod
    async def ainvoke(self,
                      items: list[TTCItem],
                      original_prompt: str | None = None,
                      agent_context: str | None = None,
                      **kwargs) -> [TTCItem]:
        pass

    @abstractmethod
    def supported_pipeline_types(self) -> [PipelineTypeEnum]:
        """Return the stage types supported by this selector."""
        pass

    @abstractmethod
    def stage_type(self) -> StageTypeEnum:
        """Return the stage type of this strategy."""
        pass

    def set_pipeline_type(self, pipeline_type: PipelineTypeEnum) -> None:
        """Set the pipeline type for this strategy."""
        if pipeline_type in self.supported_pipeline_types():
            self.pipeline_type = pipeline_type
        else:
            raise ValueError(f"Pipeline type {pipeline_type} is not supported by this strategy.")
