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

from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_ttc_strategy
from aiq.experimental.test_time_compute.models.selection_config import BestOfNSelectionConfig
from aiq.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from aiq.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from aiq.experimental.test_time_compute.models.strategy_base import StrategyBase
from aiq.experimental.test_time_compute.models.ttc_item import TTCItem

logger = logging.getLogger(__name__)


class BestOfNSelector(StrategyBase):

    async def build_components(self, builder: Builder) -> None:
        pass

    def supported_pipeline_types(self) -> [PipelineTypeEnum]:
        return [PipelineTypeEnum.PLANNING, PipelineTypeEnum.AGENT_EXECUTION, PipelineTypeEnum.TOOL_USE]

    def stage_type(self) -> StageTypeEnum:
        return StageTypeEnum.SELECTION

    async def ainvoke(self,
                      items: list[TTCItem],
                      original_prompt: str | None = None,
                      agent_context: str | None = None,
                      **kwargs) -> [TTCItem]:

        # Assert that every planning item has a non NoneType score
        for item in items:
            if item.score is None:
                raise ValueError("Every planning item must have a score. Did you use a scorer before this?")

        # Pick the planning item with the highest score
        best_item = max(items, key=lambda x: x.score)

        return [best_item]


@register_ttc_strategy(config_type=BestOfNSelectionConfig)
async def register_best_of_n_selector(config: BestOfNSelectionConfig, builder: Builder):
    """
    Register the BestOfNSelector strategy.
    """
    selector = BestOfNSelector(config)
    yield selector
