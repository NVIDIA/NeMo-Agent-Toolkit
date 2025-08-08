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
from aiq.experimental.test_time_compute.models.selection_config import ThresholdSelectionConfig
from aiq.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from aiq.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from aiq.experimental.test_time_compute.models.strategy_base import StrategyBase
from aiq.experimental.test_time_compute.models.ttc_item import TTCItem

logger = logging.getLogger(__name__)


class ThresholdSelector(StrategyBase):
    """
    Downselects only those TTCItems whose 'score' >= config.threshold.
    """

    async def build_components(self, builder: Builder) -> None:
        # No special components needed
        pass

    def supported_pipeline_types(self) -> list[PipelineTypeEnum]:
        return [PipelineTypeEnum.TOOL_USE]

    def stage_type(self) -> StageTypeEnum:
        return StageTypeEnum.SELECTION

    async def ainvoke(self,
                      items: list[TTCItem],
                      original_prompt: str | None = None,
                      agent_context: str | None = None,
                      **kwargs) -> list[TTCItem]:
        threshold = self.config.threshold
        selected = [itm for itm in items if (itm.score is not None and itm.score >= threshold)]
        logger.info("ThresholdSelector: %d items => %d items (threshold=%.1f)", len(items), len(selected), threshold)
        return selected


@register_ttc_strategy(config_type=ThresholdSelectionConfig)
async def register_threshold_selector(config: ThresholdSelectionConfig, builder: Builder):
    selector = ThresholdSelector(config)
    yield selector
