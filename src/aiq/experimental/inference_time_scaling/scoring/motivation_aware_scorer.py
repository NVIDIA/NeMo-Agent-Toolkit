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

import asyncio
import logging
import re

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_its_strategy
from aiq.data_models.its_strategy import ITSStrategyBaseConfig
from aiq.experimental.inference_time_scaling.models.its_item import ITSItem
from aiq.experimental.inference_time_scaling.models.scoring_config import MotivationAwareScoringConfig
from aiq.experimental.inference_time_scaling.models.stage_enums import PipelineTypeEnum
from aiq.experimental.inference_time_scaling.models.stage_enums import StageTypeEnum
from aiq.experimental.inference_time_scaling.models.strategy_base import StrategyBase
from aiq.utils.io.model_processing import remove_r1_think_tags

logger = logging.getLogger(__name__)


class MotivationAwareScorer(StrategyBase):
    """
    A strategy that scores an ITSItem's output based on how well it
    addresses both the original input (task) and the 'motivation' from metadata.
    """

    def __init__(self, config: ITSStrategyBaseConfig) -> None:
        super().__init__(config)
        self.llm_bound = None

    async def build_components(self, builder: Builder) -> None:
        self.llm_bound = await builder.get_llm(self.config.scoring_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    def supported_pipeline_types(self) -> list[PipelineTypeEnum]:
        return [PipelineTypeEnum.TOOL_USE]

    def stage_type(self) -> StageTypeEnum:
        return StageTypeEnum.SCORING

    async def ainvoke(self,
                      items: list[ITSItem],
                      original_prompt: str | None = None,
                      agent_context: str | None = None,
                      **kwargs) -> list[ITSItem]:
        """
        Scores each item by combining the original 'task_description' and 'motivation' with the 'output'.
        The resulting score is stored in item.score.
        """
        from langchain_core.language_models import BaseChatModel
        from langchain_core.prompts import PromptTemplate

        if not isinstance(self.llm_bound, BaseChatModel):
            raise ValueError("scoring_llm must be a BaseChatModel instance for MotivationAwareScorer.")

        scoring_model: BaseChatModel = self.llm_bound

        scoring_template = PromptTemplate(template=self.config.scoring_template,
                                          input_variables=["task", "motivation", "output"],
                                          validate_template=True)

        async def score_item(item: ITSItem) -> float:
            task_str = str(item.input) or ""
            motivation_str = str(item.metadata) if item.metadata else ""
            output_str = str(item.output) or ""

            prompt = (await scoring_template.ainvoke({
                "task": task_str, "motivation": motivation_str, "output": output_str
            })).to_string()

            response = (await scoring_model.ainvoke(prompt)).content
            response = remove_r1_think_tags(response or "")

            match = re.search(r'FINAL SCORE:\s*([\d.]+)', response)
            if not match:
                logger.warning(f"Could not parse score from response: {response}")
                return 0.0

            score_str = match.group(1)
            try:
                return float(score_str)
            except ValueError:
                logger.warning(f"Could not convert score '{score_str}' to float.")
                return 0.0

        tasks = [score_item(item) for item in items]
        scores = await asyncio.gather(*tasks)

        for i, s in enumerate(scores):
            items[i].score = s

        return items


@register_its_strategy(config_type=MotivationAwareScoringConfig)
async def register_motivation_aware_scorer(config: MotivationAwareScoringConfig, builder: Builder):
    scorer = MotivationAwareScorer(config)
    await scorer.build_components(builder)
    yield scorer
