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

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_ttc_strategy
from nat.data_models.ttc_strategy import TTCStrategyBaseConfig
from nat.experimental.test_time_compute.models.scoring_config import LLMBasedPlanScoringConfig
from nat.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from nat.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from nat.experimental.test_time_compute.models.strategy_base import StrategyBase
from nat.experimental.test_time_compute.models.ttc_item import TTCItem
from nat.utils.io.model_processing import remove_r1_think_tags

logger = logging.getLogger(__name__)


class LLMBasedPlanScorer(StrategyBase):

    def __init__(self, config: TTCStrategyBaseConfig) -> None:
        super().__init__(config)
        self.llm_bound = None

    async def build_components(self, builder: Builder) -> None:
        """
        Build the components required for the planner.
        """
        self.llm_bound = await builder.get_llm(self.config.scoring_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    def supported_pipeline_types(self) -> [PipelineTypeEnum]:
        return [PipelineTypeEnum.PLANNING]

    def stage_type(self) -> StageTypeEnum:
        return StageTypeEnum.SCORING

    async def score_single(self, original_prompt: str, agent_context: str, planning_item: TTCItem) -> float:
        """
        Score a single planning item using the LLM.

        Args:
            original_prompt (str): The original prompt.
            agent_context (str): The agent context.
            planning_item (TTCItem): The item to score.

        Returns:
            float: The score of the item.
        """

        try:
            from langchain_core.language_models import BaseChatModel
            from langchain_core.prompts import PromptTemplate
        except ImportError:
            raise ImportError("langchain-core is not installed. Please install it to use SingleShotMultiPlanPlanner.\n"
                              "This error can be resolved by installing nvidia-nat-langchain.")

        if not isinstance(self.llm_bound, BaseChatModel):
            raise ValueError("The `scoring_llm` must be an instance of `BaseChatModel`.")

        model: BaseChatModel = self.llm_bound

        prompt_template = PromptTemplate(
            template=self.config.scoring_template,
            input_variables=["original_prompt", "context", "plan"],
            validate_template=True,
        )

        prompt = (await prompt_template.ainvoke(
            input={
                "original_prompt": original_prompt,
                "context": agent_context,
                "plan": remove_r1_think_tags(planning_item.plan)
            }))

        response = (await model.ainvoke(prompt)).content

        # Score will following the format of `FINAL SCORE: <float>` in the response from the LLM
        if not isinstance(response, str):
            logger.warning(f"Invalid response from LLM for scoring: {response}.")
            raise ValueError("Unable to parse the score from the LLM response.")

        response = response.strip()
        match = re.search(r'FINAL SCORE:\s*([\d.]+)', response)
        if not match:
            logger.warning(f"Could not parse the score from the response: {response}.")
            score_str = '0.0'
        else:
            score_str = match.group(1)

        try:
            score = float(score_str)
        except ValueError:
            logger.warning(f"Could not convert the score string '{score_str}' to float.")
            raise ValueError(f"Unable to convert the extracted score '{score_str}' to a float.")

        return score

    async def ainvoke(self,
                      items: list[TTCItem],
                      original_prompt: str | None = None,
                      agent_context: str | None = None,
                      **kwargs) -> list[TTCItem]:
        """
        Score a list of planning items.

        Args:
            original_prompt (str): The original prompt.
            agent_context (str): The agent context.
            items (list[TTCItem]): The list of planning items to score.

        Returns:
            list[float]: A list of scores corresponding to each planning item.
        """
        # Run score single concurrently for all planning items
        # Then set the score attribute on each planning item
        if not items:
            return []
        tasks = [
            self.score_single(original_prompt=original_prompt, agent_context=agent_context, planning_item=item)
            for item in items
        ]

        # Gather all scores concurrently
        scores = await asyncio.gather(*tasks)

        if len(scores) != len(items):
            logger.warning(f"Number of scores {len(scores)} does not match the number of planning items {len(items)}.")
            raise ValueError("Mismatch in number of scores and planning items.")

        logger.debug("Scores for planning items: %s", scores)

        # Set the score on each planning item for reference
        for idx, score in enumerate(scores):
            items[idx].score = score

        return items


@register_ttc_strategy(config_type=LLMBasedPlanScoringConfig)
async def register_llm_based_plan_scorer(config: LLMBasedPlanScoringConfig, builder: Builder):
    """
    Register the LLM-based plan scorer strategy.

    Args:
        config (LLMBasedPlanScoringConfig): The configuration for the strategy.
        builder (Builder): The builder instance.

    Returns:
        LLMBasedPlanScorer: The registered LLM-based plan scorer.
    """
    scorer = LLMBasedPlanScorer(config)
    await scorer.build_components(builder)
    yield scorer
