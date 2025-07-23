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
from aiq.experimental.inference_time_scaling.models.scoring_config import LLMBasedAgentScoringConfig
from aiq.experimental.inference_time_scaling.models.stage_enums import PipelineTypeEnum
from aiq.experimental.inference_time_scaling.models.stage_enums import StageTypeEnum
from aiq.experimental.inference_time_scaling.models.strategy_base import StrategyBase
from aiq.utils.io.model_processing import remove_r1_think_tags

logger = logging.getLogger(__name__)


class LLMBasedAgentScorer(StrategyBase):

    def __init__(self, config: ITSStrategyBaseConfig) -> None:
        super().__init__(config)
        self.llm_bound = None

    async def build_components(self, builder: Builder) -> None:
        """
        Build the components required for the planner.
        """
        self.llm_bound = await builder.get_llm(self.config.scoring_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    def supported_pipeline_types(self) -> [PipelineTypeEnum]:
        return [PipelineTypeEnum.AGENT_EXECUTION]

    def stage_type(self) -> StageTypeEnum:
        return StageTypeEnum.SCORING

    async def score_single(self, original_prompt: str, agent_context: str, item: ITSItem) -> float:
        """
        Score a single planning item using the LLM.

        Args:
            original_prompt (str): The original prompt.
            agent_context (str): The agent context.
            item (ITSItem): The item to score.

        Returns:
            float: The score of the item.
        """

        try:
            from langchain_core.language_models import BaseChatModel
            from langchain_core.prompts import PromptTemplate
        except ImportError:
            raise ImportError("langchain-core is not installed. Please install it to use SingleShotMultiPlanPlanner.\n"
                              "This error can be resolved by installing aiqtoolkit-langchain.")

        if not isinstance(self.llm_bound, BaseChatModel):
            raise ValueError("The `scoring_llm` must be an instance of `BaseChatModel`.")

        model: BaseChatModel = self.llm_bound

        prompt_template = PromptTemplate(
            template=self.config.scoring_template,
            input_variables=["objective", "input", "output"],
            validate_template=True,
        )

        prompt = (await prompt_template.ainvoke(
            input={
                "objective": agent_context,
                "input": str(item.input) if not original_prompt else original_prompt,
                "output": str(item.output)
            }))

        response = (await model.ainvoke(prompt)).content
        response = remove_r1_think_tags(response)

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
                      items: list[ITSItem],
                      original_prompt: str | None = None,
                      agent_context: str | None = None,
                      **kwargs) -> list[ITSItem]:
        """
        Score a list of planning items.

        Args:
            original_prompt (str): The original prompt.
            agent_context (str): The agent context.
            items (list[ITSItem]): The list of planning items to score.

        Returns:
            list[float]: A list of scores corresponding to each planning item.
        """
        # Run score single concurrently for all planning items
        # Then set the score attribute on each planning item
        if not items:
            return []
        tasks = [
            self.score_single(original_prompt=original_prompt, agent_context=agent_context, item=item) for item in items
        ]

        # Gather all scores concurrently
        scores = await asyncio.gather(*tasks)

        if len(scores) != len(items):
            logger.warning(f"Number of scores {len(scores)} does not match the number of items {len(items)}.")
            raise ValueError("Mismatch in number of scores and planning items.")

        logger.debug("Scores for planning items: %s", scores)

        # Set the score on each planning item for reference
        for idx, score in enumerate(scores):
            items[idx].score = score

        return items


@register_its_strategy(config_type=LLMBasedAgentScoringConfig)
async def register_llm_based_agent_scorer(config: LLMBasedAgentScoringConfig, builder: Builder):
    """
    Register the LLM-based agent scorer with the provided configuration and builder.

    Args:
        config (LLMBasedAgentScoringConfig): The configuration for the LLM-based agent scorer.
        builder (Builder): The builder instance to use for building components.

    Returns:
        LLMBasedAgentScorer: The registered LLM-based agent scorer.
    """
    scorer = LLMBasedAgentScorer(config)
    await scorer.build_components(builder)
    yield scorer
