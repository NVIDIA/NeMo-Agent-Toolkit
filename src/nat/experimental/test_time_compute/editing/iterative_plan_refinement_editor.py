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
from nat.experimental.test_time_compute.models.editor_config import IterativePlanRefinementConfig
from nat.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from nat.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from nat.experimental.test_time_compute.models.strategy_base import StrategyBase
from nat.experimental.test_time_compute.models.ttc_item import TTCItem
from nat.utils.io.model_processing import remove_r1_think_tags

logger = logging.getLogger(__name__)


class IterativePlanRefinementEditor(StrategyBase):
    """
    A planner that generates an initial plan, then refines it multiple times
    using the same LLM. Each iteration updates the plan to (hopefully) be better.
    """

    def __init__(self, config: TTCStrategyBaseConfig) -> None:
        super().__init__(config)
        self.llm_bound = None

    def supported_pipeline_types(self) -> [PipelineTypeEnum]:
        return [PipelineTypeEnum.PLANNING]

    def stage_type(self) -> StageTypeEnum:
        return StageTypeEnum.EDITING

    async def build_components(self, builder: Builder) -> None:
        """
        Build the components required for the iterative planner.
        """
        logger.debug("Building components for IterativePlanRefinementEditor")
        self.llm_bound = await builder.get_llm(self.config.editor_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    async def refine_single(self, prompt: str, context: str, ttc_item: TTCItem, prompt_idx: int) -> TTCItem:
        from langchain_core.language_models import BaseChatModel
        from langchain_core.prompts import PromptTemplate

        if not isinstance(self.llm_bound, BaseChatModel):
            raise ValueError("editor_llm must be a BaseChatModel instance for iterative plan refinement.")

        llm: BaseChatModel = self.llm_bound

        # Refinement loop
        refinement_template = PromptTemplate(
            template=self.config.refinement_template,
            input_variables=["current_plan", "context", "original_prompt"],
            validate_template=True,
        )

        current_plan = ttc_item.plan
        for iteration in range(1, self.config.num_iterations + 1):
            logger.info("Refinement iteration %d / %d for prompt %d", iteration, self.config.num_iterations, prompt_idx)
            refine_prompt = (await refinement_template.ainvoke({
                "current_plan": current_plan, "context": context, "original_prompt": prompt
            })).to_string()

            refine_response = await llm.ainvoke(refine_prompt)
            refined_plan = remove_r1_think_tags(
                refine_response.content if hasattr(refine_response, 'content') else str(refine_response))
            refined_plan = re.sub(r'(?i)^\s*EDITED PLAN:\s*', '', refined_plan).strip()
            if refined_plan:
                current_plan = refined_plan
            else:
                logger.warning("Refinement iteration %d for prompt %d produced an empty plan; keeping existing plan.",
                               iteration,
                               prompt_idx)

        logger.info("IterativePlanRefinementPlanner produced a final plan after %d iterations.",
                    self.config.num_iterations)

        ttc_item.plan = current_plan
        # Return a single final plan
        return ttc_item

    async def ainvoke(self,
                      items: list[TTCItem],
                      original_prompt: str | None = None,
                      agent_context: str | None = None,
                      **kwargs) -> list[TTCItem]:
        """
        Runs the iterative plan refinement process on the provided planning items.

        Each planning item is refined in parallel the configured number of times. Default is 3.

        Args:
            items (list[TTCItem]): The planning items to refine.
            original_prompt (str): The original prompt used to generate the plans.
            agent_context (str): The context for the agent.

        Returns:
            list[TTCItem]: The refined planning items.
        """

        if not original_prompt or not agent_context:
            raise ValueError("Arguments original_prompt and agent_context must be provdied.")

        # Generate feedback for each planning item concurrently
        tasks = [
            self.refine_single(prompt=original_prompt, context=agent_context, ttc_item=item, prompt_idx=i + 1)
            for i, item in enumerate(items)
        ]

        # Run the tasks concurrently and gather results
        refined_planning_items = await asyncio.gather(*tasks)

        return refined_planning_items


@register_ttc_strategy(config_type=IterativePlanRefinementConfig)
async def register_iterative_plan_refinement_editor(config: IterativePlanRefinementConfig, builder: Builder):
    """
    Register the IterativePlanRefinementEditor strategy.

    Args:
        config (IterativePlanRefinementConfig): The configuration for the strategy.

    Returns:
        IterativePlanRefinementEditor: The registered strategy instance.
    """

    editor = IterativePlanRefinementEditor(config)
    await editor.build_components(builder=builder)

    yield editor
