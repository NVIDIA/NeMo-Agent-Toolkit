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
from nat.experimental.test_time_compute.models.search_config import SingleShotMultiPlanConfig
from nat.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from nat.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from nat.experimental.test_time_compute.models.strategy_base import StrategyBase
from nat.experimental.test_time_compute.models.ttc_item import TTCItem
from nat.utils.io.model_processing import remove_r1_think_tags

logger = logging.getLogger(__name__)


class SingleShotMultiPlanPlanner(StrategyBase):
    """
    Implementation of the Single Shot Multi Plan Planner.
    This planner generates multiple plans in a single shot.
    """

    def __init__(self, config: TTCStrategyBaseConfig) -> None:
        super().__init__(config)
        self.llm_bound = None

    async def build_components(self, builder: Builder) -> None:
        self.llm_bound = await builder.get_llm(self.config.planning_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    def supported_pipeline_types(self) -> [PipelineTypeEnum]:
        return [PipelineTypeEnum.PLANNING]

    def stage_type(self) -> StageTypeEnum:
        return StageTypeEnum.SEARCH

    async def ainvoke(self,
                      items: list[TTCItem],
                      original_prompt: str | None = None,
                      agent_context: str | None = None,
                      **kwargs) -> list[TTCItem]:
        """
        Generate a TTCItem based on the provided prompt.
        """

        try:
            from langchain_core.language_models import BaseChatModel
            from langchain_core.prompts import PromptTemplate
        except ImportError:
            raise ImportError("langchain-core is not installed. Please install it to use SingleShotMultiPlanPlanner.\n"
                              "This error can be resolve by installing nvidia-nat-langchain.")

        planning_template = PromptTemplate(template=self.config.planning_template,
                                           input_variables=["context", "prompt"],
                                           validate_template=True)
        prompt = (await planning_template.ainvoke(input={
            "context": agent_context, "prompt": original_prompt
        })).to_string()

        # assert self.config.planning llm is a BaseChatModel
        if not isinstance(self.llm_bound, BaseChatModel):
            raise ValueError("The `planning_llm` must be an instance of `BaseChatModel`.")

        model: BaseChatModel = self.llm_bound

        async def generate_plan(llm: BaseChatModel, plan_prompt: str, temperature: float) -> TTCItem:
            """
            Helper function to generate a plan using the provided prompt and temperature.
            """
            llm_bound = llm.bind(temperature=temperature)
            response = await llm_bound.ainvoke(plan_prompt)
            cleaned = remove_r1_think_tags(response.content if hasattr(response, 'content') else str(response))

            # Plan will be the string following 'PLAN:'. Use Regex tpo extract
            cleaned = re.sub(r'(?i)^\s*PLAN:\s*', '', cleaned).strip()

            if not cleaned:
                logger.warning(f"No plan generated for the prompt: {plan_prompt}.")
                # Return an empty PlanningItem to avoid breaking the generation loop
                return TTCItem(plan="Plan was not generated")

            return TTCItem(plan=cleaned)

        # Define a list of temperatures based on min and max temperature in the config and number of plans to generate
        temperatures = [
            self.config.min_temperature + (i / (self.config.num_plans - 1)) *
            (self.config.max_temperature - self.config.min_temperature) for i in range(self.config.num_plans)
        ]

        # Generate plans using the defined temperatures in parallel using asyncio
        tasks = [generate_plan(model, prompt, temperature) for temperature in temperatures]
        # Run the tasks concurrently and gather results
        plans = await asyncio.gather(*tasks)

        if not plans:
            raise ValueError("No plans were generated. Please check the LLM response.")

        logger.info("Generated %d plans from the SingleShotMultiPlanPlanner", self.config.num_plans)

        logger.debug("Generated plans: %s", [plan.dict() for plan in plans])

        return plans


@register_ttc_strategy(config_type=SingleShotMultiPlanConfig)
async def register_single_shot_multi_plan_planner(config: SingleShotMultiPlanConfig, builder: Builder):
    """
    Register the SingleShotMultiPlanPlanner strategy with the provided configuration.
    """
    planner = SingleShotMultiPlanPlanner(config)
    await planner.build_components(builder)
    yield planner
