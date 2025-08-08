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
from aiq.cli.register_workflow import register_ttc_strategy
from aiq.experimental.test_time_compute.models.search_config import MultiLLMPlanConfig
from aiq.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from aiq.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from aiq.experimental.test_time_compute.models.strategy_base import StrategyBase
from aiq.experimental.test_time_compute.models.ttc_item import TTCItem
from aiq.utils.io.model_processing import remove_r1_think_tags

logger = logging.getLogger(__name__)


class MultiLLMPlanner(StrategyBase):
    """
    A planner that uses multiple LLMs to generate plans. Each LLM can generate
    a specified number of plans, and all plans are combined.
    """

    def __init__(self, config: MultiLLMPlanConfig) -> None:
        super().__init__(config)
        self.config = config
        self.llms_bound = []  # Will hold the "bound" LLMs after build_components

    async def build_components(self, builder: Builder) -> None:
        """
        Build the components required for this multi-LLM planner.
        Binds each LLMRef from the config with the selected framework wrapper (LANGCHAIN).
        """
        logger.debug("Building components for MultiLLMPlanner")
        self.llms_bound = []
        for llm_ref in self.config.llms:
            bound_llm = await builder.get_llm(llm_ref, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
            self.llms_bound.append(bound_llm)

    def supported_pipeline_types(self) -> [PipelineTypeEnum]:
        return [PipelineTypeEnum.PLANNING]

    def stage_type(self) -> StageTypeEnum:
        return StageTypeEnum.SEARCH

    async def _generate_plan_for_temperature(self, llm, base_prompt: str, temperature: float) -> TTCItem:
        bound_llm = llm.bind(temperature=temperature)
        response = await bound_llm.ainvoke(base_prompt)
        cleaned = remove_r1_think_tags(response.content if hasattr(response, 'content') else str(response))
        # The plan is expected to start with "PLAN:" and all the text after it is the plan
        cleaned = re.sub(r'(?i)^\s*PLAN:\s*', '', cleaned).strip()

        if not cleaned:
            logger.warning(f"No plan generated for the prompt: {base_prompt}.")
            # Return an empty PlanningItem to avoid breaking the generation loop
            return TTCItem(plan="Plan was not generated")

        return TTCItem(plan=cleaned)

    async def _generate_plans_for_llm(self, llm, base_prompt: str) -> list[TTCItem]:
        if self.config.plans_per_llm == 1:
            temps = [self.config.min_temperature]
        else:
            temps = [
                self.config.min_temperature + (i / (self.config.plans_per_llm - 1)) *
                (self.config.max_temperature - self.config.min_temperature) for i in range(self.config.plans_per_llm)
            ]
        tasks = [self._generate_plan_for_temperature(llm, base_prompt, temp) for temp in temps]
        return await asyncio.gather(*tasks)

    async def ainvoke(self,
                      items: list[TTCItem],
                      original_prompt: str | None = None,
                      agent_context: str | None = None,
                      **kwargs) -> list[TTCItem]:
        """
        Generate a list of PlanningItems by querying each LLM in self.llms_bound.
        Each LLM produces 'plans_per_llm' plans.
        """
        try:
            from langchain_core.prompts import PromptTemplate
        except ImportError:
            raise ImportError("langchain-core is not installed. Please install it to use MultiLLMPlanner.\n"
                              "This error can be resolve by installing aiqtoolkit-langchain.")

        # Create a single PromptTemplate
        planning_template = PromptTemplate(template=self.config.planning_template,
                                           input_variables=["context", "prompt"],
                                           validate_template=True)

        # Format the prompt once
        base_prompt = (await planning_template.ainvoke({
            "context": agent_context, "prompt": original_prompt
        })).to_string()

        # Launch generation for each llm concurrently using the new helper method
        tasks = [self._generate_plans_for_llm(llm, base_prompt) for llm in self.llms_bound]
        results_nested = await asyncio.gather(*tasks)

        # Flatten the nested lists of TTCItem
        all_plans: list[TTCItem] = [p for sub in results_nested for p in sub]
        logger.info("MultiLLMPlanner generated %d plans total.", len(all_plans))
        return all_plans


@register_ttc_strategy(config_type=MultiLLMPlanConfig)
async def register_multi_llm_planner(config: MultiLLMPlanConfig, builder: Builder):
    """
    Register the MultiLLMPlanner strategy with the provided configuration.
    """
    planner = MultiLLMPlanner(config)
    await planner.build_components(builder)
    yield planner
