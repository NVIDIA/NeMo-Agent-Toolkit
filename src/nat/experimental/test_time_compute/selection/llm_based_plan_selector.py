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
import re

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_ttc_strategy
from nat.data_models.ttc_strategy import TTCStrategyBaseConfig
from nat.experimental.test_time_compute.models.selection_config import LLMBasedPlanSelectionConfig
from nat.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from nat.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from nat.experimental.test_time_compute.models.strategy_base import StrategyBase
from nat.experimental.test_time_compute.models.ttc_item import TTCItem
from nat.utils.io.model_processing import remove_r1_think_tags

logger = logging.getLogger(__name__)


class LLMBasedPlanSelector(StrategyBase):

    def __init__(self, config: TTCStrategyBaseConfig) -> None:
        super().__init__(config)
        self.llm_bound = None

    async def build_components(self, builder: Builder) -> None:
        """
        Build the components required for the selector.
        """
        self.llm_bound = await builder.get_llm(self.config.selection_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    def supported_pipeline_types(self) -> [PipelineTypeEnum]:
        return [PipelineTypeEnum.PLANNING]

    def stage_type(self) -> StageTypeEnum:
        return StageTypeEnum.SELECTION

    async def ainvoke(self,
                      items: list[TTCItem],
                      original_prompt: str | None = None,
                      agent_context: str | None = None,
                      **kwargs) -> [TTCItem]:
        """
        Select the planning items based on the selection strategy.

        Args:
            original_prompt (str): The prompt the user provided the agent.
            agent_context (str): The context of the agent, if applicable.
            items (list[TTCItem]): The list of planning items to select from.

        Returns:
            TTCItem: The selected planning item.
        """

        try:
            from langchain_core.language_models import BaseChatModel
            from langchain_core.prompts import PromptTemplate
        except ImportError:
            raise ImportError("langchain-core is not installed. Please install it to use SingleShotMultiPlanPlanner.\n"
                              "This error can be resolved by installing aiqtoolkit-langchain.")

        if not isinstance(self.llm_bound, BaseChatModel):
            raise ValueError("The `selection_llm` must be an instance of `BaseChatModel`.")

        model: BaseChatModel = self.llm_bound

        plans = ""
        for idx, item in enumerate(items):
            plans += f"{idx + 1}. {remove_r1_think_tags(item.plan)}\n"

        prompt_template = PromptTemplate(
            template=self.config.selection_template,
            input_variables=["original_prompt", "context", "plans"],
            validate_template=True,
        )

        prompt = (await prompt_template.ainvoke(input={
            "original_prompt": original_prompt, "context": agent_context, "plans": plans
        })).to_string()

        selected_plan_index = remove_r1_think_tags((await model.ainvoke(prompt)).content)

        # Model Response will be 'Plan {plan number}'
        # Use RegEx to extrac Plan {idx} from response strong
        if not isinstance(selected_plan_index, str):
            logger.warning(f"Invalid response from LLM for selected plan index: {selected_plan_index}.")
            raise ValueError("Unable to parse the selected plan index.")
        selected_plan_index = selected_plan_index.strip()
        match = re.match(r'^\s*SELECTED PLAN:\s+(\d+)', selected_plan_index)
        if not match:
            logger.warning(f"Could not parse the selected plan index from the response: {selected_plan_index}.")
            raise ValueError("The response format for selecting the plan is incorrect.")
        index = match.group(1)

        try:
            selected_index = int(index) - 1
            if selected_index < 0 or selected_index >= len(items):
                raise ValueError("Selected index is out of range.")

            # Return the selected planning item
            return [items[selected_index]]
        except ValueError as e:
            logger.warning(f"Error parsing the selected plan index: {index}. Exception: {str(e)}")
            raise ValueError(f"Failed to parse the selected plan index from the LLM response: {selected_plan_index}. "
                             "Ensure the response follows the expected format.") from e


@register_ttc_strategy(config_type=LLMBasedPlanSelectionConfig)
async def register_llm_based_plan_selection(config: LLMBasedPlanSelectionConfig, builder: Builder):
    """
    Register the LLMBasedPlanSelector with the provided configuration.
    """
    selector = LLMBasedPlanSelector(config)
    await selector.build_components(Builder())
    yield selector
