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

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_ttc_strategy
from nat.data_models.ttc_strategy import TTCStrategyBaseConfig
from nat.experimental.test_time_compute.models.selection_config import LLMBasedOutputMergingConfig
from nat.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from nat.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from nat.experimental.test_time_compute.models.strategy_base import StrategyBase
from nat.experimental.test_time_compute.models.ttc_item import TTCItem
from nat.utils.io.model_processing import remove_r1_think_tags

logger = logging.getLogger(__name__)


class LLMBasedOutputMergingSelector(StrategyBase):

    def __init__(self, config: TTCStrategyBaseConfig) -> None:
        super().__init__(config)
        self.llm_bound = None

    async def build_components(self, builder: Builder) -> None:
        """
        Build the components required for the selector.
        """
        self.llm_bound = await builder.get_llm(self.config.selection_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    def supported_pipeline_types(self) -> [PipelineTypeEnum]:
        return [PipelineTypeEnum.AGENT_EXECUTION, PipelineTypeEnum.PLANNING]

    def stage_type(self) -> StageTypeEnum:
        return StageTypeEnum.SELECTION

    async def ainvoke(self,
                      items: list[TTCItem],
                      original_prompt: str | None = None,
                      agent_context: str | None = None,
                      **kwargs) -> [TTCItem]:
        """
        Merge the outputs of multiple planning items into a single output

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
                              "This error can be resolved by installing nvidia-nat-langchain.")

        from typing import Callable

        from pydantic import BaseModel

        if not isinstance(self.llm_bound, BaseChatModel):
            raise ValueError("The `selection_llm` must be an instance of `BaseChatModel`.")

        if not self.pipeline_type:
            raise RuntimeError("Pipeline type is not set. Ensure that the pipeline "
                               "type is set before invoking the selector.")

        model: BaseChatModel = self.llm_bound

        results = ""
        if self.pipeline_type == PipelineTypeEnum.AGENT_EXECUTION:
            for idx, item in enumerate(items):
                item_str = str(item.output.model_dump()) if isinstance(item.output, BaseModel) else str(item.output)
                results += f"{idx + 1}. {remove_r1_think_tags(item_str)}\n\n"
        else:
            for idx, item in enumerate(items):
                item_str = str(item.plan)
                results += f"{idx + 1}. {remove_r1_think_tags(item_str)}\n\n"

        prompt_template = PromptTemplate(
            template=self.config.selection_template,
            input_variables=["pipeline_type", "objective", "input", "results"],
            validate_template=True,
        )

        if self.pipeline_type == PipelineTypeEnum.PLANNING:
            pipeline_objective = "execution plans for a given objective and input."
        else:
            pipeline_objective = "outputs from an agent system based on the provided objective and input."

        prompt = (await prompt_template.ainvoke(
            input={
                "objective": agent_context,
                "input": original_prompt,
                "results": results,
                "pipeline_type": pipeline_objective
            })).to_string()

        merged_output = remove_r1_think_tags((await model.ainvoke(prompt)).content)

        if not isinstance(merged_output, str):
            logger.warning(f"Invalid response from LLM for merged_plan: {merged_output}.")
            raise ValueError("Unable to parse merged plan.")
        merged_output = merged_output.strip()

        # match = split the string after 'MERGED OUTPUT:'
        matches = merged_output.split("MERGED OUTPUT:")
        if len(matches) > 1:
            merged_output = matches[-1].strip()
        else:
            raise ValueError("Merged output does not contain 'MERGED OUTPUT:' prefix.")

        # Check if a callable argument is provided in kwargs called output_parser
        output_parser: Callable | None = kwargs.get('output_parser', None)
        if output_parser:
            try:
                merged_output = output_parser(merged_output)
            except Exception as e:
                logger.error(f"Error parsing merged output: {e}")
                raise ValueError("Failed to parse merged output.")

        logger.info("Merged output: %s", str(merged_output))

        # Create a new TTCItem with the merged plan or output
        if self.pipeline_type == PipelineTypeEnum.PLANNING:
            merged_item = TTCItem(input=items[0].input, output=merged_output, plan=merged_output)
        else:
            merged_item = TTCItem(input=items[0].input, output=merged_output)

        return [merged_item]


@register_ttc_strategy(config_type=LLMBasedOutputMergingConfig)
async def register_llm_based_output_merging_selector(config: LLMBasedOutputMergingConfig, builder: Builder):
    """
    Register the LLMBasedOutputMergingSelector with the builder.
    """
    selector = LLMBasedOutputMergingSelector(config)
    await selector.build_components(builder)
    yield selector
