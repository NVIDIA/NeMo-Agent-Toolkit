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
from nat.experimental.test_time_compute.models.selection_config import LLMJudgeSelectionConfig
from nat.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from nat.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from nat.experimental.test_time_compute.models.strategy_base import StrategyBase
from nat.experimental.test_time_compute.models.ttc_item import TTCItem
from nat.utils.io.model_processing import remove_r1_think_tags

logger = logging.getLogger(__name__)


class LLMJudgeSelection(StrategyBase):
    """
    A selection strategy that uses a configured Judge LLM to select the best response.
    """

    def __init__(self, config: LLMJudgeSelectionConfig) -> None:
        super().__init__(config)
        self.config = config
        self.judge_llm_bound = None

    async def build_components(self, builder: Builder) -> None:
        """
        Builds the Judge LLM configured in the strategy.
        """
        logger.debug("Building components for LLMJudgeSelection")
        self.judge_llm_bound = await builder.get_llm(self.config.judge_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    def supported_pipeline_types(self) -> list[PipelineTypeEnum]:
        return [PipelineTypeEnum.CUSTOM, PipelineTypeEnum.PLANNING, PipelineTypeEnum.AGENT_EXECUTION]

    def stage_type(self) -> StageTypeEnum:
        return StageTypeEnum.SELECTION

    async def ainvoke(self,
                      items: list[TTCItem],
                      original_prompt: str | None = None,
                      agent_context: str | None = None,
                      **kwargs) -> list[TTCItem]:
        """
        Select the best item using the configured Judge LLM.
        """
        if not self.judge_llm_bound:
            raise ValueError("Judge LLM not bound. Ensure `build_components` has been called.")

        if not items:
            logger.warning("No items provided for selection.")
            return []

        try:
            from langchain_core.prompts import PromptTemplate
            from pydantic import BaseModel
        except ImportError as exc:
            raise ImportError("langchain-core is not installed.") from exc

        # Format the results for the prompt
        results_str = ""
        for idx, item in enumerate(items):
            item_output = (str(item.output.model_dump()) if isinstance(item.output, BaseModel) else str(item.output))
            results_str += f"{idx + 1}. {remove_r1_think_tags(item_output)}\n\n"

        prompt_template = PromptTemplate(
            template=self.config.selection_template,
            input_variables=["original_prompt", "results"],
            validate_template=True,
        )

        # Use input from first item if original_prompt is missing
        query = original_prompt if original_prompt else (items[0].input or "Unknown Query")

        prompt = (await prompt_template.ainvoke(input={"original_prompt": query, "results": results_str})).to_string()

        logger.info("Asking Judge LLM to select the best response.")
        judge_response = await self.judge_llm_bound.ainvoke(prompt)
        judge_content = remove_r1_think_tags(
            judge_response.content if hasattr(judge_response, 'content') else str(judge_response))

        # Parse selection
        # Expected format: 'SELECTED ITEM: <number>'
        match = re.search(r'SELECTED ITEM:\s*(\d+)', judge_content, re.IGNORECASE)
        if match:
            try:
                index = int(match.group(1)) - 1
                if 0 <= index < len(items):
                    logger.info("Judge selected item %d", index + 1)
                    selected_item = items[index]
                    # Optionally attach judge's reasoning to metadata
                    if selected_item.metadata is None:
                        selected_item.metadata = {}
                    selected_item.metadata["judge_reasoning"] = judge_content
                    return [selected_item]
                else:
                    logger.warning("Judge selected index %d which is out of range.", index + 1)
            except ValueError:
                logger.warning("Failed to parse integer from judge selection.")

        logger.warning("Could not parse valid selection from judge response. "
                       "Returning first item as fallback.")
        # Fallback to first item
        return [items[0]]


@register_ttc_strategy(config_type=LLMJudgeSelectionConfig)
async def register_llm_judge_selection(config: LLMJudgeSelectionConfig, builder: Builder):
    strategy = LLMJudgeSelection(config)
    await strategy.build_components(builder)
    yield strategy
