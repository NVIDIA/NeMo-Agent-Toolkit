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

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_ttc_strategy
from nat.experimental.test_time_compute.models.search_config import MultiLLMGenerationConfig
from nat.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from nat.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from nat.experimental.test_time_compute.models.strategy_base import StrategyBase
from nat.experimental.test_time_compute.models.ttc_item import TTCItem
from nat.utils.io.model_processing import remove_r1_think_tags

logger = logging.getLogger(__name__)


class MultiLLMGeneration(StrategyBase):
    """
    A search strategy that uses multiple configured LLMs to generate responses.
    """

    def __init__(self, config: MultiLLMGenerationConfig) -> None:
        super().__init__(config)
        self.config = config
        self.llms_bound = []

    async def build_components(self, builder: Builder) -> None:
        """
        Builds the LLMs configured in the strategy.
        """
        logger.debug("Building components for MultiLLMGeneration")
        self.llms_bound = []
        for llm_ref in self.config.llms:
            bound_llm = await builder.get_llm(llm_ref, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
            self.llms_bound.append(bound_llm)

    def supported_pipeline_types(self) -> list[PipelineTypeEnum]:
        return [PipelineTypeEnum.CUSTOM, PipelineTypeEnum.PLANNING, PipelineTypeEnum.AGENT_EXECUTION]

    def stage_type(self) -> StageTypeEnum:
        return StageTypeEnum.SEARCH

    async def _generate_response(self, llm, prompt: str) -> TTCItem:
        try:
            response = await llm.ainvoke(prompt)
            content = (response.content if hasattr(response, 'content') else str(response))
            cleaned = remove_r1_think_tags(content)
            return TTCItem(output=cleaned, metadata={"model": getattr(llm, "model_name", "unknown")})
        except Exception as exc:
            logger.error("Error generating response from LLM: %s", exc)
            return TTCItem(output=f"Error: {str(exc)}", metadata={"error": str(exc)})

    async def ainvoke(self,
                      items: list[TTCItem],
                      original_prompt: str | None = None,
                      agent_context: str | None = None,
                      **kwargs) -> list[TTCItem]:
        """
        Generate responses using the configured LLMs.
        """
        if not self.llms_bound:
            raise ValueError("No LLMs bound. Ensure `build_components` has been called.")

        try:
            from langchain_core.prompts import PromptTemplate
        except ImportError as exc:
            raise ImportError("langchain-core is not installed.") from exc

        # Use original_prompt if available, otherwise try to get from items
        if not original_prompt and items and items[0].input:
            original_prompt = items[0].input

        if not original_prompt:
            logger.warning("No prompt provided for generation.")
            return []

        prompt_template = PromptTemplate(template=self.config.generation_template,
                                         input_variables=["prompt"],
                                         validate_template=True)

        formatted_prompt = (await prompt_template.ainvoke({"prompt": original_prompt})).to_string()

        logger.info("Generating responses using %d LLMs.", len(self.llms_bound))
        tasks = [self._generate_response(llm, formatted_prompt) for llm in self.llms_bound]
        results = await asyncio.gather(*tasks)

        # If we have input items, we might want to attach the new outputs to them
        # or create new items. Since search usually expands, we return the new
        # items. We'll ensure the input is preserved.
        for res in results:
            res.input = original_prompt

        return results


@register_ttc_strategy(config_type=MultiLLMGenerationConfig)
async def register_multi_llm_generation(config: MultiLLMGenerationConfig, builder: Builder):
    strategy = MultiLLMGeneration(config)
    await strategy.build_components(builder)
    yield strategy
