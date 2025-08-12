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
from nat.experimental.test_time_compute.models.editor_config import MotivationAwareSummarizationConfig
from nat.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from nat.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from nat.experimental.test_time_compute.models.strategy_base import StrategyBase
from nat.experimental.test_time_compute.models.ttc_item import TTCItem
from nat.utils.io.model_processing import remove_r1_think_tags

logger = logging.getLogger(__name__)


class MotivationAwareSummarization(StrategyBase):
    """
    A strategy that, for each incoming TTCItem, summarizes the output based on input
    and motivation.
    """

    def __init__(self, config: MotivationAwareSummarizationConfig) -> None:
        super().__init__(config)
        self.config = config
        self.llm_bound = None

    async def build_components(self, builder: Builder) -> None:
        """
        Binds each LLMRef in self.config.llms to an actual LLM client.
        """
        bound_llm = await builder.get_llm(self.config.editor_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        self.llm_bound = bound_llm

    def supported_pipeline_types(self) -> list[PipelineTypeEnum]:
        return [PipelineTypeEnum.TOOL_USE]

    def stage_type(self) -> StageTypeEnum:
        return StageTypeEnum.EDITING

    async def ainvoke(self,
                      items: list[TTCItem],
                      original_prompt: str | None = None,
                      agent_context: str | None = None,
                      **kwargs) -> list[TTCItem]:
        """
        For each TTCItem, rewrite the 'input' using each LLM to create a new perspective.
        The new TTCItems' 'output' field will store the newly generated query.
        """
        try:
            from langchain_core.prompts import PromptTemplate
        except ImportError:
            raise ImportError("langchain-core is required for MultiQueryRetrievalSearch. "
                              "Install aiqtoolkit-langchain or similar.")

        new_ttc_items: list[TTCItem] = []

        # Create a single PromptTemplate object for rewriting the query
        template_vars = ["task", "motivation", "output"]
        query_template = PromptTemplate(template=self.config.editor_template,
                                        input_variables=template_vars,
                                        validate_template=True)

        for item in items:
            original_task = str(item.input) or ""
            motivation = str(item.metadata) if item.metadata else ""
            output = str(item.output) if item.output else ""

            prompt = await (query_template.ainvoke(input={
                "task": original_task, "motivation": motivation, "output": output
            }))

            llm_response = await self.llm_bound.ainvoke(prompt.to_string())
            llm_response = remove_r1_think_tags(llm_response.content)

            logger.info("LLM response from summarization: %s", llm_response)

            new_ttc_items.append(
                TTCItem(
                    input=item.input,
                    output=remove_r1_think_tags(llm_response),
                    metadata=item.metadata,
                    name=item.name,  # keep the original tool name
                ))

        return new_ttc_items


@register_ttc_strategy(config_type=MotivationAwareSummarizationConfig)
async def register_multi_query_retrieval_search(config: MotivationAwareSummarizationConfig, builder: Builder):
    strategy = MotivationAwareSummarization(config)
    await strategy.build_components(builder)
    yield strategy
