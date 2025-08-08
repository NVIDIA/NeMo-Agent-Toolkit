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

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_ttc_strategy
from aiq.experimental.test_time_compute.models.search_config import MultiQueryRetrievalSearchConfig
from aiq.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from aiq.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from aiq.experimental.test_time_compute.models.strategy_base import StrategyBase
from aiq.experimental.test_time_compute.models.ttc_item import TTCItem
from aiq.utils.io.model_processing import remove_r1_think_tags

logger = logging.getLogger(__name__)


class MultiQueryRetrievalSearch(StrategyBase):
    """
    A strategy that, for each incoming TTCItem, generates multiple new items by
    re-writing the input 'task_description' from different perspectives.
    Uses multiple LLMs to encourage diversity.
    """

    def __init__(self, config: MultiQueryRetrievalSearchConfig) -> None:
        super().__init__(config)
        self.config = config
        self.llms_bound = []

    async def build_components(self, builder: Builder) -> None:
        """
        Binds each LLMRef in self.config.llms to an actual LLM client.
        """
        self.llms_bound = []
        for llm_ref in self.config.llms:
            bound_llm = await builder.get_llm(llm_ref, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
            self.llms_bound.append(bound_llm)

    def supported_pipeline_types(self) -> list[PipelineTypeEnum]:
        return [PipelineTypeEnum.TOOL_USE]

    def stage_type(self) -> StageTypeEnum:
        return StageTypeEnum.SEARCH

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
        template_vars = ["task", "motivation"]
        query_template = PromptTemplate(template=self.config.query_generation_template,
                                        input_variables=template_vars,
                                        validate_template=True)

        for item in items:
            original_task = str(item.input) or ""
            motivation = str(item.metadata) if item.metadata else ""
            new_ttc_items.append(
                TTCItem(
                    input=item.input,
                    output=item.input,
                    metadata=item.metadata,
                    name=item.name,  # keep the original tool name
                ))

            for llm in self.llms_bound:
                prompt_str = (await query_template.ainvoke({
                    "task": original_task, "motivation": motivation
                })).to_string()

                # We'll call each LLM to produce a new query
                response = await llm.ainvoke(prompt_str)
                cleaned = remove_r1_think_tags(response.content if hasattr(response, 'content') else str(response))
                cleaned = cleaned.strip()

                # Create a new TTCItem for each newly generated query
                new_item = TTCItem(
                    input=item.input,  # keep the original input for reference
                    output=cleaned,  # store the newly generated query in the output
                    metadata=item.metadata,
                    name=item.name,  # same tool name or optional new name
                )
                new_ttc_items.append(new_item)

        logger.info("MultiQueryRetrievalSearch produced %d new items from %d original items.",
                    len(new_ttc_items),
                    len(items))

        return new_ttc_items


@register_ttc_strategy(config_type=MultiQueryRetrievalSearchConfig)
async def register_multi_query_retrieval_search(config: MultiQueryRetrievalSearchConfig, builder: Builder):
    strategy = MultiQueryRetrievalSearch(config)
    await strategy.build_components(builder)
    yield strategy
