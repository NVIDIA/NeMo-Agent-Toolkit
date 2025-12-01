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

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import TTCStrategyRef
from nat.data_models.function import FunctionBaseConfig
from nat.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from nat.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from nat.experimental.test_time_compute.models.ttc_item import TTCItem

logger = logging.getLogger(__name__)


class MultiLLMJudgeFunctionConfig(FunctionBaseConfig, name="multi_llm_judge_function"):
    """
    Configuration for a function that orchestrates multi-LLM search and
    judge-based selection.
    """
    search_strategy: TTCStrategyRef = Field(description="Strategy to search/generate responses "
                                            "(e.g. multi_llm_generation)")
    selection_strategy: TTCStrategyRef = Field(description="Strategy to select the best response "
                                               "(e.g. llm_judge_selection)")


@register_function(config_type=MultiLLMJudgeFunctionConfig)
async def execute_multi_llm_judge_function(config: MultiLLMJudgeFunctionConfig, builder: Builder):
    # Resolve Strategies
    # Using CUSTOM pipeline type as this is a custom orchestration
    search_strat = await builder.get_ttc_strategy(strategy_name=config.search_strategy,
                                                  pipeline_type=PipelineTypeEnum.CUSTOM,
                                                  stage_type=StageTypeEnum.SEARCH)

    select_strat = await builder.get_ttc_strategy(strategy_name=config.selection_strategy,
                                                  pipeline_type=PipelineTypeEnum.CUSTOM,
                                                  stage_type=StageTypeEnum.SELECTION)

    async def execute_fn(user_query: str) -> str:
        logger.info("Starting Multi-LLM Judge Function execution.")

        # Step 1: Search (Generate responses)
        # Create initial item with input
        initial_items = [TTCItem(input=user_query)]

        logger.info("Executing search strategy...")
        generated_items = await search_strat.ainvoke(items=initial_items, original_prompt=user_query)

        if not generated_items:
            logger.warning("Search strategy produced no items. Returning empty string.")
            return ""

        logger.info("Generated %d responses.", len(generated_items))

        # Step 2: Selection (Judge)
        logger.info("Executing selection strategy...")
        selected_items = await select_strat.ainvoke(items=generated_items, original_prompt=user_query)

        if not selected_items:
            logger.warning("Selection strategy returned no items. "
                           "Returning first generated item.")
            return str(generated_items[0].output)

        result = str(selected_items[0].output)
        logger.info("Function execution completed.")
        return result

    yield FunctionInfo.from_fn(
        fn=execute_fn,
        description=("This function queries multiple LLMs with a user query and "
                     "uses a Judge LLM to select the best response."),
    )
