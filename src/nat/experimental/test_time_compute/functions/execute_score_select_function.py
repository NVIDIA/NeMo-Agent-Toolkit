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
from nat.builder.function import Function
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import TTCStrategyRef
from nat.data_models.function import FunctionBaseConfig
from nat.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from nat.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from nat.experimental.test_time_compute.models.ttc_item import TTCItem

logger = logging.getLogger(__name__)


class ExecuteScoreSelectFunctionConfig(FunctionBaseConfig, name="execute_score_select_function"):
    scorer: TTCStrategyRef | None = Field(description="Strategy to score the output of the function", default=None)
    selector: TTCStrategyRef = Field(description="Strategy to select the best output of the function")
    augmented_fn: FunctionRef = Field(description="Function that will be executed")

    num_executions: int = Field(3, description="Number of times to execute the function")


@register_function(config_type=ExecuteScoreSelectFunctionConfig)
async def execute_score_select_function(config: ExecuteScoreSelectFunctionConfig, builder: Builder):
    import asyncio
    import warnings

    from pydantic import BaseModel

    executable_fn: Function = await builder.get_function(name=config.augmented_fn)

    if config.scorer:
        scorer = await builder.get_ttc_strategy(strategy_name=config.scorer,
                                                pipeline_type=PipelineTypeEnum.AGENT_EXECUTION,
                                                stage_type=StageTypeEnum.SCORING)
    else:
        scorer = None

    selector = await builder.get_ttc_strategy(strategy_name=config.selector,
                                              pipeline_type=PipelineTypeEnum.AGENT_EXECUTION,
                                              stage_type=StageTypeEnum.SELECTION)

    if executable_fn.has_streaming_output:
        warnings.warn("Streaming output is not supported for this function. "
                      "The function will be executed in non-streaming mode.")

    def convert_to_str(arg):
        if isinstance(arg, BaseModel):
            return str(arg.model_dump())
        return str(arg)

    async def execute_fn(input_msg: executable_fn.input_type) -> executable_fn.single_output_type:

        logger.info("Executing function %d times", config.num_executions)
        tasks = [executable_fn.ainvoke(input_msg) for _ in range(config.num_executions)]
        results = await asyncio.gather(*tasks)

        input_str = convert_to_str(input_msg)
        function_outputs = [convert_to_str(out) for out in results]
        its_items = [TTCItem(
            input=input_str,
            output=out,
        ) for out in function_outputs]

        if scorer:
            logger.info("Beginning scoring")
            its_items = await scorer.ainvoke(items=its_items)

        logger.info("Beginning selection")
        selected_item = (await selector.ainvoke(items=its_items, original_prompt=its_items[0].input))[0]

        # Find the index of selected item in its_items by matching the output
        selected_output = selected_item.output
        selected_index = -1
        for i, item in enumerate(its_items):
            if item.output == selected_output:
                selected_index = i
                break

        return results[selected_index] if selected_index != -1 else selected_output

    yield FunctionInfo.from_fn(
        fn=execute_fn,
        description=("This function executes a given function multiple times, scores the outputs, "
                     "and selects the best output based on the specified scoring and selection strategies."),
    )
