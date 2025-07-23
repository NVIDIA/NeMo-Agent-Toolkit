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

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.component_ref import ITSStrategyRef
from aiq.data_models.function import FunctionBaseConfig
from aiq.experimental.inference_time_scaling.models.its_item import ITSItem
from aiq.experimental.inference_time_scaling.models.stage_enums import PipelineTypeEnum
from aiq.experimental.inference_time_scaling.models.stage_enums import StageTypeEnum
from aiq.experimental.inference_time_scaling.models.tool_use_config import ToolUseInputSchema
from aiq.experimental.inference_time_scaling.models.tool_use_config import ToolUselist

logger = logging.getLogger(__name__)


class ITSToolOrchestrationFunctionConfig(FunctionBaseConfig, name="its_tool_orchestration"):
    """
    Configuration for the ITSToolOrchestrationFunction, which is used to orchestrate multiple functions.
    """

    augmented_fns: list[FunctionRef] = Field(
        description="list of FunctionRefs for the functions to be orchestrated. Must be wrapped in `its_tool_wrapper`.")

    search_strategy: ITSStrategyRef | None = Field(
        description="The ITS search strategy to use for orchestrating invocation of the functions."
        " If None, no search will be performed.",
        default=None,
    )

    editing_strategy: ITSStrategyRef | None = Field(
        default=None,
        description="The ITS editing strategy to use for orchestrating invocation of the functions. "
        "If None, no editing will be performed.",
    )

    scoring_strategy: ITSStrategyRef | None = Field(
        default=None,
        description="The ITS scoring strategy to use for orchestrating invocation of the functions. "
        "If None, no scoring will be performed.",
    )

    selection_strategy: ITSStrategyRef = Field(
        description="The ITS selection strategy to use for orchestrating invocation of the functions.")


@register_function(config_type=ITSToolOrchestrationFunctionConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def register_its_tool_orchestration_function(
    config: ITSToolOrchestrationFunctionConfig,
    builder: Builder,
):
    """
    Registers an ITS-based orchestration function that:
     1. Instantiates all relevant strategies (search, editing, scoring, selection).
     2. Accepts a ToolUselist, converts each item to an ITSItem, optionally runs search/editing.
     3. Calls the correct augmented_fn per item using name=tool name.
     4. If configured, runs scoring and selection on the result.
     5. Returns a new ToolUselist with each output set.
    """

    # 1) Gather references to all augmented (wrapped) functions
    function_map = {}
    for fn_ref in config.augmented_fns:
        # Retrieve the actual function from the builder
        fn_obj = builder.get_function(fn_ref)
        function_map[fn_ref] = fn_obj

    # 2) Instantiate search, editing, scoring, selection strategies (if any)
    search = None
    if config.search_strategy is not None:
        search = await builder.get_its_strategy(
            strategy_name=config.search_strategy,
            pipeline_type=PipelineTypeEnum.TOOL_USE,
            stage_type=StageTypeEnum.SEARCH,
        )

    editing = None
    if config.editing_strategy is not None:
        editing = await builder.get_its_strategy(
            strategy_name=config.editing_strategy,
            pipeline_type=PipelineTypeEnum.TOOL_USE,
            stage_type=StageTypeEnum.EDITING,
        )

    scoring = None
    if config.scoring_strategy is not None:
        scoring = await builder.get_its_strategy(
            strategy_name=config.scoring_strategy,
            pipeline_type=PipelineTypeEnum.TOOL_USE,
            stage_type=StageTypeEnum.SCORING,
        )

    selection = await builder.get_its_strategy(
        strategy_name=config.selection_strategy,
        pipeline_type=PipelineTypeEnum.TOOL_USE,
        stage_type=StageTypeEnum.SELECTION,
    )

    fn_description = ("\n".join(f"- **{fn_ref}**: {function_map[fn_ref].description or 'No description provided.'}"
                                for fn_ref in config.augmented_fns))

    # 3) Create the inner function to handle single (non-streaming) calls.
    async def single_inner(tool_list: ToolUselist) -> ToolUselist:
        """
        Orchestrates multiple tool usages, optionally using search/editing/scoring/selection steps.
        """
        # Convert each ToolUseInputSchema to ITSItem
        its_items = []
        for t in tool_list.tools:
            item = ITSItem(
                input=t.task_description,  # The user "task"
                output=None,
                name=t.tool_name,  # The "tool name"
                metadata=t.motivation,  # The "justification"
            )
            its_items.append(item)

        # Run search strategy if present
        if search is not None:
            its_items = await search.ainvoke(its_items)

        logger.info("ITS orchestration function: %d items after search", len(its_items))

        # Invoke the correct augmented function for each item concurrently
        # Helper coroutine to invoke a tool function and capture result or error
        async def _invoke_tool(item: ITSItem, fn):
            try:
                result = await fn.acall_invoke(item.output)
                return item, result, None
            except Exception as e:
                logger.error(f"Error invoking function '{item.name}': {e}")
                return item, None, str(e)

        tasks = []
        for item in its_items:
            if item.name not in function_map:
                logger.error(f"Function '{item.name}' not found in function map.")
                item.output = f"Error: Function '{item.name}' not found in function map. Check your input"
            else:
                fn = function_map[item.name]
                tasks.append(_invoke_tool(item, fn))

        # Await all tasks and assign outputs
        if tasks:
            results = await asyncio.gather(*tasks)
            for item, result, error in results:
                if error:
                    item.output = f"Error invoking function '{item.name}': {error}"
                else:
                    item.output = result

        if editing:
            its_items = await editing.ainvoke(its_items)

        # Run scoring strategy if present
        if scoring is not None:
            its_items = await scoring.ainvoke(its_items)

        # Run selection strategy
        if selection is not None:
            its_items = await selection.ainvoke(its_items)

        logger.info("ITS orchestration function: %d items after selection", len(its_items))

        # Convert final results from ITSItems back to a ToolUselist
        final_list = ToolUselist(tools=[])
        for item in its_items:
            # Compose a new ToolUseInputSchema with final output
            new_tool = ToolUseInputSchema(
                tool_name=item.name,
                task_description=str(item.input),
                motivation=item.metadata if item.metadata else None,
                output=str(item.output) if item.output is not None else None,
            )
            final_list.tools.append(new_tool)

        return final_list

    # 4) Return the function info (only a single_fn is needed; no streaming)
    yield FunctionInfo.create(
        single_fn=single_inner,
        stream_fn=None,  # No streaming required
        input_schema=ToolUselist,
        single_output_schema=ToolUselist,
        description=fn_description)
