# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.builder.workflow_builder import WorkflowBuilder
from aiq.cli.register_workflow import register_function
from aiq.data_models.api_server import AIQChatRequest
from aiq.data_models.api_server import AIQChatResponse
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.function import FunctionBaseConfig
from aiq.utils.type_converter import GlobalTypeConverter

logger = logging.getLogger(__name__)


class RetryReactAgentConfig(FunctionBaseConfig, name="retry_react_agent"):
    use_openai_api: bool = Field(default=False, description="Whether to use the OpenAI API")
    max_retries: int = Field(default=3, description="Maximum number of retry attempts")
    max_iterations_increment: int = Field(default=1, description="How much to increase max_iterations on each retry")
    description: str = Field(default="Retry React Agent",
                             description="This agent retries the react agent with an increasing number of iterations.")
    hitl_approval_fn: FunctionRef = Field(..., description="The hitl approval function")
    react_agent_fn: FunctionRef = Field(..., description="The react agent to retry")


@register_function(config_type=RetryReactAgentConfig)
async def retry_react_agent(config: RetryReactAgentConfig, builder: Builder):
    import re

    from langgraph.errors import GraphRecursionError

    from aiq.agent.react_agent.register import ReActAgentWorkflowConfig
    from aiq.builder.function import Function

    react_agent: Function = builder.get_function(config.react_agent_fn)
    react_agent_config: ReActAgentWorkflowConfig = builder.get_function_config(config.react_agent_fn)
    hitl_approval_fn: Function = builder.get_function(config.hitl_approval_fn)

    # Regex pattern to detect GraphRecursionError message
    recursion_error_pattern = re.compile(r"Recursion limit of \d+ reached without hitting a stop condition\. "
                                         r"You can increase the limit by setting the `recursion_limit` config key\.")

    def is_recursion_error(response_content: str) -> bool:
        if isinstance(response_content, str):
            return bool(recursion_error_pattern.search(response_content))
        return False

    async def get_temp_react_agent(original_config: ReActAgentWorkflowConfig,
                                   retry_config: RetryReactAgentConfig) -> tuple[Function, FunctionBaseConfig]:

        async with WorkflowBuilder() as temp_builder:
            # Add the LLM needed by the react agent
            original_llm_config = builder.get_llm_config(original_config.llm_name)
            await temp_builder.add_llm(original_config.llm_name, original_llm_config)

            # Add any tools needed by the react agent
            for tool_name in original_config.tool_names:
                tool_config = builder.get_function_config(tool_name)
                await temp_builder.add_function(tool_name, tool_config)

            # Create the retry agent
            retry_agent = await temp_builder.add_function("retry_agent", retry_config)
            retry_agent_config = temp_builder.get_function_config("retry_agent")

            return retry_agent, retry_agent_config

    async def handle_recursion_error(input_message: AIQChatRequest) -> AIQChatResponse:
        temp_react_agent: Function
        temp_react_agent_config: ReActAgentWorkflowConfig
        temp_react_agent, temp_react_agent_config = await get_temp_react_agent(
            react_agent_config, react_agent_config.model_copy(deep=True))  # type: ignore

        for attempt in range(config.max_retries):
            try:
                updated_max_iterations = temp_react_agent_config.max_iterations + config.max_iterations_increment
                logger.info("Attempt %d: Increasing max_iterations to %d", attempt + 2, updated_max_iterations)
                temp_react_agent_config.max_iterations += config.max_iterations_increment
                response = await temp_react_agent.acall_invoke(input_message)
                if is_recursion_error(response):
                    raise GraphRecursionError(response)
                return response
            except GraphRecursionError:
                logger.info("Recursion error detected, prompting user to increase recursion limit")
                selected_option = await hitl_approval_fn.acall_invoke()
                if selected_option:
                    continue

        return response

    async def _response_fn(input_message: AIQChatRequest) -> AIQChatResponse:

        try:
            response = await react_agent.acall_invoke(input_message)

            if is_recursion_error(response):
                raise GraphRecursionError(response)
            return response  # type: ignore

        except GraphRecursionError:
            logger.info("Recursion error detected, prompting user to increase recursion limit")
            selected_option = await hitl_approval_fn.acall_invoke()
            if selected_option:
                return await handle_recursion_error(input_message)
            return AIQChatResponse.from_string("I seem to be having a problem.")

        except Exception:
            return AIQChatResponse.from_string("I seem to be having a problem.")

    if (config.use_openai_api):
        yield FunctionInfo.from_fn(_response_fn, description=config.description)
    else:

        async def _str_api_fn(input_message: str) -> str:
            oai_input = GlobalTypeConverter.get().convert(input_message, to_type=AIQChatRequest)

            oai_output = await _response_fn(oai_input)

            return GlobalTypeConverter.get().convert(oai_output, to_type=str)

        yield FunctionInfo.from_fn(_str_api_fn, description=config.description)
