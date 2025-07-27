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
from aiq.builder.context import AIQContext
from aiq.builder.function_info import FunctionInfo
from aiq.builder.workflow_builder import WorkflowBuilder
from aiq.cli.register_workflow import register_function
from aiq.data_models.api_server import AIQChatRequest
from aiq.data_models.api_server import AIQChatResponse
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.function import FunctionBaseConfig
from aiq.data_models.interactive import HumanPromptText
from aiq.data_models.interactive import InteractionResponse

logger = logging.getLogger(__name__)


class RetryReactAgentConfig(FunctionBaseConfig, name="retry_react_agent"):
    """
    This function creates a wrapper around a React agent that can automatically
    retry failed attempts due to recursion limits. It uses human-in-the-loop
    approval to get permission before retrying with increased max_iterations.

    This is particularly useful for complex reasoning tasks where the agent might need
    more iterations to complete successfully.
    """

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

    from aiq.builder.function import Function

    # Get references to the underlying React agent and approval function
    react_agent: Function = builder.get_function(config.react_agent_fn)
    react_agent_config: FunctionBaseConfig = builder.get_function_config(
        config.react_agent_fn)  # ReActAgentWorkflowConfig
    hitl_approval_fn: Function = builder.get_function(config.hitl_approval_fn)

    # Regex pattern to detect GraphRecursionError message
    # This pattern matches the specific error message format from LangGraph
    recursion_error_pattern = re.compile(r"Recursion limit of \d+ reached without hitting a stop condition\. "
                                         r"You can increase the limit by setting the `recursion_limit` config key\.")

    def is_recursion_error(response_content: str) -> bool:
        """
        Check if the response content contains a recursion error message.

        Args:
            response_content: The response content to check

        Returns:
            bool: True if the response contains a recursion error message
        """
        if isinstance(response_content, str):
            return bool(recursion_error_pattern.search(response_content))
        return False

    async def get_temp_react_agent(original_config: RetryReactAgentConfig,
                                   retry_config: RetryReactAgentConfig) -> tuple[Function, FunctionBaseConfig]:
        """
        Create a temporary React agent instance for retry attempts.

        This function creates a new React agent with the same configuration as the original,
        but allows for modification of parameters (like max_iterations) during retries.

        Args:
            original_config: Configuration of the original React agent
            retry_config: Configuration for the retry mechanism

        Returns:
            tuple: A tuple containing the temporary React agent function and its config
        """
        async with WorkflowBuilder() as temp_builder:
            # Add the LLM needed by the react agent
            original_llm_config = builder.get_llm_config(original_config.llm_name)
            await temp_builder.add_llm(original_config.llm_name, original_llm_config)

            # Add any tools needed by the react agent
            # This ensures the temporary agent has access to all the same tools
            for tool_name in original_config.tool_names:
                tool_config = builder.get_function_config(tool_name)
                await temp_builder.add_function(tool_name, tool_config)

            # Create the retry agent with the original configuration
            temp_retry_agent = await temp_builder.add_function("retry_agent", retry_config)

            return temp_retry_agent, retry_config

    async def handle_recursion_error(input_message: AIQChatRequest) -> AIQChatResponse:
        """
        Handle recursion errors by retrying with increased max_iterations.

        This function implements the core retry logic:
        1. Creates a temporary React agent
        2. Progressively increases max_iterations on each retry
        3. Attempts up to max_retries times
        4. Asks for human approval before each retry

        Args:
            input_message: The original input message to process

        Returns:
            AIQChatResponse: The response from the successful retry or error message
        """
        temp_react_agent: Function
        temp_react_agent_config: RetryReactAgentConfig
        temp_react_agent, temp_react_agent_config = await get_temp_react_agent(
            react_agent_config, react_agent_config.model_copy(deep=True))  # type: ignore

        # Attempt retries up to the configured maximum
        for attempt in range(config.max_retries):
            try:
                # Increase max_iterations for this retry attempt
                updated_max_iterations = temp_react_agent_config.max_iterations + config.max_iterations_increment
                logger.info("Attempt %d: Increasing max_iterations to %d", attempt + 2, updated_max_iterations)
                temp_react_agent_config.max_iterations += config.max_iterations_increment

                # Try to execute the agent with increased iterations
                response = await temp_react_agent.acall_invoke(input_message)

                # Check if we still got a recursion error
                if is_recursion_error(response):
                    raise GraphRecursionError(response)

                # Success! Return the response
                return response

            except GraphRecursionError:
                # Log the recursion error and ask for human approval to continue
                logger.info("Recursion error detected, prompting user to increase recursion limit")
                selected_option = await hitl_approval_fn.acall_invoke()

                # If user doesn't approve, return error message
                if not selected_option:
                    return AIQChatResponse.from_string("I seem to be having a problem.")

        # If we exhausted all retries, return the last response
        return response

    async def _response_fn(input_message: AIQChatRequest) -> AIQChatResponse:
        """
        Main response function that handles the initial attempt and retry logic.

        This function:
        1. First tries the original React agent
        2. If it encounters a recursion error, asks for human approval
        3. If approved, delegates to the retry handler
        4. Handles any other exceptions gracefully

        Args:
            input_message: The input message to process

        Returns:
            AIQChatResponse: The response from the agent or error message
        """
        try:
            # First attempt: try the original React agent
            response = await react_agent.acall_invoke(input_message)

            # Check if we got a recursion error
            if is_recursion_error(response):
                raise GraphRecursionError(response)

            return response  # type: ignore

        except GraphRecursionError:
            # Recursion error detected - ask for human approval before retrying
            logger.info("Recursion error detected, prompting user to increase recursion limit")
            selected_option = await hitl_approval_fn.acall_invoke()

            if selected_option:
                # User approved - proceed with retry logic
                return await handle_recursion_error(input_message)

            # User declined - return error message
            return AIQChatResponse.from_string("I seem to be having a problem.")

        except Exception:
            # Handle any other unexpected exceptions
            return AIQChatResponse.from_string("I seem to be having a problem.")

    yield FunctionInfo.from_fn(_response_fn, description=config.description)


class TimeZonePromptConfig(FunctionBaseConfig, name="time_zone_prompt"):
    pass


@register_function(config_type=TimeZonePromptConfig)
async def time_zone_prompt(config: TimeZonePromptConfig, builder: Builder):

    async def _response_fn(empty: None) -> str:

        response: InteractionResponse = await AIQContext.get().user_interaction_manager.prompt_user_input(
            HumanPromptText(text="What is the current time in the user's timezone?", required=True, placeholder=""))

        return response.content.text

    yield FunctionInfo.from_fn(_response_fn, description="Prompt the user for their time zone")
