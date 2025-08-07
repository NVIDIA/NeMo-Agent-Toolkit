# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.component_ref import GuardrailsRef
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class AgentWithRailsConfig(FunctionBaseConfig, name="agent_with_rails"):
    """Configuration for Safe Information Lookup with guardrails."""

    tool_names: list[FunctionRef] = Field(description="Search tools to use")
    llm_name: LLMRef = Field(description="LLM for enhancing responses")
    guardrails_name: GuardrailsRef = Field(description="Guardrails to apply")
    description: str = Field(default="Safe information lookup with guardrails",
                             description="Description of the function")


@register_function(config_type=AgentWithRailsConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def agent_with_rails(config: AgentWithRailsConfig, builder: Builder):
    """
    A simple agent with rails example

    A straightforward information lookup agent that:
    1. Blocks harmful queries with input guardrails
    2. Searches Wikipedia for safe queries
    3. Summarizes results with LLM
    4. Applies output guardrails
    """

    # Get components from builder
    guardrails = await builder.get_guardrails(config.guardrails_name)
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    tools = builder.get_tools(config.tool_names, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    search_tool = tools[0] if tools else None

    logger.debug("Loaded guardrails: %s", config.guardrails_name)
    logger.debug("Loaded search tool: %s", config.tool_names[0] if config.tool_names else 'None')

    async def _agent_with_rails(query: str) -> str:
        """
        A simple agent with rails example
        """
        try:
            logger.debug("Processing query: '%s'", query)

            # Apply input guardrails
            processed_input, should_continue = await guardrails.apply_input_guardrails(query)

            if not should_continue:
                logger.warning("Input blocked by guardrails")
                fallback = getattr(guardrails.config, 'fallback_response', "I cannot help with that request.")
                return fallback

            logger.debug("Input passed safety check")

            # Search for information
            if not search_tool:
                return "Sorry, no search tool is available."

            try:
                search_results = await search_tool.ainvoke(str(processed_input))
                logger.debug("Wikipedia search completed")
            except Exception as e:
                logger.error("Search failed: %s", e)
                return "I encountered an error while searching for information."

            # Summarize results with LLM
            try:
                enhancement_prompt = f"""Based on this Wikipedia information about "{processed_input}":

                                        {search_results}

                                        Please provide a clear, helpful summary in 2-3 sentences.
                                        Focus on the most important and interesting facts."""

                response = await llm.ainvoke([{"role": "user", "content": enhancement_prompt}])
                enhanced_response = response.content if hasattr(response, 'content') else str(response)
                logger.debug("LLM enhancement completed")
            except Exception as e:
                logger.error("LLM enhancement failed: %s", e)
                # Fallback to raw search results
                enhanced_response = f"Here's what I found: {search_results}"

            # Apply output guardrails
            final_result = await guardrails.apply_output_guardrails(enhanced_response, processed_input)
            logger.debug("Output passed safety check")

            return str(final_result)

        except Exception as e:
            logger.error("Unexpected error: %s", e)
            return "I'm sorry, I encountered an unexpected error while processing your request."

    try:
        yield FunctionInfo.from_fn(_agent_with_rails, description=config.description)
    except GeneratorExit:
        logger.debug("Exited early")
    finally:
        logger.debug("Cleaning up")
