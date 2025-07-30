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
import json
import logging
from abc import ABC
from abc import abstractmethod
from enum import Enum
from typing import Any

from colorama import Fore
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph.graph import CompiledGraph

logger = logging.getLogger(__name__)

TOOL_NOT_FOUND_ERROR_MESSAGE = "There is no tool named {tool_name}. Tool must be one of {tools}."
INPUT_SCHEMA_MESSAGE = ". Arguments must be provided as a valid JSON object following this format: {schema}"
NO_INPUT_ERROR_MESSAGE = "No human input received to the agent, Please ask a valid question."

AGENT_LOG_PREFIX = "[AGENT]"
AGENT_CALL_LOG_MESSAGE = f"\n{'-' * 30}\n" + \
                                 AGENT_LOG_PREFIX + "\n" + \
                                 Fore.YELLOW + \
                                 "Agent input: %s\n" + \
                                 Fore.CYAN + \
                                 "Agent's thoughts: \n%s" + \
                                 Fore.RESET + \
                                 f"\n{'-' * 30}"

TOOL_CALL_LOG_MESSAGE = f"\n{'-' * 30}\n" + \
                                 AGENT_LOG_PREFIX + "\n" + \
                                 Fore.WHITE + \
                                 "Calling tools: %s\n" + \
                                 Fore.YELLOW + \
                                 "Tool's input: %s\n" + \
                                 Fore.CYAN + \
                                 "Tool's response: \n%s" + \
                                 Fore.RESET + \
                                 f"\n{'-' * 30}"


class AgentDecision(Enum):
    TOOL = "tool"
    END = "finished"


class BaseAgent(ABC):

    def __init__(self,
                 llm: BaseChatModel,
                 tools: list[BaseTool],
                 callbacks: list[AsyncCallbackHandler] | None = None,
                 detailed_logs: bool = False) -> None:
        logger.debug("Initializing Agent Graph")
        self.llm = llm
        self.tools = tools
        self.callbacks = callbacks or []
        self.detailed_logs = detailed_logs
        self.graph = None

    async def _stream_llm(self,
                          runnable: Any,
                          inputs: dict[str, Any],
                          config: RunnableConfig | None = None) -> AIMessage:
        """
        Stream from LLM runnable. Retry logic is handled automatically by the underlying LLM client.

        Parameters
        ----------
        runnable : Any
            The LLM runnable (prompt | llm or similar)
        inputs : Dict[str, Any]
            The inputs to pass to the runnable
        config : RunnableConfig | None
            The config to pass to the runnable (should include callbacks)

        Returns
        -------
        AIMessage
            The LLM response
        """
        output_message = ""
        async for event in runnable.astream(inputs, config=config):
            output_message += event.content

        return AIMessage(content=output_message)

    async def _call_llm(self, messages: list[BaseMessage]) -> AIMessage:
        """
        Call the LLM directly. Retry logic is handled automatically by the underlying LLM client.

        Parameters
        ----------
        messages : list[BaseMessage]
            The messages to send to the LLM

        Returns
        -------
        AIMessage
            The LLM response
        """
        response = await self.llm.ainvoke(messages)
        return AIMessage(content=str(response.content))

    async def _call_tool(self,
                         tool: BaseTool,
                         tool_input: dict[str, Any] | str,
                         config: RunnableConfig | None = None,
                         max_retries: int = 3) -> ToolMessage:
        """
        Call a tool with retry logic and error handling.

        Parameters
        ----------
        tool : BaseTool
            The tool to call
        tool_input : Union[Dict[str, Any], str]
            The input to pass to the tool
        config : RunnableConfig | None
            The config to pass to the tool
        max_retries : int
            Maximum number of retry attempts (default: 3)

        Returns
        -------
        ToolMessage
            The tool response
        """
        last_exception = None

        for attempt in range(1, max_retries + 1):
            try:
                response = await tool.ainvoke(tool_input, config=config)

                # Handle empty responses
                if response is None or (isinstance(response, str) and response == ""):
                    return ToolMessage(name=tool.name,
                                       tool_call_id=tool.name,
                                       content=f"The tool {tool.name} provided an empty response.")

                return ToolMessage(name=tool.name, tool_call_id=tool.name, content=response)

            except Exception as e:
                last_exception = e

                # If this was the last attempt, don't sleep
                if attempt == max_retries:
                    break

                logger.warning("%s Tool call attempt %d/%d failed for tool %s: %s",
                               AGENT_LOG_PREFIX,
                               attempt,
                               max_retries,
                               tool.name,
                               str(e))

                # Exponential backoff: 2^attempt seconds
                sleep_time = 2**attempt
                logger.debug("%s Retrying tool call for %s in %d seconds...", AGENT_LOG_PREFIX, tool.name, sleep_time)
                await asyncio.sleep(sleep_time)

        # All retries exhausted, return error message
        error_content = "Tool call failed after all retry attempts. Last error: %s" % str(last_exception)
        logger.error("%s %s", AGENT_LOG_PREFIX, error_content)
        return ToolMessage(name=tool.name, tool_call_id=tool.name, content=error_content, status="error")

    def _log_tool_response(self, tool_name: str, tool_input: Any, tool_response: str, max_chars: int = 1000) -> None:
        """
        Log tool response with consistent formatting and length limits.

        Parameters
        ----------
        tool_name : str
            The name of the tool that was called
        tool_input : Any
            The input that was passed to the tool
        tool_response : str
            The response from the tool
        max_chars : int
            Maximum number of characters to log (default: 1000)
        """
        if self.detailed_logs:
            # Truncate tool response if too long
            display_response = tool_response[:max_chars] + "...(rest of response truncated)" if len(
                tool_response) > max_chars else tool_response

            # Format the tool input for display
            tool_input_str = str(tool_input)

            tool_response_log_message = TOOL_CALL_LOG_MESSAGE % (tool_name, tool_input_str, display_response)
            logger.info(tool_response_log_message)

    def _parse_json(self, json_string: str) -> dict[str, Any]:
        """
        Safely parse JSON with graceful error handling.
        If JSON parsing fails, returns an empty dict or error info.

        Parameters
        ----------
        json_string : str
            The JSON string to parse

        Returns
        -------
        Dict[str, Any]
            The parsed JSON or error information
        """
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            logger.warning("%s JSON parsing failed, returning the original string: %s", AGENT_LOG_PREFIX, str(e))
            return {"error": f"JSON parsing failed: {str(e)}", "original_string": json_string}
        except Exception as e:
            logger.warning("%s Unexpected error during JSON parsing: %s", AGENT_LOG_PREFIX, str(e))
            return {"error": f"Unexpected parsing error: {str(e)}", "original_string": json_string}

    @abstractmethod
    async def _build_graph(self, state_schema: type) -> CompiledGraph:
        pass
