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

# pylint: disable=R0917
import logging

from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages.base import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from pydantic import Field

from aiq.agent.base import AGENT_CALL_LOG_MESSAGE
from aiq.agent.base import AGENT_LOG_PREFIX
from aiq.agent.base import AgentDecision
from aiq.agent.dual_node import DualNodeAgent

logger = logging.getLogger(__name__)


class ToolCallAgentGraphState(BaseModel):
    """State schema for the Tool Calling Agent Graph"""
    messages: list[BaseMessage] = Field(default_factory=list)  # input and output of the Agent


class ToolCallAgentGraph(DualNodeAgent):
    """Configurable LangGraph Tool Calling Agent. A Tool Calling Agent requires an LLM which supports tool calling.
    A tool Calling Agent utilizes the tool input parameters to select the optimal tool.  Supports handling tool errors.
    Argument "detailed_logs" toggles logging of inputs, outputs, and intermediate steps."""

    def __init__(self,
                 llm: BaseChatModel,
                 tools: list[BaseTool],
                 callbacks: list[AsyncCallbackHandler] = None,
                 detailed_logs: bool = False,
                 handle_tool_errors: bool = True):
        super().__init__(llm=llm, tools=tools, callbacks=callbacks, detailed_logs=detailed_logs)
        self.tool_caller = ToolNode(tools, handle_tool_errors=handle_tool_errors)
        logger.debug("%s Initialized Tool Calling Agent Graph", AGENT_LOG_PREFIX)

    async def agent_node(self, state: ToolCallAgentGraphState):
        try:
            logger.debug('%s Starting the Tool Calling Agent Node', AGENT_LOG_PREFIX)
            if len(state.messages) == 0:
                raise RuntimeError('No input received in state: "messages"')
            response = await self.llm.ainvoke(state.messages, config=RunnableConfig(callbacks=self.callbacks))
            if self.detailed_logs:
                agent_input = "\n".join(str(message.content) for message in state.messages)
                logger.info(AGENT_CALL_LOG_MESSAGE, agent_input, response)

            state.messages += [response]
            return state
        except Exception as ex:
            logger.exception("%s Failed to call agent_node: %s", AGENT_LOG_PREFIX, ex, exc_info=True)
            raise ex

    async def conditional_edge(self, state: ToolCallAgentGraphState):
        try:
            logger.debug("%s Starting the Tool Calling Conditional Edge", AGENT_LOG_PREFIX)
            last_message = state.messages[-1]
            if last_message.tool_calls:
                # the agent wants to call a tool
                logger.debug('%s Agent is calling a tool', AGENT_LOG_PREFIX)
                return AgentDecision.TOOL
            if self.detailed_logs:
                logger.debug("%s Final answer:\n%s", AGENT_LOG_PREFIX, state.messages[-1].content)
            return AgentDecision.END
        except Exception as ex:
            logger.exception("%s Failed to determine whether agent is calling a tool: %s",
                             AGENT_LOG_PREFIX,
                             ex,
                             exc_info=True)
            logger.warning("%s Ending graph traversal", AGENT_LOG_PREFIX)
            return AgentDecision.END

    async def tool_node(self, state: ToolCallAgentGraphState):
        try:
            logger.debug("%s Starting Tool Node", AGENT_LOG_PREFIX)
            tool_calls = state.messages[-1].tool_calls
            tools = [tool.get('name') for tool in tool_calls]
            tool_input = state.messages[-1]
            tool_response = await self.tool_caller.ainvoke(input={"messages": [tool_input]},
                                                           config=RunnableConfig(callbacks=self.callbacks,
                                                                                 configurable={}))
            # this configurable = {} argument is needed due to a bug in LangGraph PreBuilt ToolNode ^

            for response in tool_response.get('messages'):
                if self.detailed_logs:
                    self._log_tool_response(str(tools), str(tool_input), response.content)
                state.messages += [response]

            return state
        except Exception as ex:
            logger.exception("%s Failed to call tool_node: %s", AGENT_LOG_PREFIX, ex, exc_info=ex)
            raise ex

    async def build_graph(self):
        try:
            await super()._build_graph(state_schema=ToolCallAgentGraphState)
            logger.debug("%s Tool Calling Agent Graph built and compiled successfully", AGENT_LOG_PREFIX)
            return self.graph
        except Exception as ex:
            logger.exception("%s Failed to build Tool Calling Agent Graph: %s", AGENT_LOG_PREFIX, ex, exc_info=ex)
            raise ex
