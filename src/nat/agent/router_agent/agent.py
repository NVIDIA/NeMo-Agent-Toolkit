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
import typing

from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from pydantic import Field

from nat.agent.base import AGENT_CALL_LOG_MESSAGE
from nat.agent.base import AGENT_LOG_PREFIX
from nat.agent.base import AgentDecision
from nat.agent.dual_node import DualNodeAgent

if typing.TYPE_CHECKING:
    from nat.agent.tool_calling_agent.register import ToolCallAgentWorkflowConfig

logger = logging.getLogger(__name__)


class RouterAgentGraphState(BaseModel):
    """State schema for the Router Agent Graph"""
    messages: list[BaseMessage] = Field(default_factory=list)  # input and output of the Agent


class RouterAgentGraph(DualNodeAgent):
    """Configurable LangGraph Router Agent. A Router Agent requires an LLM which supports routing.
    A tool Calling Agent utilizes the tool input parameters to select the optimal tool.  Supports handling tool errors.
    Argument "detailed_logs" toggles logging of inputs, outputs, and intermediate steps."""

    def __init__(
        self,
        llm: BaseChatModel,
        branches: list[BaseTool],
        prompt: ChatPromptTemplate,
        callbacks: list[AsyncCallbackHandler] = None,
        detailed_logs: bool = False,
        log_response_max_chars: int = 1000,
    ):
        super().__init__(llm=llm,
                         tools=branches,
                         callbacks=callbacks,
                         detailed_logs=detailed_logs,
                         log_response_max_chars=log_response_max_chars)

        self._branches = branches
        branch_names = ",".join([branch.name for branch in branches])
        branch_names_and_descriptions = "\n".join([f"{branch.name}: {branch.description}" for branch in branches])
        prompt = prompt.partial(branches=branch_names_and_descriptions, branch_names=branch_names)

        self.agent = prompt | self._maybe_bind_llm_and_yield()

    async def agent_node(self, state: RouterAgentGraphState):
        try:
            if len(state.messages) == 0:
                raise RuntimeError('No input received in state: "messages"')
            chat_history = self._get_chat_history(state.messages)
            response = await self._call_llm(self.agent, {
                "question": state.messages[-1].content, "chat_history": chat_history
            },
                                            config=RunnableConfig(callbacks=self.callbacks))
            if self.detailed_logs:
                agent_input = "\n".join(str(message.content) for message in state.messages)
                logger.info(AGENT_CALL_LOG_MESSAGE, agent_input, response)

            state.messages.append(response)
            return state
        except Exception as ex:
            logger.error("%s Failed to call agent_node: %s", AGENT_LOG_PREFIX, ex)
            raise

    async def conditional_edge(self, state: RouterAgentGraphState):
        try:
            logger.debug("%s Starting the Router Agent Conditional Edge", AGENT_LOG_PREFIX)
            last_message = state.messages[-1]
            for branch in self._branches:
                if branch.name.lower() in str(last_message.content).lower():
                    return AgentDecision.TOOL
            if self.detailed_logs:
                logger.debug("%s Final answer:\n%s", AGENT_LOG_PREFIX, state.messages[-1].content)
            return AgentDecision.END
        except Exception as ex:
            logger.exception("%s Failed to determine whether agent is calling a tool: %s", AGENT_LOG_PREFIX, ex)
            logger.warning("%s Ending graph traversal", AGENT_LOG_PREFIX)
            return AgentDecision.END

    async def tool_node(self, state: RouterAgentGraphState):
        try:
            logger.debug("%s Starting Tool Node", AGENT_LOG_PREFIX)
            tool_calls = state.messages[-1].tool_calls
            tools = [tool.get("name") for tool in tool_calls]
            tool_input = state.messages[-1]
            tool_response = await self.tool_caller.ainvoke(
                input={"messages": [tool_input]},
                config=RunnableConfig(callbacks=self.callbacks, configurable={}),
            )
            # this configurable = {} argument is needed due to a bug in LangGraph PreBuilt ToolNode ^

            for response in tool_response.get("messages"):
                if self.detailed_logs:
                    self._log_tool_response(str(tools), str(tool_input), response.content)
                state.messages += [response]

            return state
        except Exception as ex:
            logger.error("%s Failed to call tool_node: %s", AGENT_LOG_PREFIX, ex)
            raise

    async def build_graph(self):
        try:
            await super()._build_graph(state_schema=RouterAgentGraphState)
            logger.debug(
                "%s Tool Calling Agent Graph built and compiled successfully",
                AGENT_LOG_PREFIX,
            )
            return self.graph
        except Exception as ex:
            logger.error("%s Failed to build Router Agent Graph: %s", AGENT_LOG_PREFIX, ex)
            raise

    @staticmethod
    def validate_system_prompt(system_prompt: str) -> bool:
        return True


def create_router_agent_prompt(config: "RouterAgentWorkflowConfig") -> ChatPromptTemplate:
    from nat.agent.router_agent.prompt import SYSTEM_PROMPT
    from nat.agent.router_agent.prompt import USER_PROMPT
    """
    Create a Router Agent prompt from the config.

    Args:
        config (RouterAgentWorkflowConfig): The config to use for the prompt.

    Returns:
        ChatPromptTemplate: The Router Agent prompt.
    """
    # the Router Agent prompt can be customized via config option system_prompt and user_prompt.

    if config.system_prompt:
        system_prompt = config.system_prompt
    else:
        system_prompt = SYSTEM_PROMPT

    if config.user_prompt:
        user_prompt = config.user_prompt
    else:
        user_prompt = USER_PROMPT

    if not RouterAgentGraph.validate_system_prompt(system_prompt):
        logger.error("%s Invalid system_prompt", AGENT_LOG_PREFIX)
        raise ValueError("Invalid system_prompt")

    return ChatPromptTemplate([("system", system_prompt), ("user", user_prompt)])
