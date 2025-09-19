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
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from pydantic import BaseModel
from pydantic import Field

from nat.agent.base import AGENT_CALL_LOG_MESSAGE
from nat.agent.base import AGENT_LOG_PREFIX
from nat.agent.base import BaseAgent

if typing.TYPE_CHECKING:
    from nat.control_flow.router_agent.register import RouterAgentWorkflowConfig

logger = logging.getLogger(__name__)


class RouterAgentGraphState(BaseModel):
    """State schema for the Router Agent Graph.

    This class defines the state structure used throughout the Router Agent's
    execution graph, containing messages, routing information, and branch selection.

    Attributes:
        messages: A list of messages representing the conversation history.
        forward_message: The message to be forwarded to the chosen branch.
        chosen_branch: The name of the branch selected by the router agent.
    """
    messages: list[BaseMessage] = Field(default_factory=list)
    forward_message: BaseMessage = Field(default_factory=lambda: HumanMessage(content=""))
    chosen_branch: str = Field(default="")


class RouterAgentGraph(BaseAgent):
    """Configurable Router Agent for routing requests to different branches.

    A Router Agent analyzes incoming requests and routes them to one of the
    configured branches based on the conte nt and context. It makes a single
    routing decision and executes only the selected branch before returning.

    This agent is useful for creating multi-path workflows where different
    types of requests need to be handled by specialized sub-agents or tools.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        branches: list[BaseTool],
        prompt: ChatPromptTemplate,
        max_router_retries: int = 3,
        callbacks: list[AsyncCallbackHandler] | None = None,
        detailed_logs: bool = False,
        log_response_max_chars: int = 1000,
    ):
        """Initialize the Router Agent.

        Args:
            llm: The language model to use for routing decisions.
            branches: List of tools/branches that the agent can route to.
            prompt: The chat prompt template for the routing agent.
            max_router_retries: Maximum number of retries if branch selection fails.
            callbacks: Optional list of async callback handlers.
            detailed_logs: Whether to enable detailed logging.
            log_response_max_chars: Maximum characters to log in responses.
        """
        super().__init__(llm=llm,
                         tools=branches,
                         callbacks=callbacks,
                         detailed_logs=detailed_logs,
                         log_response_max_chars=log_response_max_chars)

        self._branches = branches
        self._branches_dict = {branch.name: branch for branch in branches}
        branch_names = ",".join([branch.name for branch in branches])
        branch_names_and_descriptions = "\n".join([f"{branch.name}: {branch.description}" for branch in branches])

        prompt = prompt.partial(branches=branch_names_and_descriptions, branch_names=branch_names)
        self.agent = prompt | self.llm

        self.max_router_retries = max_router_retries

    def _get_branch(self, branch_name: str) -> BaseTool | None:
        return self._branches_dict.get(branch_name, None)

    async def agent_node(self, state: RouterAgentGraphState):
        """Execute the agent node to select a branch for routing.

        This method processes the incoming request and determines which branch
        should handle it. It uses the configured LLM to analyze the request
        and select the most appropriate branch.

        Args:
            state: The current state of the router agent graph.

        Returns:
            RouterAgentGraphState: Updated state with the chosen branch.

        Raises:
            RuntimeError: If the agent fails to choose a branch after max retries.
        """
        logger.debug("%s Starting the Router Agent Node", AGENT_LOG_PREFIX)
        chat_history = self._get_chat_history(state.messages)
        request = state.forward_message.content
        for attempt in range(1, self.max_router_retries + 1):
            try:
                agent_response = await self._call_llm(self.agent, {"request": request, "chat_history": chat_history})
                if self.detailed_logs:
                    logger.info(AGENT_CALL_LOG_MESSAGE, request, agent_response)

                state.messages += [agent_response]

                # Determine chosen branch based on agent response
                if state.chosen_branch == "":
                    for branch in self._branches:
                        if branch.name.lower() in str(agent_response.content).lower():
                            state.chosen_branch = branch.name
                            if self.detailed_logs:
                                logger.debug("%s Router Agent has chosen branch: %s", AGENT_LOG_PREFIX, branch.name)
                            return state

                # The agent failed to choose a branch
                if state.chosen_branch == "":
                    if attempt == self.max_router_retries:
                        logger.error("%s Router Agent has empty chosen branch", AGENT_LOG_PREFIX)
                        raise RuntimeError("Router Agent failed to choose a branch")
                    logger.warning("%s Router Agent failed to choose a branch, retrying %d out of %d",
                                   AGENT_LOG_PREFIX,
                                   attempt,
                                   self.max_router_retries)

            except Exception as ex:
                logger.error("%s Router Agent failed to call agent_node: %s", AGENT_LOG_PREFIX, ex)
                raise

        return state

    async def branch_node(self, state: RouterAgentGraphState):
        """Execute the selected branch with the forwarded message.

        This method calls the tool/branch that was selected by the agent node
        and processes the response.

        Args:
            state: The current state containing the chosen branch and message.

        Returns:
            RouterAgentGraphState: Updated state with the branch response.

        Raises:
            RuntimeError: If no branch was chosen or branch execution fails.
            ValueError: If the requested tool is not found in the configuration.
        """
        logger.debug("%s Starting Router Agent Tool Node", AGENT_LOG_PREFIX)
        try:
            if state.chosen_branch == "":
                logger.error("%s Router Agent has empty chosen branch", AGENT_LOG_PREFIX)
                raise RuntimeError("Router Agent failed to choose a branch")
            requested_branch = self._get_branch(state.chosen_branch)
            if not requested_branch:
                logger.error("%s Router Agent wants to call tool %s but it is not in the config file",
                             AGENT_LOG_PREFIX,
                             state.chosen_branch)
                raise ValueError("Tool not found in config file")

            branch_input = state.forward_message.content
            branch_response = await self._call_tool(requested_branch, branch_input)
            state.messages += [branch_response]
            if self.detailed_logs:
                self._log_tool_response(requested_branch.name, branch_input, branch_response.content)

            return state

        except Exception as ex:
            logger.error("%s Router Agent throws exception during branch node execution: %s", AGENT_LOG_PREFIX, ex)
            raise

    async def _build_graph(self, state_schema):
        logger.debug("%s Building and compiling the Router Agent Graph", AGENT_LOG_PREFIX)

        graph = StateGraph(state_schema)
        graph.add_node("agent", self.agent_node)
        graph.add_node("branch", self.branch_node)
        graph.add_edge("agent", "branch")
        graph.set_entry_point("agent")

        self.graph = graph.compile()
        logger.debug("%s Router Agent Graph built and compiled successfully", AGENT_LOG_PREFIX)

        return self.graph

    async def build_graph(self):
        """Build and compile the router agent execution graph.

        Creates a state graph with agent and branch nodes, configures the
        execution flow, and compiles the graph for execution.

        Returns:
            The compiled execution graph.

        Raises:
            Exception: If graph building or compilation fails.
        """
        try:
            await self._build_graph(state_schema=RouterAgentGraphState)
            return self.graph
        except Exception as ex:
            logger.error("%s Router Agent failed to build graph: %s", AGENT_LOG_PREFIX, ex)
            raise

    @staticmethod
    def validate_system_prompt(system_prompt: str) -> bool:
        """Validate that the system prompt contains required variables.

        Checks that the system prompt includes necessary template variables
        for branch information that the router agent needs.

        Args:
            system_prompt: The system prompt string to validate.

        Returns:
            True if the prompt is valid, False otherwise.
        """
        errors = []
        required_prompt_variables = {
            "{branches}": "The system prompt must contain {branches} so the agent knows about configured branches.",
            "{branch_names}": "The system prompt must contain {branch_names} so the agent knows branch names."
        }
        for variable_name, error_message in required_prompt_variables.items():
            if variable_name not in system_prompt:
                errors.append(error_message)
        if errors:
            error_text = "\n".join(errors)
            logger.error("%s %s", AGENT_LOG_PREFIX, error_text)
            return False
        return True

    @staticmethod
    def validate_user_prompt(user_prompt: str) -> bool:
        """Validate that the user prompt contains required variables.

        Checks that the user prompt includes necessary template variables
        for chat history and other required information.

        Args:
            user_prompt: The user prompt string to validate.

        Returns:
            True if the prompt is valid, False otherwise.
        """
        errors = []
        if not user_prompt:
            errors.append("The user prompt cannot be empty.")
        else:
            required_prompt_variables = {
                "{chat_history}":
                    "The user prompt must contain {chat_history} so the agent knows about the conversation history.",
                "{request}":
                    "The user prompt must contain {request} so the agent sees the current request.",
            }
            for variable_name, error_message in required_prompt_variables.items():
                if variable_name not in user_prompt:
                    errors.append(error_message)
        if errors:
            error_text = "\n".join(errors)
            logger.error("%s %s", AGENT_LOG_PREFIX, error_text)
            return False
        return True


def create_router_agent_prompt(config: "RouterAgentWorkflowConfig") -> ChatPromptTemplate:
    """Create a Router Agent prompt from the configuration.

    Builds a ChatPromptTemplate using either custom prompts from the config
    or default system and user prompts. Validates the prompts to ensure they
    contain required template variables.

    Args:
        config: The router agent workflow configuration containing prompt settings.

    Returns:
        A configured ChatPromptTemplate for the router agent.

    Raises:
        ValueError: If the system_prompt or user_prompt validation fails.
    """
    from nat.control_flow.router_agent.prompt import SYSTEM_PROMPT
    from nat.control_flow.router_agent.prompt import USER_PROMPT
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

    if not RouterAgentGraph.validate_user_prompt(user_prompt):
        logger.error("%s Invalid user_prompt", AGENT_LOG_PREFIX)
        raise ValueError("Invalid user_prompt")

    return ChatPromptTemplate([("system", system_prompt), ("user", user_prompt)])
