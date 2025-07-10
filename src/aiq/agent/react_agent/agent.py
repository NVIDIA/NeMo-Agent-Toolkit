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

import json
# pylint: disable=R0917
import logging
from json import JSONDecodeError

from langchain_core.agents import AgentAction
from langchain_core.agents import AgentFinish
from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from pydantic import Field

from aiq.agent.base import AGENT_LOG_PREFIX
from aiq.agent.base import AGENT_RESPONSE_LOG_MESSAGE
from aiq.agent.base import INPUT_SCHEMA_MESSAGE
from aiq.agent.base import NO_INPUT_ERROR_MESSAGE
from aiq.agent.base import TOOL_NOT_FOUND_ERROR_MESSAGE
from aiq.agent.base import TOOL_RESPONSE_LOG_MESSAGE
from aiq.agent.base import AgentDecision
from aiq.agent.dual_node import DualNodeAgent
from aiq.agent.react_agent.output_parser import ReActOutputParser
from aiq.agent.react_agent.output_parser import ReActOutputParserException
from aiq.agent.react_agent.prompt import SYSTEM_PROMPT
from aiq.agent.react_agent.prompt import USER_PROMPT
from aiq.agent.react_agent.register import ReActAgentWorkflowConfig

logger = logging.getLogger(__name__)


class ReActGraphState(BaseModel):
    """State schema for the ReAct Agent Graph"""
    messages: list[BaseMessage] = Field(default_factory=list)  # input and output of the ReAct Agent
    agent_scratchpad: list[AgentAction] = Field(default_factory=list)  # agent thoughts / intermediate steps
    tool_responses: list[BaseMessage] = Field(default_factory=list)  # the responses from any tool calls


class ReActAgentGraph(DualNodeAgent):
    """Configurable LangGraph ReAct Agent. A ReAct Agent performs reasoning inbetween tool calls, and utilizes the tool
    names and descriptions to select the optimal tool.  Supports retrying on output parsing errors.  Argument
    "detailed_logs" toggles logging of inputs, outputs, and intermediate steps."""

    def __init__(self,
                 llm: BaseChatModel,
                 prompt: ChatPromptTemplate,
                 tools: list[BaseTool],
                 use_tool_schema: bool = True,
                 callbacks: list[AsyncCallbackHandler] = None,
                 detailed_logs: bool = False,
                 retry_parsing_errors: bool = True,
                 max_retries: int = 1):
        super().__init__(llm=llm, tools=tools, callbacks=callbacks, detailed_logs=detailed_logs)
        self.retry_parsing_errors = retry_parsing_errors
        self.max_tries = (max_retries + 1) if retry_parsing_errors else 1
        logger.debug(
            "%s Filling the prompt variables 'tools' and 'tool_names', using the tools provided in the config.",
            AGENT_LOG_PREFIX)
        tool_names = ",".join([tool.name for tool in tools[:-1]]) + ',' + tools[-1].name  # prevent trailing ","
        if not use_tool_schema:
            tool_names_and_descriptions = "\n".join(
                [f"{tool.name}: {tool.description}"
                 for tool in tools[:-1]]) + "\n" + f"{tools[-1].name}: {tools[-1].description}"  # prevent trailing "\n"
        else:
            logger.debug("%s Adding the tools' input schema to the tools' description", AGENT_LOG_PREFIX)
            tool_names_and_descriptions = "\n".join([
                f"{tool.name}: {tool.description}. {INPUT_SCHEMA_MESSAGE.format(schema=tool.input_schema.model_fields)}"
                for tool in tools[:-1]
            ]) + "\n" + (f"{tools[-1].name}: {tools[-1].description}. "
                         f"{INPUT_SCHEMA_MESSAGE.format(schema=tools[-1].input_schema.model_fields)}")
        prompt = prompt.partial(tools=tool_names_and_descriptions, tool_names=tool_names)
        # construct the ReAct Agent
        llm = llm.bind(stop=["Observation:"])
        self.agent = prompt | llm
        self.tools_dict = {tool.name: tool for tool in tools}
        logger.debug("%s Initialized ReAct Agent Graph", AGENT_LOG_PREFIX)

    def _get_tool(self, tool_name):
        try:
            return self.tools_dict.get(tool_name)
        except Exception as ex:
            logger.exception("%s Unable to find tool with the name %s\n%s",
                             AGENT_LOG_PREFIX,
                             tool_name,
                             ex,
                             exc_info=True)
            raise ex

    async def agent_node(self, state: ReActGraphState):
        try:
            logger.debug("%s Starting the ReAct Agent Node", AGENT_LOG_PREFIX)
            # keeping a working state allows us to resolve parsing errors without polluting the agent scratchpad
            # the agent "forgets" about the parsing error after solving it - prevents hallucinations in next cycles
            working_state = []
            for attempt in range(1, self.max_tries + 1):
                # the first time we are invoking the ReAct Agent, it won't have any intermediate steps / agent thoughts
                if len(state.agent_scratchpad) == 0 and len(working_state) == 0:
                    # the user input comes from the "messages" state channel
                    if len(state.messages) == 0:
                        raise RuntimeError('No input received in state: "messages"')
                    # to check is any human input passed or not, if no input passed Agent will return the state
                    if state.messages[0].content.strip() == "":
                        logger.error("%s No human input passed to the agent.", AGENT_LOG_PREFIX)
                        state.messages += [AIMessage(content=NO_INPUT_ERROR_MESSAGE)]
                        return state
                    question = state.messages[0].content
                    logger.debug("%s Querying agent, attempt: %s", AGENT_LOG_PREFIX, attempt)
                    output_message = ""
                    async for event in self.agent.astream({"question": question},
                                                          config=RunnableConfig(callbacks=self.callbacks)):
                        output_message += event.content
                    output_message = AIMessage(content=output_message)
                    if self.detailed_logs:
                        logger.info(AGENT_RESPONSE_LOG_MESSAGE, question, output_message.content)
                else:
                    # ReAct Agents require agentic cycles
                    # in an agentic cycle, preserve the agent's thoughts from the previous cycles,
                    # and give the agent the response from the tool it called
                    agent_scratchpad = []
                    for index, intermediate_step in enumerate(state.agent_scratchpad):
                        agent_thoughts = AIMessage(content=intermediate_step.log)
                        agent_scratchpad.append(agent_thoughts)
                        tool_response = HumanMessage(content=state.tool_responses[index].content)
                        agent_scratchpad.append(tool_response)
                    agent_scratchpad += working_state
                    question = state.messages[0].content
                    logger.debug("%s Querying agent, attempt: %s", AGENT_LOG_PREFIX, attempt)
                    output_message = ""
                    async for event in self.agent.astream({
                            "question": question, "agent_scratchpad": agent_scratchpad
                    },
                                                          config=RunnableConfig(callbacks=self.callbacks)):
                        output_message += event.content
                    output_message = AIMessage(content=output_message)
                    if self.detailed_logs:
                        logger.info(AGENT_RESPONSE_LOG_MESSAGE, question, output_message.content)
                        logger.debug("%s The agent's scratchpad (with tool result) was:\n%s",
                                     AGENT_LOG_PREFIX,
                                     agent_scratchpad)
                try:
                    # check if the agent has the final answer yet
                    logger.debug("%s Successfully obtained agent response. Parsing agent's response", AGENT_LOG_PREFIX)
                    agent_output = await ReActOutputParser().aparse(output_message.content)
                    logger.debug("%s Successfully parsed agent's response", AGENT_LOG_PREFIX)
                    if attempt > 1:
                        logger.debug("%s Successfully parsed agent response after %s attempts",
                                     AGENT_LOG_PREFIX,
                                     attempt)
                    if isinstance(agent_output, AgentFinish):
                        final_answer = agent_output.return_values.get('output', output_message.content)
                        logger.debug("%s The agent has finished, and has the final answer", AGENT_LOG_PREFIX)
                        # this is where we handle the final output of the Agent, we can clean-up/format/postprocess here
                        # the final answer goes in the "messages" state channel
                        state.messages += [AIMessage(content=final_answer)]
                    else:
                        # the agent wants to call a tool, ensure the thoughts are preserved for the next agentic cycle
                        agent_output.log = output_message.content
                        logger.debug("%s The agent wants to call a tool: %s", AGENT_LOG_PREFIX, agent_output.tool)
                        state.agent_scratchpad += [agent_output]
                    return state
                except ReActOutputParserException as ex:
                    # the agent output did not meet the expected ReAct output format. This can happen for a few reasons:
                    # the agent mentioned a tool, but already has the final answer, this can happen with Llama models
                    #   - the ReAct Agent already has the answer, and is reflecting on how it obtained the answer
                    # the agent might have also missed Action or Action Input in its output
                    logger.warning("%s Error parsing agent output\nObservation:%s\nAgent Output:\n%s",
                                   AGENT_LOG_PREFIX,
                                   ex.observation,
                                   output_message.content)
                    if attempt == self.max_tries:
                        logger.exception(
                            "%s Failed to parse agent output after %d attempts, consider enabling or "
                            "increasing max_retries",
                            AGENT_LOG_PREFIX,
                            attempt,
                            exc_info=True)
                        # the final answer goes in the "messages" state channel
                        output_message.content = ex.observation + '\n' + output_message.content
                        state.messages += [output_message]
                        return state
                    # retry parsing errors, if configured
                    logger.info("%s Retrying ReAct Agent, including output parsing Observation", AGENT_LOG_PREFIX)
                    working_state.append(output_message)
                    working_state.append(HumanMessage(content=ex.observation))
        except Exception as ex:
            logger.exception("%s Failed to call agent_node: %s", AGENT_LOG_PREFIX, ex, exc_info=True)
            raise ex

    async def conditional_edge(self, state: ReActGraphState):
        try:
            logger.debug("%s Starting the ReAct Conditional Edge", AGENT_LOG_PREFIX)
            if len(state.messages) > 1:
                # the ReAct Agent has finished executing, the last agent output was AgentFinish
                logger.debug("%s Final answer:\n%s", AGENT_LOG_PREFIX, state.messages[-1].content)
                return AgentDecision.END
            # else the agent wants to call a tool
            agent_output = state.agent_scratchpad[-1]
            logger.debug("%s The agent wants to call: %s with input: %s",
                         AGENT_LOG_PREFIX,
                         agent_output.tool,
                         agent_output.tool_input)
            return AgentDecision.TOOL
        except Exception as ex:
            logger.exception("Failed to determine whether agent is calling a tool: %s", ex, exc_info=True)
            logger.warning("%s Ending graph traversal", AGENT_LOG_PREFIX)
            return AgentDecision.END

    async def tool_node(self, state: ReActGraphState):
        try:
            logger.debug("%s Starting the Tool Call Node", AGENT_LOG_PREFIX)
            if len(state.agent_scratchpad) == 0:
                raise RuntimeError('No tool input received in state: "agent_scratchpad"')
            agent_thoughts = state.agent_scratchpad[-1]
            # the agent can run any installed tool, simply install the tool and add it to the config file
            requested_tool = self._get_tool(agent_thoughts.tool)
            if not requested_tool:
                configured_tool_names = list(self.tools_dict.keys())
                logger.warning(
                    "%s ReAct Agent wants to call tool %s. In the ReAct Agent's configuration within the config file,"
                    "there is no tool with that name: %s",
                    AGENT_LOG_PREFIX,
                    agent_thoughts.tool,
                    configured_tool_names)
                tool_response = ToolMessage(name='agent_error',
                                            tool_call_id='agent_error',
                                            content=TOOL_NOT_FOUND_ERROR_MESSAGE.format(tool_name=agent_thoughts.tool,
                                                                                        tools=configured_tool_names))
                state.tool_responses += [tool_response]
                return state

            logger.debug("%s Calling tool %s with input: %s",
                         AGENT_LOG_PREFIX,
                         requested_tool.name,
                         agent_thoughts.tool_input)

            # Run the tool. Try to use structured input, if possible.
            try:
                tool_input_str = agent_thoughts.tool_input.strip().replace("'", '"')
                tool_input_dict = json.loads(tool_input_str) if tool_input_str != 'None' else tool_input_str
                logger.debug("%s Successfully parsed structured tool input from Action Input", AGENT_LOG_PREFIX)
                tool_response = await requested_tool.ainvoke(tool_input_dict,
                                                             config=RunnableConfig(callbacks=self.callbacks))
                if self.detailed_logs:
                    # The tool response can be very large, so we log only the first 1000 characters
                    tool_response_str = str(tool_response)
                    tool_response_str = tool_response_str[:1000] + "..." if len(
                        tool_response_str) > 1000 else tool_response_str
                    tool_response_log_message = TOOL_RESPONSE_LOG_MESSAGE % (
                        requested_tool.name, tool_input_str, tool_response_str)
                    logger.info(tool_response_log_message)
            except JSONDecodeError as ex:
                logger.warning(
                    "%s Unable to parse structured tool input from Action Input. Using Action Input as is."
                    "\nParsing error: %s",
                    AGENT_LOG_PREFIX,
                    ex,
                    exc_info=True)
                tool_input_str = agent_thoughts.tool_input
                tool_response = await requested_tool.ainvoke(tool_input_str,
                                                             config=RunnableConfig(callbacks=self.callbacks))

            # some tools, such as Wikipedia, will return an empty response when no search results are found
            if tool_response is None or tool_response == "":
                tool_response = "The tool provided an empty response.\n"
            # put the tool response in the graph state
            tool_response = ToolMessage(name=agent_thoughts.tool,
                                        tool_call_id=agent_thoughts.tool,
                                        content=tool_response)
            logger.debug("%s Called tool %s with input: %s\nThe tool returned: %s",
                         AGENT_LOG_PREFIX,
                         requested_tool.name,
                         agent_thoughts.tool_input,
                         tool_response.content)
            state.tool_responses += [tool_response]
            return state
        except Exception as ex:
            logger.exception("%s Failed to call tool_node: %s", AGENT_LOG_PREFIX, ex, exc_info=ex)
            raise ex

    async def build_graph(self):
        try:
            await super()._build_graph(state_schema=ReActGraphState)
            logger.debug("%s ReAct Graph built and compiled successfully", AGENT_LOG_PREFIX)
            return self.graph
        except Exception as ex:
            logger.exception("%s Failed to build ReAct Graph: %s", AGENT_LOG_PREFIX, ex, exc_info=ex)
            raise ex

    @staticmethod
    def validate_system_prompt(system_prompt: str) -> bool:
        errors = []
        if not system_prompt:
            errors.append("The system prompt cannot be empty.")
        required_prompt_variables = {
            "{tools}": "The system prompt must contain {tools} so the agent knows about configured tools.",
            "{tool_names}": "The system prompt must contain {tool_names} so the agent knows tool names."
        }
        for variable_name, error_message in required_prompt_variables.items():
            if variable_name not in system_prompt:
                errors.append(error_message)
        if errors:
            error_text = "\n".join(errors)
            logger.exception("%s %s", AGENT_LOG_PREFIX, error_text)
            raise ValueError(error_text)
        return True


def create_react_agent_prompt(config: ReActAgentWorkflowConfig) -> ChatPromptTemplate:
    """
    Create a ReAct Agent prompt from the config.

    Args:
        config (ReActAgentWorkflowConfig): The config to use for the prompt.

    Returns:
        ChatPromptTemplate: The ReAct Agent prompt.
    """
    # the ReAct Agent prompt can be customized via config option system_prompt and additional_instructions.

    if config.system_prompt:
        prompt_str = config.system_prompt
    else:
        prompt_str = SYSTEM_PROMPT

    if config.additional_instructions:
        prompt_str += f" {config.additional_instructions}"

    valid_prompt = ReActAgentGraph.validate_system_prompt(prompt_str)
    if not valid_prompt:
        logger.exception("%s Invalid system_prompt", AGENT_LOG_PREFIX)
        raise ValueError("Invalid system_prompt")
    prompt = ChatPromptTemplate([("system", prompt_str), ("user", USER_PROMPT),
                                 MessagesPlaceholder(variable_name='agent_scratchpad', optional=True)])
    return prompt
