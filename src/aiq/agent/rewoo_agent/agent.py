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
import re
from json import JSONDecodeError

from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from pydantic import BaseModel
from pydantic import Field

from aiq.agent.base import AgentDecision
from aiq.agent.base import BaseAgent

from .prompt import REWOO_PLAN_PATTERN

logger = logging.getLogger(__name__)

TOOL_NOT_FOUND_ERROR_MESSAGE = "There is no tool named {tool_name}. Tool must be one of {tools}."
INPUT_SCHEMA_MESSAGE = ". Arguments must be provided as a valid JSON object following this format: {schema}"
NO_INPUT_ERROR_MESSAGE = "No human input recieved to the agent, Please ask a valid question."


class ReWOOGraphState(BaseModel):
    """State schema for the ReAct Agent Graph"""
    task: str = Field(default="")  # the task to be performed
    plan: str = Field(default="")  # the plan to be executed
    steps: list[tuple[str, str, str, str]] = Field(default_factory=list)  # the steps to be executed
    intermediate_results: dict[str, str] = Field(default_factory=dict)  # the intermediate results of each step
    result: str = Field(default="")  # the final result of the task


class ReWOOAgentGraph(BaseAgent):
    """Configurable LangGraph ReWOO Agent. A ReWOO Agent performs reasoning by interacting with other objects or tools
    and utilizes their outputs to make decisions. Supports retrying on output parsing errors. Argument
    "detailed_logs" toggles logging of inputs, outputs, and intermediate steps."""

    def __init__(self,
                 llm: BaseChatModel,
                 planner_prompt: ChatPromptTemplate,
                 solver_prompt: ChatPromptTemplate,
                 tools: list[BaseTool],
                 use_tool_schema: bool = True,
                 callbacks: list[AsyncCallbackHandler] = None,
                 detailed_logs: bool = False):
        super().__init__(llm=llm, tools=tools, callbacks=callbacks, detailed_logs=detailed_logs)

        logger.info('Filling the prompt variables "tools" and "tool_names", using the tools provided in the config.')
        tool_names = ",".join([tool.name for tool in tools[:-1]]) + ',' + tools[-1].name  # prevent trailing ","
        if not use_tool_schema:
            tool_names_and_descriptions = "\n".join(
                [f"{tool.name}: {tool.description}"
                 for tool in tools[:-1]]) + "\n" + f"{tools[-1].name}: {tools[-1].description}"  # prevent trailing "\n"
        else:
            logger.info("Adding the tools' input schema to the tools' description")
            tool_names_and_descriptions = "\n".join([
                f"{tool.name}: {tool.description}. {INPUT_SCHEMA_MESSAGE.format(schema=tool.input_schema.model_fields)}"
                for tool in tools[:-1]
            ]) + "\n" + (f"{tools[-1].name}: {tools[-1].description}. "
                         f"{INPUT_SCHEMA_MESSAGE.format(schema=tools[-1].input_schema.model_fields)}")

        self.planner_prompt = planner_prompt.partial(tools=tool_names_and_descriptions, tool_names=tool_names)
        self.solver_prompt = solver_prompt
        self.tools_dict = {tool.name: tool for tool in tools}

        logger.info("Initialized ReWOO Agent Graph")

    def _get_tool(self, tool_name):
        try:
            return self.tools_dict.get(tool_name)
        except Exception as ex:
            logger.exception("Unable to find tool with the name %s\n%s", tool_name, ex, exc_info=True)
            raise ex

    def _get_current_step(self, state: ReWOOGraphState) -> int:
        if len(state.steps) == 0:
            raise RuntimeError('No steps received in ReWOOGraphState')
        # Get the current step according to the number of intermediate results
        if len(state.intermediate_results) == 0:
            return 1
        if len(state.intermediate_results) == len(state.steps):
            # all steps are done
            return -1
        return len(state.intermediate_results) + 1

    async def planner_node(self, state: ReWOOGraphState):
        try:
            logger.debug("Starting the ReWOO Planner Node")

            planner = self.planner_prompt | self.llm
            task = state.task
            if not task:
                logger.error("No task provided to the ReWOO Agent. Please provide a valid task.")
                return {"result": NO_INPUT_ERROR_MESSAGE}

            plan = ""
            async for event in planner.astream({"task": task}, config=RunnableConfig(callbacks=self.callbacks)):
                plan += event.content

            steps = re.findall(REWOO_PLAN_PATTERN, plan)

            if self.detailed_logs:
                logger.info("The task was: %s", task)
                logger.info("The planner's thoughts are:\n%s", plan)

            return {"plan": plan, "steps": steps}

        except Exception as ex:
            logger.exception("Failed to call planner_node: %s", ex, exc_info=True)
            raise ex

    async def executor_node(self, state: ReWOOGraphState):
        try:
            logger.debug("Starting the ReWOO Executor Node")

            current_step = self._get_current_step(state)
            if current_step < 1:
                logger.error("ReWOO Executor is invoked with an invalid step number: %s", current_step)
                raise RuntimeError(f"ReWOO Executor is invoked with an invalid step number: {current_step}")

            _, step_name, tool, tool_input = state.steps[current_step - 1]
            intermediate_results = state.intermediate_results
            for k, v in intermediate_results.items():
                tool_input = tool_input.replace(k, v)

            requested_tool = self._get_tool(tool)
            if not requested_tool:
                configured_tool_names = list(self.tools_dict.keys())
                logger.warning(
                    "ReWOO Agent wants to call tool %s. In the ReWOO Agent's configuration within the config file,"
                    "there is no tool with that name: %s",
                    tool,
                    configured_tool_names)

                intermediate_results[step_name] = TOOL_NOT_FOUND_ERROR_MESSAGE.format(tool_name=tool,
                                                                                      tools=configured_tool_names)
                return {"intermediate_results": intermediate_results}

            if self.detailed_logs:
                logger.info("Calling tool %s with input: %s", requested_tool.name, tool_input)
            try:
                # Run the tool. Try to use structured input, if possible
                tool_input_str = tool_input.strip().replace("'", '"')
                tool_input_dict = json.loads(tool_input_str) if tool_input_str != 'None' else tool_input_str
                logger.info("Successfully parsed structured tool input from Action Input")
                tool_response = await requested_tool.ainvoke(tool_input_dict,
                                                             config=RunnableConfig(callbacks=self.callbacks))
            except JSONDecodeError:
                logger.info("Unable to parse structured tool input from Action Input. Using Action Input as is.")
                tool_input_str = tool_input
                tool_response = await requested_tool.ainvoke(tool_input_str,
                                                             config=RunnableConfig(callbacks=self.callbacks))

            # some tools, such as Wikipedia, will return an empty response when no search results are found
            if tool_response is None or tool_response == "":
                tool_response = "The tool provided an empty response.\n"

            tool_response = ToolMessage(name=tool, tool_call_id=tool, content=tool_response)
            logger.debug("Successfully called the tool")
            if self.detailed_logs:
                logger.debug('The tool returned: %s', tool_response)

            intermediate_results[step_name] = str(tool_response.content)
            return {"intermediate_results": intermediate_results}

        except Exception as ex:
            logger.exception("Failed to call executor_node: %s", ex, exc_info=True)
            raise ex

    async def solver_node(self, state: ReWOOGraphState):
        try:
            logger.debug("Starting the ReWOO Solver Node")

            plan = ""
            # Add results of each step to the plan
            for _plan, step_name, tool, tool_input in state.steps:
                intermediate_results = state.intermediate_results
                for k, v in intermediate_results.items():
                    tool_input = tool_input.replace(k, v)
                    step_name = step_name.replace(k, v)
                plan += f"Plan: {_plan}\n{step_name} = {tool}[{tool_input}]"

            task = state.task
            solver_prompt = self.solver_prompt.partial(plan=plan)
            solver = solver_prompt | self.llm
            output_message = ""
            async for event in solver.astream({"task": task}, config=RunnableConfig(callbacks=self.callbacks)):
                output_message += event.content

            output_message = AIMessage(content=output_message)
            state.result = str(output_message.content)
            return {"result": state.result}

        except Exception as ex:
            logger.exception("Failed to call solver_node: %s", ex, exc_info=True)
            raise ex

    async def conditional_edge(self, state: ReWOOGraphState):
        try:
            logger.debug("Starting the ReWOO Conditional Edge")

            current_step = self._get_current_step(state)
            if current_step == -1:
                logger.debug("The ReWOO Executor has finished its task")
                return AgentDecision.END
            else:
                logger.debug("The ReWOO Executor is still working on the task")
                return AgentDecision.TOOL

        except Exception as ex:
            logger.exception("Failed to determine whether agent is calling a tool: %s", ex, exc_info=True)
            logger.warning("Ending graph traversal")
            return AgentDecision.END

    async def _build_graph(self, state_schema):
        try:
            logger.debug("Building and compiling the ReWOO Graph")

            graph = StateGraph(state_schema)
            graph.add_node("planner", self.planner_node)
            graph.add_node("executor", self.executor_node)
            graph.add_node("solver", self.solver_node)

            graph.add_edge("planner", "executor")
            conditional_edge_possible_outputs = {AgentDecision.TOOL: "executor", AgentDecision.END: "solver"}
            graph.add_conditional_edges("executor", self.conditional_edge, conditional_edge_possible_outputs)

            graph.set_entry_point("planner")
            graph.set_finish_point("solver")

            self.graph = graph.compile()
            logger.info("ReWOO Graph built and compiled successfully")

            return self.graph

        except Exception as ex:
            logger.exception("Failed to build ReWOO Graph: %s", ex, exc_info=ex)
            raise ex

    async def build_graph(self):
        try:
            await self._build_graph(state_schema=ReWOOGraphState)
            logger.info("ReAct Graph built and compiled successfully")
            return self.graph
        except Exception as ex:
            logger.exception("Failed to build ReAct Graph: %s", ex, exc_info=ex)
            raise ex

    @staticmethod
    def validate_planner_prompt(planner_prompt: str) -> bool:
        errors = []
        if not planner_prompt:
            errors.append("The planner prompt cannot be empty.")
        required_prompt_variables = {
            "{tools}": "The planner prompt must contain {tools} so the planner agent knows about configured tools.",
            "{tool_names}": "The planner prompt must contain {tool_names} so the planner agent knows tool names."
        }
        for variable_name, error_message in required_prompt_variables.items():
            if variable_name not in planner_prompt:
                errors.append(error_message)
        if errors:
            error_text = "\n".join(errors)
            logger.exception(error_text)
            raise ValueError(error_text)
        return True

    @staticmethod
    def validate_solver_prompt(solver_prompt: str) -> bool:
        errors = []
        if not solver_prompt:
            errors.append("The solver prompt cannot be empty.")
        if errors:
            error_text = "\n".join(errors)
            logger.exception(error_text)
            raise ValueError(error_text)
        return True
