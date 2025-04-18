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
from abc import ABC
from abc import abstractmethod
from enum import Enum

from colorama import Fore
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from pydantic import BaseModel

log = logging.getLogger(__name__)

TOOL_NOT_FOUND_ERROR_MESSAGE = "There is no tool named {tool_name}. Tool must be one of {tools}."
INPUT_SCHEMA_MESSAGE = ". Arguments must be provided as a valid JSON object following this format: {schema}"
NO_INPUT_ERROR_MESSAGE = "No human input recieved to the agent, Please ask a valid question."

AGENT_RESPONSE_LOG_MESSAGE = f"\n{'-' * 30}\n" + \
                                 Fore.YELLOW + \
                                 "Agent input: %s\n" + \
                                 Fore.CYAN + \
                                 "Agent's thoughts: \n%s" + \
                                 Fore.RESET + \
                                 f"\n{'-' * 30}"

TOOL_RESPONSE_LOG_MESSAGE = f"\n{'-' * 30}\n" + \
                                 Fore.WHITE + \
                                 "Calling Tools: %s\n" + \
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
                 callbacks: list[AsyncCallbackHandler] = None,
                 detailed_logs: bool = False):
        log.debug("Initializing Agent Graph")
        self.llm = llm
        self.tools = tools
        self.callbacks = callbacks or []
        self.detailed_logs = detailed_logs
        self.graph = None

    @abstractmethod
    async def agent_node(self, state: BaseModel) -> BaseModel:
        pass

    @abstractmethod
    async def tool_node(self, state: BaseModel) -> BaseModel:
        pass

    @abstractmethod
    async def conditional_edge(self, state: BaseModel) -> str:
        pass

    async def _build_graph(self, state) -> CompiledGraph:
        log.debug("Building and compiling the Agent Graph")
        graph = StateGraph(state)
        graph.add_node("agent", self.agent_node)
        graph.add_node("tool", self.tool_node)
        graph.add_edge("tool", "agent")
        conditional_edge_possible_outputs = {AgentDecision.TOOL: "tool", AgentDecision.END: "__end__"}
        graph.add_conditional_edges("agent", self.conditional_edge, conditional_edge_possible_outputs)
        graph.set_entry_point("agent")
        self.graph = graph.compile()
        return self.graph
