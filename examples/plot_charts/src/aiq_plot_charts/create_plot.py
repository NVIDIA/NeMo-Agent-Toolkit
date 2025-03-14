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
import os

from colorama import Fore
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.language_models import LLM
from langchain_core.runnables.history import RunnableWithMessageHistory

from .graph_instruction import LINE_GRAPH_INSTRUCTION
from .plot_chain_agent import PlotChartAgents
from .plot_chain_agent import plot_bar_plot
from .plot_chain_agent import plot_line_chart

load_dotenv()
nvapi_key = os.environ["NVIDIA_API_KEY"]
logger = logging.getLogger(__name__)


class DrawPlotAgent:
    """
    Implementation of the Vred agent + retriever + memory & chat history + routing of topic per user query to
    (1) retriever (2) tool/action command (3) geenral chitchat
    """

    def __init__(self, llm: LLM):
        """
            Initialize the XPlane agent and create appropriate LangGraph workflow
        """
        self.llm = llm
        self.agent = PlotChartAgents(llm)
        self.chat_history_internal = ChatMessageHistory()
        self.router = self.agent.routing_chain
        self.routing_chain_with_message_history = RunnableWithMessageHistory(
            self.router,
            lambda session_id: self.chat_history_internal,
            history_messages_key="chat_history",
        )

        # make line chart creation lcel chain as line_chart_agent
        self.line_chart_agent = self.agent.line_graph_creator

        # make bar chart creation lcel chain as bar_chart_agent
        self.bar_chart_agent = self.agent.bar_plot_tool_chain
        # make general chitchat agent
        self.chitchat = self.agent.general_chain

    def run(self, user_message, data):
        routed_output = self.router.invoke({"input": user_message}, {"configurable": {"session_id": "unused"}})
        logger.info("%srouted_output=%s", Fore.BLUE, routed_output)
        if 'line_chart' in routed_output.lower():
            try:
                output = self.line_chart_agent.invoke({
                    "data": data,
                    "lineGraphIntstruction": LINE_GRAPH_INSTRUCTION,
                    'chat_history': self.chat_history_internal.messages
                })
                logger.info("%s**line_chart**%s", Fore.BLUE, output)
                img_path = plot_line_chart(
                    output.xValues,
                    output.yValues,
                    output.chart_name,
                    llm=self.llm,
                    save_fig=True,
                )
                bot_message = f"line chart is generated, the image path can be found here : {img_path}"
                logger.info("%s**line_chart**%s", Fore.BLUE, output)
            except Exception:
                bot_message = "something went wrong, clear the memory and try again !"
                logger.exception("%sEXCEPTION !!! **bot_message** %s", Fore.BLUE, bot_message, exc_info=True)
                img_path = ""
                pass

        elif 'bar_chart' in routed_output.lower():
            try:
                output = self.bar_chart_agent.invoke({
                    "data": data,
                    "bar_instruction": self.agent.bar_instruction,
                    'chat_history': self.chat_history_internal.messages
                })
                logger.info("%s**bar_chart %s", Fore.CYAN, output)
                img_path = plot_bar_plot(output.xValues, output.yValues, output.chart_name, llm=self.llm, save_fig=True)
                bot_message = f"bar chart is generated, the image path can be found here : {img_path}"
                logger.info("%s**bot_message** %s", Fore.CYAN, bot_message)
            except Exception:
                bot_message = "something went wrong, clear the memory and try again !"
                logger.exception("%sEXCEPTION!!!**bot_message** %s", Fore.CYAN, bot_message, exc_info=True)
                img_path = ""
        else:
            output = self.chitchat.invoke({
                "input": user_message, 'chat_history': self.chat_history_internal.messages
            }).content
            img_path = None
            logger.info("%s**chitchat**%s", Fore.GREEN, output)
            bot_message = output

        populate_state_d = {
            "input": user_message,
            "invoked_chain": routed_output,
            "chat_history": self.chat_history_internal.messages,
            "bot_message": bot_message,
            "img_path": img_path
        }
        return populate_state_d
