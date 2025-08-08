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

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

from . import haystack_agent  # noqa: F401, pylint: disable=unused-import
from . import langchain_research_tool  # noqa: F401, pylint: disable=unused-import
from . import llama_index_rag_tool  # noqa: F401, pylint: disable=unused-import

logger = logging.getLogger(__name__)


class MultiFrameworksWorkflowConfig(FunctionBaseConfig, name="multi_frameworks"):
    # Add your custom configuration parameters here
    llm: LLMRef = "nim_llm"
    data_dir: str = "/home/coder/dev/ai-query-engine/examples/frameworks/multi_frameworks/data/"
    research_tool: FunctionRef
    rag_tool: FunctionRef
    chitchat_agent: FunctionRef


@register_function(config_type=MultiFrameworksWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def multi_frameworks_workflow(config: MultiFrameworksWorkflowConfig, builder: Builder):
    # Implement your workflow logic here
    from typing import TypedDict

    from colorama import Fore
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.messages import BaseMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langgraph.graph import END
    from langgraph.graph import StateGraph

    # Use builder to generate framework specific tools and llms
    logger.info("workflow config = %s", config)

    llm = await builder.get_llm(llm_name=config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    research_tool = builder.get_tool(fn_name=config.research_tool, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    rag_tool = builder.get_tool(fn_name=config.rag_tool, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    chitchat_agent = builder.get_tool(fn_name=config.chitchat_agent, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    chat_hist = ChatMessageHistory()

    router_prompt = """
    Given the user input below, classify it as either being about 'Research', 'Retrieve' or 'General' topic.
    Just use one of these words as your response. \
    'Research' - any question related to a need to do research on arxiv papers and get a summary. such as "find research papers about RAG for me" or " what is Compound AI?"...etc
    'Retrieve' - any question related to the topic of AIQ Toolkit or its workflows, especially concerning the particular workflow called multi_frameworks which show case using multiple frameworks such as langchain, llama-index ..etc
    'General' - answering small greeting or chitchat type of questions or everything else that does not fall into any of the above topics.
    User query: {input}
    Classifcation topic:"""  # noqa: E501

    routing_chain = ({
        "input": RunnablePassthrough()
    }
                     | PromptTemplate.from_template(router_prompt)
                     | llm
                     | StrOutputParser())

    supervisor_chain_with_message_history = RunnableWithMessageHistory(
        routing_chain,
        lambda _: chat_hist,
        history_messages_key="chat_history",
    )

    class AgentState(TypedDict):
        """"
            Will hold the agent state in between messages
        """
        input: str
        chat_history: list[BaseMessage] | None
        chosen_worker_agent: str | None
        final_output: str | None

    async def supervisor(state: AgentState):
        query = state["input"]
        chosen_agent = (await supervisor_chain_with_message_history.ainvoke(
            {"input": query},
            {"configurable": {
                "session_id": "unused"
            }},
        ))
        logger.info("%s========== inside **supervisor node**  current status = \n %s", Fore.BLUE, state)

        return {'input': query, "chosen_worker_agent": chosen_agent, "chat_history": chat_hist}

    async def router(state: AgentState):
        """
        Route the response to the appropriate handler
        """

        status = list(state.keys())
        logger.info("========== inside **router node**  current status = \n %s, %s", Fore.CYAN, status)
        if 'final_output' in status:
            route_to = "end"
        elif 'chosen_worker_agent' not in status:
            logger.info(" ############# router to --> supervisor %s", Fore.RESET)
            route_to = "supevisor"
        elif 'chosen_worker_agent' in status:
            logger.info(" ############# router to --> workers %s", Fore.RESET)
            route_to = "workers"
        else:
            route_to = "end"
        return route_to

    async def workers(state: AgentState):
        query = state["input"]
        worker_choice = state["chosen_worker_agent"]
        logger.info("========== inside **workers node**  current status = \n %s, %s", Fore.YELLOW, state)
        if "retrieve" in worker_choice.lower():
            out = (await rag_tool.ainvoke(query))
            output = out
            logger.info("**using rag_tool via llama_index_rag_agent >>> output:  \n %s, %s", output, Fore.RESET)
        elif "general" in worker_choice.lower():
            output = (await chitchat_agent.ainvoke(query))
            logger.info("**using general chitchat chain >>> output:  \n %s, %s", output, Fore.RESET)
        elif 'research' in worker_choice.lower():
            inputs = {"inputs": query}
            output = (await research_tool.ainvoke(inputs))
        else:
            output = ("Apologies, I am not sure what to say, I can answer general questions retrieve info this "
                      "multi_frameworks workflow and answer light coding questions, but nothing more.")
            logger.info("**!!! not suppose to happen, try to debug this >>> output:  \n %s, %s", output, Fore.RESET)

        return {'input': query, "chosen_worker_agent": worker_choice, "chat_history": chat_hist, "final_output": output}

    workflow = StateGraph(AgentState)
    workflow.add_node("supervisor", supervisor)
    workflow.set_entry_point("supervisor")
    workflow.add_node("workers", workers)
    workflow.add_conditional_edges(
        "supervisor",
        router,
        {
            "workers": "workers", "end": END
        },
    )
    workflow.add_edge("supervisor", "workers")
    workflow.add_edge("workers", END)
    app = workflow.compile()

    async def _response_fn(input_message: str) -> str:
        # Process the input_message and generate output

        try:
            logger.debug("Starting agent execution")
            out = (await app.ainvoke({"input": input_message, "chat_history": chat_hist}))
            output = out["final_output"]
            logger.info("final_output : %s ", output)
            return output
        finally:
            logger.debug("Finished agent execution")

    try:
        yield _response_fn
    except GeneratorExit:
        logger.exception("Exited early!", exc_info=True)
    finally:
        logger.debug("Cleaning up multi_frameworks workflow.")
