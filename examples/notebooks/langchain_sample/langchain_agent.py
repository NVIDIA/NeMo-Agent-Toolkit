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
import os

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Initialize a tool to search the web
tavily_kwargs = {"max_results": 2, "api_key": os.getenv("TAVILY_API_KEY")}
search = TavilySearchResults(**tavily_kwargs)

# Create a list of tools for the agent
tools = [search]

# Initialize a LLM client
llm_kwargs = {
    "model_name": "meta/llama-3.3-70b-instruct",
    "temperature": 0.0,
    "max_tokens": 1024,
    "api_key": os.getenv("NVIDIA_API_KEY"),
}
llm = ChatNVIDIA(**llm_kwargs)

# Use an open source prompt
prompt = hub.pull("hwchase17/react-chat")

# Initialize a ReAct agent
react_agent = create_react_agent(llm=llm, tools=tools, prompt=prompt, stop_sequence=["\nObservation"])

# Initialize an agent executor to iterate through reasoning steps
agent_executor = AgentExecutor(agent=react_agent,
                               tools=tools,
                               max_iterations=15,
                               handle_parsing_errors=True,
                               verbose=True)

# Invoke the agent with a user query
response = agent_executor.invoke({"input": "Who is the current Pope?", "chat_history": []})

# Print the response
print(response["output"])
