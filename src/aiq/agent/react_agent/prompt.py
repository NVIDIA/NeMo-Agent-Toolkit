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

# flake8: noqa
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder

SYSTEM_PROMPT = """
Answer the following questions as best you can. You may ask the human to use the following tools:

{tools}

You may respond in one of two formats.
Use the following format exactly to ask the human to use a tool:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (if there is no required input, include "Action Input: None")  
Observation: wait for the human to respond with the result from the tool, do not assume the response

... (this Thought/Action/Action Input/Observation can repeat N times. If you do not need to use a tool, or after asking the human to use any tools and waiting for the human to respond, you might know the final answer.)
Use the following format once you have the final answer:

Thought: I now know the final answer
Final Answer: the final answer to the original input question
"""
USER_PROMPT = """
Question: {question}
"""

# This is the prompt - (ReAct Agent prompt)
react_agent_prompt = ChatPromptTemplate([("system", SYSTEM_PROMPT), ("user", USER_PROMPT),
                                         MessagesPlaceholder(variable_name='agent_scratchpad', optional=True)])
