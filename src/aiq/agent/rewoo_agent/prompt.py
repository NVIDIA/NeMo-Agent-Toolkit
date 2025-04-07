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

PLANNER_SYSTEM_PROMPT = """
For the following task, make plans that can solve the problem step by step. For each plan, indicate \
which external tool together with tool input to retrieve evidence. You can store the evidence into a \
variable #E that can be called by later tools. (Plan, #E1, Plan, #E2, Plan, ...)

You may ask the human to the following tools:

{tools}

The tools should be one of the following: [{tool_names}]

Please note that you don't need to use all the tools. You can use any of the tools you want.
Make sure to follow the pattern: #E = tool_name[tool_input]. Each Plan should be followed by only one #E.

For example,
Task: Who was the CEO of Golden State Warriors in the year represented by the result of substracting 25 from 2023?

Plan: Calculate the result of 2023 minus 25.
#E1 = calculator_subtract[2023, 25]

Plan: Get the year represented by #E1.
#E2 = haystack_chitchat_agent["Response with the result number contained in #E1"]

Plan: Search for the CEO of Golden State Warriors in the year #E2.
#E3 = internet_search["Who was the CEO of Golden State Warriors in the year #E2?"]

Begin!
Describe your plans with rich details.
"""

PLANNER_USER_PROMPT = """
task: {task}
"""

REWOO_PLAN_PATTERN = r"Plan:\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"

rewoo_planner_prompt = ChatPromptTemplate([("system", PLANNER_SYSTEM_PROMPT), ("user", PLANNER_USER_PROMPT)])

SOLVER_SYSTEM_PROMPT = """
Solve the following task or problem. To solve the problem, we have made step-by-step Plan and \
retrieved corresponding Evidence to each Plan. Use them with caution since long evidence might \
contain irrelevant information.

Now solve the question or task according to provided Evidence above. Respond with the answer
directly with no extra words.

"""
SOLVER_USER_PROMPT = """
plan: {plan}
task: {task}

Response:
"""

rewoo_solver_prompt = ChatPromptTemplate([("system", SOLVER_SYSTEM_PROMPT), ("user", SOLVER_USER_PROMPT)])
