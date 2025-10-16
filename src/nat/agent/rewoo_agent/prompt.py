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

PLANNER_SYSTEM_PROMPT = """
For the following task, make plans that can solve the problem step by step. For each plan, indicate \
which external tool together with tool input to retrieve evidence. You can store the evidence into a \
placeholder #E that can be called by later tools. (Plan, #E1, Plan, #E2, Plan, ...)

The following tools and respective requirements are available to you:

{tools}

The tool calls you make should be one of the following: [{tool_names}]

You are not required to use all the tools listed. Choose only the ones that best fit the needs of each plan step.

Your output must be a JSON array where each element represents one planning step. Each step must be an object with \
exactly two keys:

1. "plan": A string that describes in detail the action or reasoning for that step.

2. "evidence": An object representing the external tool call associated with that plan step. This object must have the \
following keys:

   -"placeholder": A string that identifies the evidence placeholder ("#E1", "#E2", ...). The numbering should \
be sequential based on the order of steps.

   -"tool": A string specifying the name of the external tool used.

   -"tool_input": The input to the tool. This can be a string, array, or object, depending on the requirements of the \
tool. Be careful about type assumptions because the output of former tools might contain noise.

Important instructions:

Do not output any additional text, comments, or markdown formatting.

Do not include any explanation or reasoning text outside of the JSON array.

The output must be a valid JSON array that can be parsed directly.

Here is an example of how a valid JSON output should look:

[
  \'{{
    "plan": "Find Alex's schedule on Sep 25, 2025",
    "evidence": \'{{
      "placeholder": "#E1",
      "tool": "search_calendar",
      "tool_input": ("Alex", "09/25/2025")
    }}\'
  }}\',
  \'{{
    "plan": "Find Bill's schedule on sep 25, 2025",
    "evidence": \'{{
      "placeholder": "#E2",
      "tool": "search_calendar",
      "tool_input": ("Bill", "09/25/2025")
    }}\'
  }}\',
  \'{{
    "plan": "Suggest a time for 1-hour meeting given Alex's and Bill's schedule.",
    "evidence": \'{{
      "placeholder": "#E3",
      "tool": "llm_chat",
      "tool_input": "Find a common 1-hour time slot for Alex and Bill given their schedules. \
Alex's schedule: #E1; Bill's schedule: #E2?"
    }}\'
  }}\'
]

Begin!
"""

PLANNER_USER_PROMPT = """
Previous conversation history:
{chat_history}

task: {task}
"""

SOLVER_SYSTEM_PROMPT = """
Solve the following task or problem. To solve the problem, we have made some Plans ahead and \
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
