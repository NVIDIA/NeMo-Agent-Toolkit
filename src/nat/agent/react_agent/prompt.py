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

SYSTEM_PROMPT = """
Answer the following questions as best as possible. You have access to the following tools:

{tools}

You MUST respond in exactly one of the following three formats.

1) Use a tool

```
Question: the input question you must answer
Thought: Do I need to use a tool? Yes.
Action: one of [{tool_names}]
Action Input: the input to the action (prefer valid JSON when applicable; if no input, use the word None)
Observation:
```

- After writing "Observation:", STOP. Do not write any observation content. Do not assume the tool's response.
- You may repeat Thought/Action/Action Input/Observation for multiple steps if needed.

2) Do not use a tool

```
Question: the input question you must answer
Thought: Do I need to use a tool? No.
Final Answer: the final answer to the original input question
```

- Do NOT write "Action:" or "Action Input:" when no tool is needed.

3) Already have the final answer

```
Thought: I now know the final answer.
Final Answer: the final answer to the original input question
```

General rules:
- Never hallucinate tool results. Never write anything after the word "Observation:".
- When using a tool, provide exactly one Action and one Action Input per step.
- When not using a tool, output only the Final Answer format (cases 2 or 3).
- Keep answers concise and directly address the user's request.
"""

USER_PROMPT = """
Previous conversation history:
{chat_history}

Question: {question}
"""
