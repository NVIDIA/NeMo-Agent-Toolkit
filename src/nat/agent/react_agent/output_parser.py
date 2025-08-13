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

import re

from langchain.agents.agent import AgentOutputParser
from langchain_core.agents import AgentAction
from langchain_core.agents import AgentFinish
from langchain_core.exceptions import LangChainException

from .prompt import SYSTEM_PROMPT

FINAL_ANSWER_ACTION = "Final Answer:"
MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE = "Invalid Format: Missing 'Action:' after 'Thought:'"
MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE = "Invalid Format: Missing 'Action Input:' after 'Action:'"
FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE = ("Parsing LLM output produced both a final answer and a parse-able "
                                                  "action:")


class ReActOutputParserException(ValueError, LangChainException):

    def __init__(self,
                 observation=None,
                 missing_action=False,
                 missing_action_input=False,
                 final_answer_and_action=False):
        self.observation = observation
        self.missing_action = missing_action
        self.missing_action_input = missing_action_input
        self.final_answer_and_action = final_answer_and_action


class ReActOutputParser(AgentOutputParser):
    """Parses ReAct-style LLM calls that have a single tool input.

    Expects output to be in one of two formats.

    If the output signals that an action should be taken,
    should be in the below format. This will result in an AgentAction
    being returned.

    ```
    Thought: agent thought here
    Action: search
    Action Input: what is the temperature in SF?
    Observation: Waiting for the tool response...
    ```

    If the output signals that a final answer should be given,
    should be in the below format. This will result in an AgentFinish
    being returned.

    ```
    Thought: agent thought here
    Final Answer: The temperature is 100 degrees
    ```

    """

    def get_format_instructions(self) -> str:
        return SYSTEM_PROMPT

    def parse(self, text: str) -> AgentAction | AgentFinish:
        includes_answer = FINAL_ANSWER_ACTION in text
        regex = r"Action\s*\d*\s*:[\s]*(.*?)\s*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*?)(?=\s*[\n|\s]\s*Observation\b|$)"
        action_match = re.search(regex, text, re.DOTALL)
        if action_match:
            if includes_answer:
                raise ReActOutputParserException(
                    final_answer_and_action=True,
                    observation=f"{FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE}: {text}")
            action = action_match.group(1).strip()
            action_input = action_match.group(2)
            tool_input = action_input.strip(" ")
            tool_input = tool_input.strip('"')

            return AgentAction(action, tool_input, text)

        if includes_answer:
            return AgentFinish({"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text)

        if not re.search(r"Action\s*\d*\s*:[\s]*(.*?)", text, re.DOTALL):
            raise ReActOutputParserException(observation=MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE,
                                             missing_action=True)
        if not re.search(r"[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)", text, re.DOTALL):
            raise ReActOutputParserException(observation=MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
                                             missing_action_input=True)
        raise ReActOutputParserException(f"Could not parse LLM output: `{text}`")

    @property
    def _type(self) -> str:
        return "react-input"
