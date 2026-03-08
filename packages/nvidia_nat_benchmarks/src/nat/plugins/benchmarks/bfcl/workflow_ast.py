# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""BFCL AST (prompting) workflow.

Serializes function schemas as text in the system prompt and makes a single
LLM call. The model outputs raw function call text (e.g. `func(param=val)`)
which BFCL's ast_checker can parse.
"""

import json
import logging

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function

from .config import BFCLASTWorkflowConfig

logger = logging.getLogger(__name__)

# System prompt matching BFCL's default AST prompting format
# Source: bfcl.model_handler.constant.DEFAULT_SYSTEM_PROMPT
SYSTEM_PROMPT = (  # noqa: E501 — verbatim from bfcl.model_handler.constant.DEFAULT_SYSTEM_PROMPT
    "You are an expert in composing functions. You are given a question and a set of possible "
    "functions. Based on the question, you will need to make one or more function/tool calls to "
    "achieve the purpose.\n"
    "If none of the functions can be used, point it out. If the given question lacks the "
    "parameters required by the function, also point it out.\n"
    "You should only return the function calls in your response.\n\n"
    "If you decide to invoke any of the function(s), you MUST put it in the format of "
    "[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]\n"
    "You SHOULD NOT include any other text in the response.\n\n"
    "Here is a list of functions in JSON format that you can invoke.\n"
    "{functions_json}\n"
)


@register_function(config_type=BFCLASTWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def bfcl_ast_workflow(config: BFCLASTWorkflowConfig, builder: Builder):
    from langchain_core.messages import HumanMessage
    from langchain_core.messages import SystemMessage

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    async def _run(input_json: str) -> str:
        """Single-turn AST prompting: serialize schemas into prompt, return raw text."""
        entry = json.loads(input_json)
        functions = entry.get("function", [])
        question_turns = entry.get("question", [[]])

        # Build system message with function schemas
        functions_json = json.dumps(functions, indent=2)
        system_msg = SystemMessage(content=SYSTEM_PROMPT.format(functions_json=functions_json))

        # Build user message(s) — BFCL uses the last turn's last message
        user_content = ""
        for turn in question_turns:
            for msg in turn:
                if msg.get("role") == "user":
                    user_content = msg["content"]

        messages = [system_msg, HumanMessage(content=user_content)]
        response = await llm.ainvoke(messages)
        return str(response.content)

    yield FunctionInfo.from_fn(_run, description=config.description)
