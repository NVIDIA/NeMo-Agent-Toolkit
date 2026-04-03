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
"""BFCL Native FC workflow.

Uses ``llm.bind_tools(schemas)`` + ``ainvoke()`` to make the LLM produce structured
tool_calls via the native function calling API. Extracts tool_calls from
AIMessage and formats them for BFCL's ast_checker.

Supports multi-call collection: if the model returns one tool call at a time (common
with native FC), the workflow loops up to ``max_fc_turns`` times, feeding canned tool
responses back to collect all parallel/sequential calls.
"""

import json
import logging

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function

from ..common.bfcl_helpers import convert_bfcl_schema_to_openai
from ..common.bfcl_helpers import extract_user_message
from ..common.bfcl_helpers import format_tool_calls_as_bfcl
from .config import BFCLFCWorkflowConfig

logger = logging.getLogger(__name__)

# Max turns to collect tool calls (prevents infinite loops)
MAX_FC_TURNS = 10


@register_function(config_type=BFCLFCWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def bfcl_fc_workflow(config: BFCLFCWorkflowConfig, builder: Builder):
    """Register the BFCL Native FC workflow."""
    from langchain_core.messages import AIMessage
    from langchain_core.messages import HumanMessage
    from langchain_core.messages import ToolMessage

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    async def _run(input_json: str) -> str:
        """FC with multi-turn collection: bind tools, loop until model stops calling."""
        entry = json.loads(input_json)
        functions = entry.get("function", [])
        question_turns = entry.get("question", [[]])

        tools = [convert_bfcl_schema_to_openai(f) for f in functions]
        bound_llm = llm.bind_tools(tools)
        user_content = extract_user_message(question_turns)

        messages = [HumanMessage(content=user_content)]
        all_calls = []

        for _turn in range(MAX_FC_TURNS):
            response = await bound_llm.ainvoke(messages)

            if not response.tool_calls:
                break

            # Collect all tool calls from this response
            for tc in response.tool_calls:
                all_calls.append({"name": tc["name"], "args": tc["args"]})

                # Feed canned response back so model can make additional calls
                messages.append(
                    AIMessage(
                        content="",
                        tool_calls=[{
                            "name": tc["name"],
                            "args": tc["args"],
                            "id": tc.get("id", f"call_{_turn}_{tc['name']}"),
                        }],
                    )
                )
                messages.append(
                    ToolMessage(
                        content=f"Successfully executed {tc['name']}.",
                        tool_call_id=tc.get("id", f"call_{_turn}_{tc['name']}"),
                    )
                )

        if all_calls:
            return format_tool_calls_as_bfcl(all_calls)

        # No tool calls found — return raw content for downstream parsing
        content = str(response.content) if response.content else ""
        return content

    yield FunctionInfo.from_fn(_run, description=config.description)
