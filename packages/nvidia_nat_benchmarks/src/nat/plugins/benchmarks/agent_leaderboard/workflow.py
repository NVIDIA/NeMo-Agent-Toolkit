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
"""Agent Leaderboard workflow.

Drives a tool-calling loop using tool stubs from the dataset.
Each scenario's available_tools are converted to OpenAI tool schemas,
bound to the LLM, and tool call intents are captured for TSQ scoring.
"""

import hashlib
import json
import logging

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function

from ..bfcl.tool_intent_stubs import (
    ToolIntentBuffer,
    clear_global_intents,
    get_global_intents,
    set_current_scenario_id,
)
from .config import AgentLeaderboardWorkflowConfig

logger = logging.getLogger(__name__)


def _tool_schema_to_openai(tool: dict) -> dict:
    """Convert an Agent Leaderboard tool schema to OpenAI function calling format."""
    properties = {}
    for name, prop in tool.get("properties", {}).items():
        converted = dict(prop) if isinstance(prop, dict) else {"type": "string", "description": str(prop)}
        properties[name] = converted

    return {
        "type": "function",
        "function": {
            "name": tool.get("title", "unknown"),
            "description": tool.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": tool.get("required", []),
            },
        },
    }


@register_function(
    config_type=AgentLeaderboardWorkflowConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN],
)
async def agent_leaderboard_workflow(config: AgentLeaderboardWorkflowConfig, builder: Builder):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    async def _run(input_json: str) -> str:
        """Run a single scenario: bind stub tools, call LLM, capture intents."""
        entry = json.loads(input_json)
        tools = entry.get("available_tools", [])
        question = entry.get("question", "")

        if not tools:
            return json.dumps([])

        # Scenario isolation
        scenario_id = f"al_{hashlib.md5(input_json[:200].encode()).hexdigest()[:12]}"
        token = set_current_scenario_id(scenario_id)
        clear_global_intents(scenario_id)
        intent_buffer = ToolIntentBuffer()

        try:
            # Convert all domain tools to OpenAI format
            openai_tools = [_tool_schema_to_openai(t) for t in tools]
            bound_llm = llm.bind_tools(openai_tools)

            # Build messages
            messages = []
            if config.system_prompt:
                messages.append(SystemMessage(content=config.system_prompt))
            messages.append(HumanMessage(content=question))

            # Tool-calling loop
            for step in range(config.max_steps):
                response = await bound_llm.ainvoke(messages)

                if not response.tool_calls:
                    break

                for tc in response.tool_calls:
                    name = tc["name"]
                    args = dict(tc["args"]) if tc["args"] else {}

                    intent_buffer.record(name, args)

                    # Canned response
                    canned = f"Successfully executed {name}. Operation completed."
                    messages.append(AIMessage(
                        content="",
                        tool_calls=[{
                            "name": name,
                            "args": args,
                            "id": tc.get("id", f"call_{step}_{name}"),
                        }],
                    ))
                    messages.append(ToolMessage(
                        content=canned,
                        tool_call_id=tc.get("id", f"call_{step}_{name}"),
                    ))

            # Return captured intents as JSON
            intents = intent_buffer.get_intents()
            return json.dumps(intents)

        finally:
            clear_global_intents(scenario_id)

    yield FunctionInfo.from_fn(_run, description=config.description)
