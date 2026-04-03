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

If the model returns ``<tool_call>`` XML (e.g. Qwen 3.5 thinking mode) instead of
structured tool_calls, the XML is parsed into tool calls automatically.
"""

import hashlib
import json
import logging
import re

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function

from ..common.tool_intent_stubs import ToolIntentBuffer
from ..common.tool_intent_stubs import clear_global_intents
from ..common.tool_intent_stubs import set_current_scenario_id
from .config import AgentLeaderboardWorkflowConfig

logger = logging.getLogger(__name__)


def _parse_xml_tool_calls(content: str) -> list[dict] | None:
    """Parse ``<tool_call>`` XML blocks from model output (e.g. Qwen 3.5 thinking mode).

    Returns a list of {"name": ..., "args": {...}} dicts, or None if no tool calls found.
    """
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

    tool_call_blocks = re.findall(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
    if not tool_call_blocks:
        return None

    calls = []
    for block in tool_call_blocks:
        func_match = re.search(r'<function=([^>]+)>', block)
        if not func_match:
            continue
        func_name = func_match.group(1).strip()

        params = {}
        for param_match in re.finditer(
            r'<parameter=([^>]+)>(.*?)</parameter>', block, re.DOTALL
        ):
            param_name = param_match.group(1).strip()
            param_value = param_match.group(2).strip()
            try:
                params[param_name] = json.loads(param_value)
            except (json.JSONDecodeError, ValueError):
                params[param_name] = param_value

        calls.append({"name": func_name, "args": params})

    return calls if calls else None


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
    """Register the Agent Leaderboard workflow."""
    from langchain_core.messages import AIMessage
    from langchain_core.messages import HumanMessage
    from langchain_core.messages import SystemMessage
    from langchain_core.messages import ToolMessage

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
        _token = set_current_scenario_id(scenario_id)
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

                tool_calls = response.tool_calls or []

                # # XML fallback (disabled — switchyard-v2 fixes native tool call parsing)
                # if not tool_calls:
                #     content = str(response.content) if response.content else ""
                #     xml_calls = _parse_xml_tool_calls(content)
                #     if xml_calls:
                #         logger.debug("Parsed %d tool call(s) from XML in response content", len(xml_calls))
                #         tool_calls = [
                #             {"name": c["name"], "args": c["args"], "id": f"xml_{step}_{i}"}
                #             for i, c in enumerate(xml_calls)
                #         ]

                if not tool_calls:
                    break

                for tc in tool_calls:
                    name = tc["name"]
                    args = dict(tc["args"]) if tc["args"] else {}
                    call_id = tc.get("id", f"call_{step}_{name}")

                    intent_buffer.record(name, args)

                    # Canned response
                    canned = f"Successfully executed {name}. Operation completed."
                    messages.append(
                        AIMessage(
                            content="",
                            tool_calls=[{
                                "name": name,
                                "args": args,
                                "id": call_id,
                            }],
                        ))
                    messages.append(ToolMessage(
                        content=canned,
                        tool_call_id=call_id,
                    ))

            # Return captured intents as JSON
            intents = intent_buffer.get_intents()
            return json.dumps(intents)

        finally:
            clear_global_intents(scenario_id)

    yield FunctionInfo.from_fn(_run, description=config.description)
