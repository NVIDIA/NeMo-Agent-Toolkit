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
"""BFCL ReAct workflow.

Drives a ReAct-style tool-calling loop using NAT's LLM with bind_tools().
Tool calls are captured via ToolIntentBuffer and formatted as BFCL-compatible
AST output for scoring. This demonstrates NAT-native agent execution against
BFCL benchmarks — the agent reasons step-by-step before making tool calls,
and tool stubs return canned responses so the agent can continue reasoning.
"""

import hashlib
import json
import logging

from pydantic import Field
from pydantic import PositiveInt

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.agent import AgentBaseConfig

from ..common.bfcl_helpers import convert_bfcl_schema_to_openai
from ..common.bfcl_helpers import extract_user_message
from ..common.bfcl_helpers import format_tool_calls_as_bfcl
from ..common.tool_intent_stubs import ToolIntentBuffer
from ..common.tool_intent_stubs import clear_global_intents
from ..common.tool_intent_stubs import set_current_scenario_id

logger = logging.getLogger(__name__)


class BFCLReActWorkflowConfig(AgentBaseConfig, name="bfcl_react_workflow"):
    """Workflow config for BFCL ReAct evaluation.

    Uses a ReAct-style loop: the LLM reasons, calls tools (stubs), observes
    results, and continues until it produces a final answer or reaches max steps.
    Tool call intents are captured for BFCL scoring.
    """

    description: str = Field(default="BFCL ReAct Workflow")
    max_steps: PositiveInt = Field(
        default=5,
        description="Maximum number of ReAct reasoning/action steps per item",
    )


def _coerce_args(args: dict, param_types: dict[str, str]) -> dict:
    """Coerce tool call arguments to expected types from BFCL schema.

    LLMs sometimes return string representations of integers/floats/booleans.
    This converts them to the expected Python types based on the schema.

    Args:
        args: The raw arguments dict from the LLM tool call.
        param_types: Mapping of parameter name to BFCL type string.

    Returns:
        A new dict with coerced values.
    """
    coerced = dict(args)
    for k, v in coerced.items():
        expected = param_types.get(k, "string")
        if expected in ("integer", "int") and isinstance(v, str):
            try:
                coerced[k] = int(v)
            except ValueError:
                pass
        elif expected in ("float", "number") and isinstance(v, str):
            try:
                coerced[k] = float(v)
            except ValueError:
                pass
        elif expected == "boolean" and isinstance(v, str):
            coerced[k] = v.lower() in ("true", "1", "yes")
    return coerced


def _deduplicate_intents(intents: list[dict]) -> list[dict]:
    """Deduplicate intents (same tool + same params = single call).

    Args:
        intents: List of intent dicts with "tool" and "parameters" keys.

    Returns:
        Deduplicated list preserving first occurrence order.
    """
    seen: set[tuple[str, str]] = set()
    unique = []
    for intent in intents:
        key = (intent["tool"], json.dumps(intent["parameters"], sort_keys=True))
        if key not in seen:
            seen.add(key)
            unique.append(intent)
    return unique


@register_function(config_type=BFCLReActWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def bfcl_react_workflow(config: BFCLReActWorkflowConfig, builder: Builder):
    """Register the BFCL ReAct workflow."""
    from langchain_core.messages import AIMessage
    from langchain_core.messages import HumanMessage
    from langchain_core.messages import SystemMessage
    from langchain_core.messages import ToolMessage

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    async def _run(input_json: str) -> str:
        """ReAct loop: reason -> call tool stubs -> observe -> repeat -> output intents."""
        entry = json.loads(input_json)
        functions = entry.get("function", [])
        question_turns = entry.get("question", [[]])

        # Set up scenario isolation for concurrent execution
        scenario_id = f"bfcl_{hashlib.md5(input_json[:200].encode()).hexdigest()[:12]}"
        _token = set_current_scenario_id(scenario_id)
        clear_global_intents(scenario_id)
        intent_buffer = ToolIntentBuffer()

        try:
            tools = [convert_bfcl_schema_to_openai(f) for f in functions]
            bound_llm = llm.bind_tools(tools)

            # Build type map for coercing args to expected types
            param_types: dict[str, dict[str, str]] = {}
            for f in functions:
                props = f.get("parameters", {}).get("properties", {})
                param_types[f["name"]] = {k: v.get("type", "string") for k, v in props.items()}

            user_content = extract_user_message(question_turns)

            messages = [
                SystemMessage(
                    content="You are a helpful assistant. Use the provided tools to answer the user's request. "
                    "Think step by step about which tool(s) to call and with what parameters."),
                HumanMessage(content=user_content),
            ]

            for step in range(config.max_steps):
                response = await bound_llm.ainvoke(messages)

                if not response.tool_calls:
                    break

                for tc in response.tool_calls:
                    name = tc["name"]
                    args = _coerce_args(dict(tc["args"]), param_types.get(name, {}))

                    intent_buffer.record(name, args)

                    canned = f"Successfully executed {name}. Operation completed."
                    tc_id = tc.get("id", f"call_{step}_{name}")
                    messages.append(AIMessage(
                        content="",
                        tool_calls=[{
                            "name": name, "args": args, "id": tc_id
                        }],
                    ))
                    messages.append(ToolMessage(content=canned, tool_call_id=tc_id))

            intents = _deduplicate_intents(intent_buffer.get_intents())
            return format_tool_calls_as_bfcl(intents)

        finally:
            clear_global_intents(scenario_id)

    yield FunctionInfo.from_fn(_run, description=config.description)
