# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.agent import AgentBaseConfig

from .tool_intent_stubs import ToolIntentBuffer, set_current_scenario_id, clear_global_intents
from .workflow_fc import _convert_bfcl_schema_to_openai, GORILLA_TO_OPENAPI

logger = logging.getLogger(__name__)


class BFCLReActWorkflowConfig(AgentBaseConfig, name="bfcl_react_workflow"):
    """Workflow config for BFCL ReAct evaluation.

    Uses a ReAct-style loop: the LLM reasons, calls tools (stubs), observes
    results, and continues until it produces a final answer or reaches max steps.
    Tool call intents are captured for BFCL scoring.
    """

    description: str = Field(default="BFCL ReAct Workflow")
    max_steps: int = Field(
        default=5,
        description="Maximum number of ReAct reasoning/action steps per item",
    )


@register_function(config_type=BFCLReActWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def bfcl_react_workflow(config: BFCLReActWorkflowConfig, builder: Builder):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    async def _run(input_json: str) -> str:
        """ReAct loop: reason → call tool stubs → observe → repeat → output intents."""
        entry = json.loads(input_json)
        functions = entry.get("function", [])
        question_turns = entry.get("question", [[]])

        # Set up scenario isolation for concurrent execution
        scenario_id = f"bfcl_{hashlib.md5(input_json[:200].encode()).hexdigest()[:12]}"
        token = set_current_scenario_id(scenario_id)
        clear_global_intents(scenario_id)
        intent_buffer = ToolIntentBuffer()

        try:
            # Convert BFCL schemas to OpenAI tool format
            tools = [_convert_bfcl_schema_to_openai(f) for f in functions]
            bound_llm = llm.bind_tools(tools)

            # Build type map for coercing args to expected types
            param_types = {}  # {func_name: {param_name: bfcl_type}}
            for f in functions:
                props = f.get("parameters", {}).get("properties", {})
                param_types[f["name"]] = {k: v.get("type", "string") for k, v in props.items()}

            # Get user message
            user_content = ""
            for turn in question_turns:
                for msg in turn:
                    if msg.get("role") == "user":
                        user_content = msg["content"]

            # Build initial messages
            system_msg = SystemMessage(
                content="You are a helpful assistant. Use the provided tools to answer the user's request. "
                "Think step by step about which tool(s) to call and with what parameters."
            )
            messages = [system_msg, HumanMessage(content=user_content)]

            # ReAct loop
            for step in range(config.max_steps):
                response = await bound_llm.ainvoke(messages)

                if not response.tool_calls:
                    # Agent decided it's done — no more tool calls
                    break

                # Process all tool calls in this step
                for tc in response.tool_calls:
                    name = tc["name"]
                    args = dict(tc["args"])

                    # Coerce args to expected types from BFCL schema
                    types = param_types.get(name, {})
                    for k, v in args.items():
                        expected = types.get(k, "string")
                        if expected in ("integer", "int") and isinstance(v, str):
                            try:
                                args[k] = int(v)
                            except ValueError:
                                pass
                        elif expected in ("float", "number") and isinstance(v, str):
                            try:
                                args[k] = float(v)
                            except ValueError:
                                pass
                        elif expected == "boolean" and isinstance(v, str):
                            args[k] = v.lower() in ("true", "1", "yes")

                    # Record intent
                    intent_buffer.record(name, args)

                    # Return canned response so the agent can continue
                    canned = f"Successfully executed {name}. Operation completed."

                    # Add the assistant message with tool call + tool response
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

            # Format captured intents as BFCL-compatible output
            intents = intent_buffer.get_intents()
            if not intents:
                return ""

            # Deduplicate intents (same tool + same params = single call)
            seen = set()
            unique_intents = []
            for intent in intents:
                key = (intent["tool"], json.dumps(intent["parameters"], sort_keys=True))
                if key not in seen:
                    seen.add(key)
                    unique_intents.append(intent)

            # Format as Python function call strings for AST parsing
            call_strs = []
            for intent in unique_intents:
                name = intent["tool"]
                params = intent["parameters"]
                param_strs = [f"{k}={repr(v)}" for k, v in params.items()]
                call_strs.append(f"{name}({', '.join(param_strs)})")

            return "[" + ", ".join(call_strs) + "]"

        finally:
            clear_global_intents(scenario_id)

    yield FunctionInfo.from_fn(_run, description=config.description)
