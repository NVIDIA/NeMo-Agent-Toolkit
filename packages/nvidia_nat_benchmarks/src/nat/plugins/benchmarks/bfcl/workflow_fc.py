# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BFCL Native FC workflow.

Uses llm.bind_tools(schemas) + ainvoke() to make the LLM produce structured
tool_calls via the native function calling API. Extracts tool_calls from
AIMessage and formats them for BFCL's ast_checker.
"""

import json
import logging

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function

from .config import BFCLFCWorkflowConfig

logger = logging.getLogger(__name__)

# Map BFCL type names to OpenAPI/JSON Schema types
GORILLA_TO_OPENAPI = {
    "integer": "integer",
    "number": "number",
    "float": "number",
    "string": "string",
    "boolean": "boolean",
    "array": "array",
    "dict": "object",
    "object": "object",
    "tuple": "array",
    "any": "string",
    "String": "string",
    "int": "integer",
}


def _convert_bfcl_schema_to_openai(func: dict) -> dict:
    """Convert a BFCL function schema to OpenAI function calling format."""
    params = func.get("parameters", {})
    properties = {}
    for name, prop in params.get("properties", {}).items():
        converted = dict(prop)
        if "type" in converted:
            converted["type"] = GORILLA_TO_OPENAPI.get(converted["type"], converted["type"])
        properties[name] = converted

    return {
        "type": "function",
        "function": {
            "name": func["name"],
            "description": func.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": params.get("required", []),
            },
        },
    }


@register_function(config_type=BFCLFCWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def bfcl_fc_workflow(config: BFCLFCWorkflowConfig, builder: Builder):
    from langchain_core.messages import HumanMessage

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    async def _run(input_json: str) -> str:
        """Single-turn FC: bind tools, call LLM, extract tool_calls."""
        entry = json.loads(input_json)
        functions = entry.get("function", [])
        question_turns = entry.get("question", [[]])

        # Convert BFCL schemas to OpenAI tool format
        tools = [_convert_bfcl_schema_to_openai(f) for f in functions]
        bound_llm = llm.bind_tools(tools)

        # Get user message
        user_content = ""
        for turn in question_turns:
            for msg in turn:
                if msg.get("role") == "user":
                    user_content = msg["content"]

        response = await bound_llm.ainvoke([HumanMessage(content=user_content)])

        if not response.tool_calls:
            # No tool calls — return empty (will be scored as failure for non-irrelevance tests)
            return str(response.content)

        # Format tool calls as BFCL expects:
        # List of function call strings like: func_name(param1=val1, param2=val2)
        call_strs = []
        for tc in response.tool_calls:
            name = tc["name"]
            args = tc["args"]
            # Format as Python function call string for AST parsing
            param_strs = []
            for k, v in args.items():
                param_strs.append(f"{k}={repr(v)}")
            call_strs.append(f"{name}({', '.join(param_strs)})")

        return "[" + ", ".join(call_strs) + "]"

    yield FunctionInfo.from_fn(_run, description=config.description)
