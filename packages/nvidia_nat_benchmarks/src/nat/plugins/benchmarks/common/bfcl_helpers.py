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
"""Shared helpers for BFCL benchmark workflows.

Provides common utilities shared across the AST, FC, and ReAct BFCL workflows.
"""

# Map BFCL type names to OpenAPI/JSON Schema types
GORILLA_TO_OPENAPI: dict[str, str] = {
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


def extract_user_message(question_turns: list[list[dict]]) -> str:
    """Extract the last user message from BFCL question turns.

    BFCL stores questions as nested lists of message dicts. This extracts
    the content of the last user-role message across all turns.

    Args:
        question_turns: Nested list of message dicts from BFCL entry["question"].

    Returns:
        The content string of the last user message, or empty string if none found.
    """
    user_content = ""
    for turn in question_turns:
        for msg in turn:
            if msg.get("role") == "user":
                user_content = msg["content"]
    return user_content


def convert_bfcl_schema_to_openai(func: dict) -> dict:
    """Convert a BFCL function schema to OpenAI function calling format.

    BFCL uses its own type names (e.g., "integer", "float", "dict") which
    need to be mapped to OpenAPI/JSON Schema types for the OpenAI tools= API.

    Args:
        func: A BFCL function schema dict with "name", "description", and "parameters".

    Returns:
        An OpenAI-format tool schema dict.
    """
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


def format_tool_calls_as_bfcl(tool_calls: list[dict]) -> str:
    """Format tool call dicts as BFCL-compatible function call strings.

    Converts a list of tool call dicts (with "name" and "args"/"parameters" keys)
    to BFCL's expected AST format: ``[func_name(param1=val1, param2=val2)]``.

    Args:
        tool_calls: List of dicts with "name" and "args" or "parameters" keys.

    Returns:
        A BFCL-format function call string, or empty string if no calls.
    """
    if not tool_calls:
        return ""

    call_strs = []
    for tc in tool_calls:
        name = tc.get("name", tc.get("tool", ""))
        args = tc.get("args", tc.get("parameters", {}))
        param_strs = [f"{k}={repr(v)}" for k, v in args.items()]
        call_strs.append(f"{name}({', '.join(param_strs)})")

    return "[" + ", ".join(call_strs) + "]"
