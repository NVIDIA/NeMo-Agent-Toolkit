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
"""Tests for common.bfcl_helpers — shared BFCL workflow utilities."""

from nat.plugins.benchmarks.common.bfcl_helpers import GORILLA_TO_OPENAPI
from nat.plugins.benchmarks.common.bfcl_helpers import convert_bfcl_schema_to_openai
from nat.plugins.benchmarks.common.bfcl_helpers import extract_user_message
from nat.plugins.benchmarks.common.bfcl_helpers import format_tool_calls_as_bfcl


class TestExtractUserMessage:
    """Tests for extract_user_message."""

    def test_single_turn_single_message(self):
        turns = [[{"role": "user", "content": "What is the weather?"}]]
        assert extract_user_message(turns) == "What is the weather?"

    def test_multi_turn_returns_last_user(self):
        turns = [
            [{
                "role": "user", "content": "First question"
            }],
            [{
                "role": "user", "content": "Second question"
            }],
        ]
        assert extract_user_message(turns) == "Second question"

    def test_mixed_roles_returns_last_user(self):
        turns = [[
            {
                "role": "system", "content": "You are helpful"
            },
            {
                "role": "user", "content": "Hello"
            },
            {
                "role": "assistant", "content": "Hi there"
            },
            {
                "role": "user", "content": "What time is it?"
            },
        ]]
        assert extract_user_message(turns) == "What time is it?"

    def test_no_user_message_returns_empty(self):
        turns = [[{"role": "system", "content": "You are a bot"}]]
        assert extract_user_message(turns) == ""

    def test_empty_turns(self):
        assert extract_user_message([]) == ""
        assert extract_user_message([[]]) == ""

    def test_nested_empty(self):
        turns = [[], [{"role": "user", "content": "Found it"}]]
        assert extract_user_message(turns) == "Found it"


class TestConvertBfclSchemaToOpenai:
    """Tests for convert_bfcl_schema_to_openai."""

    def test_basic_schema(self):
        func = {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {
                "properties": {
                    "city": {
                        "type": "string", "description": "City name"
                    },
                },
                "required": ["city"],
            },
        }
        result = convert_bfcl_schema_to_openai(func)

        assert result["type"] == "function"
        assert result["function"]["name"] == "get_weather"
        assert result["function"]["description"] == "Get the weather"
        assert result["function"]["parameters"]["type"] == "object"
        assert result["function"]["parameters"]["properties"]["city"]["type"] == "string"
        assert result["function"]["parameters"]["required"] == ["city"]

    def test_gorilla_type_mapping(self):
        """BFCL-specific types like 'float', 'dict', 'int' are mapped to OpenAPI types."""
        func = {
            "name": "calculate",
            "parameters": {
                "properties": {
                    "value": {
                        "type": "float"
                    },
                    "config": {
                        "type": "dict"
                    },
                    "count": {
                        "type": "int"
                    },
                    "items": {
                        "type": "tuple"
                    },
                },
                "required": [],
            },
        }
        result = convert_bfcl_schema_to_openai(func)
        props = result["function"]["parameters"]["properties"]

        assert props["value"]["type"] == "number"
        assert props["config"]["type"] == "object"
        assert props["count"]["type"] == "integer"
        assert props["items"]["type"] == "array"

    def test_unknown_type_preserved(self):
        """Types not in GORILLA_TO_OPENAPI are passed through."""
        func = {
            "name": "fn",
            "parameters": {
                "properties": {
                    "x": {
                        "type": "custom_type"
                    }
                },
                "required": [],
            },
        }
        result = convert_bfcl_schema_to_openai(func)
        assert result["function"]["parameters"]["properties"]["x"]["type"] == "custom_type"

    def test_missing_fields_have_defaults(self):
        """Missing description and required default to empty."""
        func = {"name": "fn", "parameters": {"properties": {}}}
        result = convert_bfcl_schema_to_openai(func)

        assert result["function"]["description"] == ""
        assert result["function"]["parameters"]["required"] == []
        assert result["function"]["parameters"]["properties"] == {}

    def test_no_parameters_key(self):
        """Function with no parameters key at all."""
        func = {"name": "noop"}
        result = convert_bfcl_schema_to_openai(func)

        assert result["function"]["name"] == "noop"
        assert result["function"]["parameters"]["properties"] == {}

    def test_preserves_extra_property_fields(self):
        """Extra fields like 'description' and 'enum' on properties are preserved."""
        func = {
            "name": "fn",
            "parameters": {
                "properties": {
                    "color": {
                        "type": "string",
                        "description": "The color",
                        "enum": ["red", "blue"],
                    },
                },
                "required": ["color"],
            },
        }
        result = convert_bfcl_schema_to_openai(func)
        color_prop = result["function"]["parameters"]["properties"]["color"]

        assert color_prop["type"] == "string"
        assert color_prop["description"] == "The color"
        assert color_prop["enum"] == ["red", "blue"]


class TestFormatToolCallsAsBfcl:
    """Tests for format_tool_calls_as_bfcl."""

    def test_single_call_with_name_args(self):
        calls = [{"name": "get_weather", "args": {"city": "SF"}}]
        result = format_tool_calls_as_bfcl(calls)
        assert result == "[get_weather(city='SF')]"

    def test_single_call_with_tool_parameters(self):
        """Also works with 'tool'/'parameters' keys (intent format)."""
        calls = [{"tool": "get_weather", "parameters": {"city": "SF"}}]
        result = format_tool_calls_as_bfcl(calls)
        assert result == "[get_weather(city='SF')]"

    def test_multiple_calls(self):
        calls = [
            {
                "name": "fn_a", "args": {
                    "x": 1
                }
            },
            {
                "name": "fn_b", "args": {
                    "y": "hello"
                }
            },
        ]
        result = format_tool_calls_as_bfcl(calls)
        assert result == "[fn_a(x=1), fn_b(y='hello')]"

    def test_no_args(self):
        calls = [{"name": "noop", "args": {}}]
        result = format_tool_calls_as_bfcl(calls)
        assert result == "[noop()]"

    def test_empty_list(self):
        assert format_tool_calls_as_bfcl([]) == ""

    def test_multiple_params(self):
        calls = [{"name": "search", "args": {"query": "test", "limit": 10, "exact": True}}]
        result = format_tool_calls_as_bfcl(calls)
        assert "search(" in result
        assert "query='test'" in result
        assert "limit=10" in result
        assert "exact=True" in result

    def test_nested_values(self):
        """Nested dicts/lists use repr formatting."""
        calls = [{"name": "fn", "args": {"data": [1, 2, 3]}}]
        result = format_tool_calls_as_bfcl(calls)
        assert result == "[fn(data=[1, 2, 3])]"


class TestGorillaToOpenapi:
    """Tests for the GORILLA_TO_OPENAPI mapping."""

    def test_all_gorilla_types_mapped(self):
        expected = {
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
        assert GORILLA_TO_OPENAPI == expected

    def test_standard_types_are_identity(self):
        """Standard JSON Schema types map to themselves."""
        for t in ("integer", "number", "string", "boolean", "array", "object"):
            assert GORILLA_TO_OPENAPI[t] == t
