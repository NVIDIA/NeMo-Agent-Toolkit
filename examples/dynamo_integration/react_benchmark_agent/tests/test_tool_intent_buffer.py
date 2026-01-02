# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Unit tests for ToolIntentBuffer and tool intent stubs.

These tests verify that:
- ToolIntentBuffer correctly records and clears tool intents
- Scenario ID isolation works correctly with contextvars
- record() and clear() are aligned (both use the same scenario ID source)
- Global registry functions work as expected
"""

import pytest
from react_benchmark_agent.tool_intent_stubs import _GLOBAL_INTENT_REGISTRY
from react_benchmark_agent.tool_intent_stubs import ToolIntentBuffer
from react_benchmark_agent.tool_intent_stubs import _current_scenario_id
from react_benchmark_agent.tool_intent_stubs import clear_global_intents
from react_benchmark_agent.tool_intent_stubs import get_current_scenario_id
from react_benchmark_agent.tool_intent_stubs import get_global_intents
from react_benchmark_agent.tool_intent_stubs import set_current_scenario_id


@pytest.fixture(autouse=True)
def clean_global_registry():
    """Clean global registry and reset contextvar before and after each test."""
    _GLOBAL_INTENT_REGISTRY.clear()
    # Reset contextvar to default
    _current_scenario_id.set("current")
    yield
    _GLOBAL_INTENT_REGISTRY.clear()
    _current_scenario_id.set("current")


class TestToolIntentBuffer:
    """Test basic ToolIntentBuffer operations."""

    def test_init_creates_empty_buffer(self):
        """Test that a new buffer starts empty."""
        buffer = ToolIntentBuffer()
        assert buffer.intents == []
        assert buffer.get_intents() == []

    def test_record_single_intent(self):
        """Test recording a single tool intent."""
        buffer = ToolIntentBuffer()
        buffer.record("get_account_balance", {"account_id": "12345"})

        intents = buffer.get_intents()
        assert len(intents) == 1
        assert intents[0]["tool"] == "get_account_balance"
        assert intents[0]["parameters"] == {"account_id": "12345"}

    def test_record_multiple_intents(self):
        """Test recording multiple tool intents."""
        buffer = ToolIntentBuffer()
        buffer.record("tool_a", {"param": "value_a"})
        buffer.record("tool_b", {"param": "value_b"})
        buffer.record("tool_c", {"param": "value_c"})

        intents = buffer.get_intents()
        assert len(intents) == 3
        assert [i["tool"] for i in intents] == ["tool_a", "tool_b", "tool_c"]

    def test_get_intents_returns_copy(self):
        """Test that get_intents returns a copy, not the original list."""
        buffer = ToolIntentBuffer()
        buffer.record("tool_a", {})

        intents = buffer.get_intents()
        intents.append({"tool": "fake", "parameters": {}})

        # Original buffer should be unchanged
        assert len(buffer.get_intents()) == 1

    def test_clear_empties_local_buffer(self):
        """Test that clear() empties the local intent list."""
        buffer = ToolIntentBuffer()
        buffer.record("tool_a", {})
        buffer.record("tool_b", {})

        buffer.clear()

        assert buffer.intents == []
        assert buffer.get_intents() == []


class TestScenarioIdContextVar:
    """Test scenario ID context variable operations."""

    def test_default_scenario_id(self):
        """Test that default scenario ID is 'current'."""
        assert get_current_scenario_id() == "current"

    def test_set_and_get_scenario_id(self):
        """Test setting and getting a custom scenario ID."""
        set_current_scenario_id("test_scenario_123")
        assert get_current_scenario_id() == "test_scenario_123"

    def test_set_scenario_id_returns_token(self):
        """Test that set_current_scenario_id returns a token for reset."""
        token = set_current_scenario_id("test_scenario")
        assert token is not None

    def test_set_scenario_id_initializes_registry(self):
        """Test that setting scenario ID initializes the registry entry."""
        scenario_id = "new_scenario_456"
        set_current_scenario_id(scenario_id)

        assert scenario_id in _GLOBAL_INTENT_REGISTRY
        assert _GLOBAL_INTENT_REGISTRY[scenario_id] == []


class TestGlobalRegistryIntegration:
    """Test ToolIntentBuffer integration with global registry."""

    def test_record_stores_in_global_registry(self):
        """Test that record() stores intents in the global registry."""
        scenario_id = "scenario_abc"
        set_current_scenario_id(scenario_id)

        buffer = ToolIntentBuffer()
        buffer.record("test_tool", {"key": "value"})

        # Check global registry has the intent
        assert scenario_id in _GLOBAL_INTENT_REGISTRY
        assert len(_GLOBAL_INTENT_REGISTRY[scenario_id]) == 1
        assert _GLOBAL_INTENT_REGISTRY[scenario_id][0]["tool"] == "test_tool"

    def test_clear_clears_global_registry_for_current_scenario(self):
        """
        Test that clear() clears intents from global registry using the current
        scenario ID from contextvar.

        This is the key fix: clear() must use get_current_scenario_id() to align
        with how record() stores intents.
        """
        scenario_id = "scenario_xyz"
        set_current_scenario_id(scenario_id)

        buffer = ToolIntentBuffer()
        buffer.record("tool_1", {})
        buffer.record("tool_2", {})

        # Verify intents are in registry
        assert len(_GLOBAL_INTENT_REGISTRY[scenario_id]) == 2

        # Clear should remove from global registry
        buffer.clear()

        # Global registry for this scenario should be empty
        assert _GLOBAL_INTENT_REGISTRY[scenario_id] == []

    def test_record_and_clear_use_same_scenario_id(self):
        """
        Verify that record() and clear() are aligned on the same scenario ID.

        This test ensures the bug fix: previously clear() used self.scenario_id
        while record() used get_current_scenario_id(), causing misalignment.
        """
        # Set scenario ID via contextvar
        scenario_id = "aligned_scenario"
        set_current_scenario_id(scenario_id)

        # Create buffer (no scenario_id parameter anymore)
        buffer = ToolIntentBuffer()
        buffer.record("tool_1", {"param": "a"})
        buffer.record("tool_2", {"param": "b"})

        # Verify intents were stored under the contextvar's scenario ID
        assert len(get_global_intents(scenario_id)) == 2

        # Clear using the same buffer
        buffer.clear()

        # Verify the same scenario's intents are cleared
        assert get_global_intents(scenario_id) == []

    def test_multiple_scenarios_isolation(self):
        """Test that different scenarios maintain isolated intent registries."""
        # Scenario A
        set_current_scenario_id("scenario_a")
        buffer_a = ToolIntentBuffer()
        buffer_a.record("tool_for_a", {})

        # Scenario B
        set_current_scenario_id("scenario_b")
        buffer_b = ToolIntentBuffer()
        buffer_b.record("tool_for_b_1", {})
        buffer_b.record("tool_for_b_2", {})

        # Check isolation
        assert len(get_global_intents("scenario_a")) == 1
        assert len(get_global_intents("scenario_b")) == 2

        # Clear scenario B should not affect scenario A
        buffer_b.clear()
        assert len(get_global_intents("scenario_a")) == 1
        assert len(get_global_intents("scenario_b")) == 0


class TestGlobalIntentFunctions:
    """Test standalone global intent functions."""

    def test_get_global_intents_returns_copy(self):
        """Test that get_global_intents returns a copy."""
        scenario_id = "copy_test"
        set_current_scenario_id(scenario_id)

        buffer = ToolIntentBuffer()
        buffer.record("tool", {})

        intents = get_global_intents(scenario_id)
        intents.append({"tool": "fake", "parameters": {}})

        # Original should be unchanged
        assert len(get_global_intents(scenario_id)) == 1

    def test_get_global_intents_missing_scenario(self):
        """Test that getting intents for non-existent scenario returns empty list."""
        intents = get_global_intents("nonexistent_scenario")
        assert intents == []

    def test_clear_global_intents(self):
        """Test clear_global_intents function."""
        scenario_id = "clear_test"
        set_current_scenario_id(scenario_id)

        buffer = ToolIntentBuffer()
        buffer.record("tool", {})

        assert len(get_global_intents(scenario_id)) == 1

        clear_global_intents(scenario_id)

        assert get_global_intents(scenario_id) == []

    def test_clear_global_intents_nonexistent_scenario(self):
        """Test that clearing non-existent scenario doesn't raise."""
        # Should not raise
        clear_global_intents("does_not_exist")


class TestPermissiveToolInput:
    """Test PermissiveToolInput validation."""

    def test_parse_dict_input(self):
        """Test that dict input is passed through."""
        from react_benchmark_agent.tool_intent_stubs import PermissiveToolInput

        model = PermissiveToolInput(input_params={"key": "value"})
        assert model.input_params == {"key": "value"}

    def test_parse_json_string_input(self):
        """Test that JSON string is parsed to dict."""
        from react_benchmark_agent.tool_intent_stubs import PermissiveToolInput

        model = PermissiveToolInput(input_params='{"key": "value"}')
        assert model.input_params == {"key": "value"}

    def test_parse_single_quote_json_string(self):
        """Test that single-quote JSON string is handled."""
        from react_benchmark_agent.tool_intent_stubs import PermissiveToolInput

        model = PermissiveToolInput(input_params="{'key': 'value'}")
        assert model.input_params == {"key": "value"}

    def test_parse_invalid_string_returns_empty_dict(self):
        """Test that invalid JSON string returns empty dict."""
        from react_benchmark_agent.tool_intent_stubs import PermissiveToolInput

        model = PermissiveToolInput(input_params="not valid json at all")
        assert model.input_params == {}


class TestCreateToolStubFunction:
    """Test create_tool_stub_function."""

    async def test_stub_records_intent(self):
        """Test that tool stub records intent to buffer."""
        from react_benchmark_agent.tool_intent_stubs import create_tool_stub_function

        buffer = ToolIntentBuffer()
        tool_schema = {
            "title": "test_tool",
            "description": "A test tool",
            "properties": {},
            "required": [],
        }

        stub_fn, input_schema, description = create_tool_stub_function(
            tool_schema, buffer, canned_response="Test response"
        )

        # Execute the stub
        result = await stub_fn({"param": "value"})

        # Check intent was recorded
        assert len(buffer.get_intents()) == 1
        assert buffer.get_intents()[0]["tool"] == "test_tool"
        assert buffer.get_intents()[0]["parameters"] == {"param": "value"}

        # Check response
        assert result == "Test response"

    async def test_stub_filters_none_values(self):
        """Test that tool stub filters out None parameter values."""
        from react_benchmark_agent.tool_intent_stubs import create_tool_stub_function

        buffer = ToolIntentBuffer()
        tool_schema = {"title": "test_tool", "description": ""}

        stub_fn, _, _ = create_tool_stub_function(tool_schema, buffer)

        await stub_fn({"valid": "value", "none_param": None, "another": "data"})

        intents = buffer.get_intents()
        assert "none_param" not in intents[0]["parameters"]
        assert intents[0]["parameters"] == {"valid": "value", "another": "data"}

    async def test_stub_handles_nested_params(self):
        """Test that tool stub handles nested 'params' dict from LangChain."""
        from react_benchmark_agent.tool_intent_stubs import create_tool_stub_function

        buffer = ToolIntentBuffer()
        tool_schema = {"title": "test_tool", "description": ""}

        stub_fn, _, _ = create_tool_stub_function(tool_schema, buffer)

        # LangChain sometimes wraps params in a 'params' key
        await stub_fn({"params": {"actual_param": "value"}})

        intents = buffer.get_intents()
        assert intents[0]["parameters"] == {"actual_param": "value"}


class TestMockResponseGeneration:
    """Test _generate_mock_response."""

    def test_generate_string_mock(self):
        """Test mock generation for string type."""
        from react_benchmark_agent.tool_intent_stubs import _generate_mock_response

        schema = {"properties": {"name": {"type": "string"}}}
        result = _generate_mock_response(schema)
        assert result["name"] == "mock_name"

    def test_generate_integer_mock(self):
        """Test mock generation for integer type."""
        from react_benchmark_agent.tool_intent_stubs import _generate_mock_response

        schema = {"properties": {"count": {"type": "integer"}}}
        result = _generate_mock_response(schema)
        assert result["count"] == 100

    def test_generate_number_mock(self):
        """Test mock generation for number type."""
        from react_benchmark_agent.tool_intent_stubs import _generate_mock_response

        schema = {"properties": {"amount": {"type": "number"}}}
        result = _generate_mock_response(schema)
        assert result["amount"] == 100.50

    def test_generate_boolean_mock(self):
        """Test mock generation for boolean type."""
        from react_benchmark_agent.tool_intent_stubs import _generate_mock_response

        schema = {"properties": {"active": {"type": "boolean"}}}
        result = _generate_mock_response(schema)
        assert result["active"] is True

    def test_generate_array_mock(self):
        """Test mock generation for array type."""
        from react_benchmark_agent.tool_intent_stubs import _generate_mock_response

        schema = {"properties": {"items": {"type": "array"}}}
        result = _generate_mock_response(schema)
        assert result["items"] == []

    def test_generate_object_mock(self):
        """Test mock generation for object type."""
        from react_benchmark_agent.tool_intent_stubs import _generate_mock_response

        schema = {"properties": {"data": {"type": "object"}}}
        result = _generate_mock_response(schema)
        assert result["data"] == {}

    def test_generate_multiple_fields_mock(self):
        """Test mock generation with multiple fields."""
        from react_benchmark_agent.tool_intent_stubs import _generate_mock_response

        schema = {
            "properties": {
                "name": {
                    "type": "string"
                },
                "balance": {
                    "type": "number"
                },
                "active": {
                    "type": "boolean"
                },
            }
        }
        result = _generate_mock_response(schema)
        assert "name" in result
        assert "balance" in result
        assert "active" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
