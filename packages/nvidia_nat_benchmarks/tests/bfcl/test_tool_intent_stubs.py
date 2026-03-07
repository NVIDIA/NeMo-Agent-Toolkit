# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ToolIntentBuffer and tool intent stubs.

Mirrors the test_tool_intent_buffer.py pattern from react_benchmark_agent,
adapted for our copy at nat.plugins.benchmarks.bfcl.tool_intent_stubs.
"""

import pytest

from nat.plugins.benchmarks.bfcl.tool_intent_stubs import (
    ToolIntentBuffer,
    PermissiveToolInput,
    _GLOBAL_INTENT_REGISTRY,
    _current_scenario_id,
    _generate_mock_response,
    clear_global_intents,
    create_tool_stub_function,
    get_current_scenario_id,
    get_global_intents,
    set_current_scenario_id,
)


@pytest.fixture(autouse=True)
def clean_global_registry():
    """Clean global registry and reset contextvar before and after each test."""
    _GLOBAL_INTENT_REGISTRY.clear()
    _current_scenario_id.set("current")
    yield
    _GLOBAL_INTENT_REGISTRY.clear()
    _current_scenario_id.set("current")


class TestToolIntentBuffer:

    def test_init_empty(self):
        buf = ToolIntentBuffer()
        assert buf.get_intents() == []

    def test_record_single(self):
        buf = ToolIntentBuffer()
        buf.record("get_balance", {"account_id": "123"})
        intents = buf.get_intents()
        assert len(intents) == 1
        assert intents[0] == {"tool": "get_balance", "parameters": {"account_id": "123"}}

    def test_record_multiple(self):
        buf = ToolIntentBuffer()
        buf.record("tool_a", {"p": "1"})
        buf.record("tool_b", {"p": "2"})
        buf.record("tool_c", {"p": "3"})
        assert len(buf.get_intents()) == 3
        assert [i["tool"] for i in buf.get_intents()] == ["tool_a", "tool_b", "tool_c"]

    def test_get_intents_returns_copy(self):
        buf = ToolIntentBuffer()
        buf.record("tool_a", {})
        copy = buf.get_intents()
        copy.append({"tool": "fake", "parameters": {}})
        assert len(buf.get_intents()) == 1

    def test_clear(self):
        buf = ToolIntentBuffer()
        buf.record("tool_a", {})
        buf.record("tool_b", {})
        buf.clear()
        assert buf.get_intents() == []


class TestScenarioIdContextVar:

    def test_default_is_current(self):
        assert get_current_scenario_id() == "current"

    def test_set_and_get(self):
        set_current_scenario_id("test_123")
        assert get_current_scenario_id() == "test_123"

    def test_returns_token(self):
        token = set_current_scenario_id("test")
        assert token is not None

    def test_initializes_registry(self):
        sid = "new_scenario"
        set_current_scenario_id(sid)
        assert sid in _GLOBAL_INTENT_REGISTRY
        assert _GLOBAL_INTENT_REGISTRY[sid] == []


class TestGlobalRegistryIntegration:

    def test_record_stores_globally(self):
        sid = "scenario_abc"
        set_current_scenario_id(sid)
        buf = ToolIntentBuffer()
        buf.record("test_tool", {"key": "val"})
        assert len(_GLOBAL_INTENT_REGISTRY[sid]) == 1
        assert _GLOBAL_INTENT_REGISTRY[sid][0]["tool"] == "test_tool"

    def test_clear_clears_global(self):
        sid = "scenario_xyz"
        set_current_scenario_id(sid)
        buf = ToolIntentBuffer()
        buf.record("t1", {})
        buf.record("t2", {})
        assert len(_GLOBAL_INTENT_REGISTRY[sid]) == 2
        buf.clear()
        assert _GLOBAL_INTENT_REGISTRY[sid] == []

    def test_record_and_clear_aligned(self):
        sid = "aligned"
        set_current_scenario_id(sid)
        buf = ToolIntentBuffer()
        buf.record("t1", {"p": "a"})
        buf.record("t2", {"p": "b"})
        assert len(get_global_intents(sid)) == 2
        buf.clear()
        assert get_global_intents(sid) == []

    def test_multiple_scenarios_isolated(self):
        set_current_scenario_id("a")
        buf_a = ToolIntentBuffer()
        buf_a.record("tool_a", {})

        set_current_scenario_id("b")
        buf_b = ToolIntentBuffer()
        buf_b.record("tool_b1", {})
        buf_b.record("tool_b2", {})

        assert len(get_global_intents("a")) == 1
        assert len(get_global_intents("b")) == 2

        buf_b.clear()
        assert len(get_global_intents("a")) == 1
        assert len(get_global_intents("b")) == 0


class TestGlobalIntentFunctions:

    def test_get_returns_copy(self):
        sid = "copy_test"
        set_current_scenario_id(sid)
        buf = ToolIntentBuffer()
        buf.record("tool", {})
        copy = get_global_intents(sid)
        copy.append({"tool": "fake", "parameters": {}})
        assert len(get_global_intents(sid)) == 1

    def test_get_missing_scenario(self):
        assert get_global_intents("nonexistent") == []

    def test_clear_global(self):
        sid = "clear_test"
        set_current_scenario_id(sid)
        buf = ToolIntentBuffer()
        buf.record("tool", {})
        clear_global_intents(sid)
        assert get_global_intents(sid) == []

    def test_clear_nonexistent_no_error(self):
        clear_global_intents("does_not_exist")


class TestPermissiveToolInput:

    def test_dict_passthrough(self):
        m = PermissiveToolInput(input_params={"key": "value"})
        assert m.input_params == {"key": "value"}

    def test_json_string_parsed(self):
        m = PermissiveToolInput(input_params='{"key": "value"}')
        assert m.input_params == {"key": "value"}

    def test_single_quote_json(self):
        m = PermissiveToolInput(input_params="{'key': 'value'}")
        assert m.input_params == {"key": "value"}

    def test_invalid_string_returns_empty(self):
        m = PermissiveToolInput(input_params="not json")
        assert m.input_params == {}


class TestCreateToolStubFunction:

    @pytest.mark.asyncio
    async def test_stub_records_intent(self):
        buf = ToolIntentBuffer()
        schema = {"title": "test_tool", "description": "A test tool"}
        stub_fn, _, desc = create_tool_stub_function(schema, buf, canned_response="OK")

        result = await stub_fn({"param": "val"})

        assert len(buf.get_intents()) == 1
        assert buf.get_intents()[0]["tool"] == "test_tool"
        assert buf.get_intents()[0]["parameters"] == {"param": "val"}
        assert result == "OK"

    @pytest.mark.asyncio
    async def test_stub_filters_none(self):
        buf = ToolIntentBuffer()
        schema = {"title": "test_tool", "description": ""}
        stub_fn, _, _ = create_tool_stub_function(schema, buf)

        await stub_fn({"valid": "v", "none_param": None})

        params = buf.get_intents()[0]["parameters"]
        assert "none_param" not in params
        assert params == {"valid": "v"}

    @pytest.mark.asyncio
    async def test_stub_handles_nested_params(self):
        buf = ToolIntentBuffer()
        schema = {"title": "test_tool", "description": ""}
        stub_fn, _, _ = create_tool_stub_function(schema, buf)

        await stub_fn({"params": {"actual": "value"}})

        assert buf.get_intents()[0]["parameters"] == {"actual": "value"}


class TestMockResponseGeneration:

    def test_string(self):
        assert _generate_mock_response({"properties": {"n": {"type": "string"}}})["n"] == "mock_n"

    def test_integer(self):
        assert _generate_mock_response({"properties": {"c": {"type": "integer"}}})["c"] == 100

    def test_number(self):
        assert _generate_mock_response({"properties": {"a": {"type": "number"}}})["a"] == 100.50

    def test_boolean(self):
        assert _generate_mock_response({"properties": {"b": {"type": "boolean"}}})["b"] is True

    def test_array(self):
        assert _generate_mock_response({"properties": {"l": {"type": "array"}}})["l"] == []

    def test_object(self):
        assert _generate_mock_response({"properties": {"d": {"type": "object"}}})["d"] == {}

    def test_multiple_fields(self):
        schema = {"properties": {"n": {"type": "string"}, "v": {"type": "integer"}, "b": {"type": "boolean"}}}
        result = _generate_mock_response(schema)
        assert set(result.keys()) == {"n", "v", "b"}
