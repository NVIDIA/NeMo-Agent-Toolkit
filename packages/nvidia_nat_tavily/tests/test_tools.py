# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect

import pytest
from tavily import AsyncTavilyClient

from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugins.tavily.tools import TavilyCrawlInput
from nat.plugins.tavily.tools import TavilyExtractInput
from nat.plugins.tavily.tools import TavilyMapInput
from nat.plugins.tavily.tools import TavilySearchInput
from nat.plugins.tavily.tools import TavilyToolsGroupConfig
from nat.plugins.tavily.tools import _build_input_schema
from nat.plugins.tavily.tools import _HIDDEN_PARAMS


def _expected_fields(method) -> set[str]:
    sig = inspect.signature(method)
    return {
        name for name, p in sig.parameters.items()
        if name not in _HIDDEN_PARAMS
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }


@pytest.mark.parametrize("schema, method, required_field", [
    (TavilySearchInput, AsyncTavilyClient.search, "query"),
    (TavilyExtractInput, AsyncTavilyClient.extract, "urls"),
    (TavilyCrawlInput, AsyncTavilyClient.crawl, "url"),
    (TavilyMapInput, AsyncTavilyClient.map, "url"),
])
def test_schema_mirrors_sdk_signature(schema, method, required_field):
    assert set(schema.model_fields) == _expected_fields(method)
    assert schema.model_fields[required_field].is_required()
    for name, field in schema.model_fields.items():
        if name == required_field:
            continue
        assert not field.is_required(), f"{schema.__name__}.{name} should default"


def test_schemas_are_independent():
    """Each tool's schema is its own class; they don't collide."""
    classes = {TavilySearchInput, TavilyExtractInput, TavilyCrawlInput, TavilyMapInput}
    assert len(classes) == 4


def test_build_input_schema_skips_var_kwargs():
    """**kwargs and timeout are dropped from the LLM-facing surface."""
    schema = _build_input_schema(AsyncTavilyClient.search, "S")
    assert "kwargs" not in schema.model_fields
    assert "timeout" not in schema.model_fields
    assert "self" not in schema.model_fields


@pytest.mark.asyncio
@pytest.mark.parametrize("tool, method_name, payload, required_kwargs", [
    ("search", "search", {"query": "weather sf"}, {"query": "weather sf"}),
    ("extract", "extract", {"urls": ["https://example.com"]}, {"urls": ["https://example.com"]}),
    ("crawl", "crawl", {"url": "https://example.com", "limit": 5}, {"url": "https://example.com", "limit": 5}),
    ("map", "map", {"url": "https://example.com"}, {"url": "https://example.com"}),
])
async def test_each_tool_routes_to_correct_sdk_method(monkeypatch, tool, method_name, payload, required_kwargs):
    """Each group tool calls its corresponding AsyncTavilyClient method with the agent kwargs."""
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    captured: dict = {}

    async def fake(self, **kwargs):
        captured["called"] = method_name
        captured["kwargs"] = kwargs
        return {"ok": True, "tool": method_name}

    monkeypatch.setattr(AsyncTavilyClient, method_name, fake)

    async with WorkflowBuilder() as builder:
        await builder.add_function_group(name="tavily", config=TavilyToolsGroupConfig())
        group = await builder.get_function_group("tavily")
        fns = await group.get_all_functions()
        fn = fns[f"tavily__{tool}"]
        result = await fn.acall_invoke(**payload)

    assert captured["called"] == method_name
    for k, v in required_kwargs.items():
        assert captured["kwargs"][k] == v
    assert result == {"ok": True, "tool": method_name}


@pytest.mark.asyncio
async def test_group_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    async with WorkflowBuilder() as builder:
        with pytest.raises(ValueError, match="Tavily API key"):
            await builder.add_function_group(name="tavily", config=TavilyToolsGroupConfig())


@pytest.mark.asyncio
async def test_group_exposes_all_four_tools(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    async with WorkflowBuilder() as builder:
        await builder.add_function_group(name="tavily", config=TavilyToolsGroupConfig())
        group = await builder.get_function_group("tavily")
        functions = await group.get_all_functions()
        names = set(functions.keys())

    assert names == {"tavily__search", "tavily__extract", "tavily__crawl", "tavily__map"}
