# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect

import pytest
from tavily import AsyncTavilyClient

from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugins.tavily.search import TavilySearchConfig
from nat.plugins.tavily.search import _build_input_schema
from nat.plugins.tavily.search import _HIDDEN_PARAMS


def test_input_schema_mirrors_sdk_signature():
    """Schema fields == SDK signature params, minus hidden infra-only and var* params."""
    schema = _build_input_schema()
    sig = inspect.signature(AsyncTavilyClient.search)
    expected = {
        n for n, p in sig.parameters.items()
        if n not in _HIDDEN_PARAMS
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    assert set(schema.model_fields) == expected
    assert schema.model_fields["query"].is_required()
    for name, field in schema.model_fields.items():
        if name == "query":
            continue
        assert not field.is_required(), f"{name} should have a default"


def test_input_schema_accepts_query_only():
    schema = _build_input_schema()
    inst = schema(query="hello")
    assert inst.model_dump(exclude_none=True) == {"query": "hello"}


def test_input_schema_accepts_full_payload():
    schema = _build_input_schema()
    inst = schema(
        query="weather sf",
        search_depth="advanced",
        topic="news",
        max_results=3,
        include_domains=["weather.com"],
    )
    dumped = inst.model_dump(exclude_none=True)
    assert dumped["query"] == "weather sf"
    assert dumped["search_depth"] == "advanced"
    assert dumped["max_results"] == 3
    assert dumped["include_domains"] == ["weather.com"]


@pytest.mark.asyncio
async def test_tavily_search_passes_kwargs_through_and_returns_dict(monkeypatch):
    """End-to-end function build: agent kwargs override config defaults; result returned as-is."""
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    captured: dict = {}

    async def fake_search(self, **kwargs):
        captured.update(kwargs)
        return {"query": kwargs.get("query"), "results": [{"url": "https://x", "content": "y"}]}

    monkeypatch.setattr(AsyncTavilyClient, "search", fake_search)

    async with WorkflowBuilder() as builder:
        await builder.add_function(
            name="tavily_search",
            config=TavilySearchConfig(max_results=5, search_depth="basic"),
        )
        fn = await builder.get_function("tavily_search")
        # Per-call max_results=2 should override the config's max_results=5.
        result = await fn.acall_invoke(query="hello", max_results=2)

    assert captured["query"] == "hello"
    assert captured["max_results"] == 2  # agent override wins
    assert captured["search_depth"] == "basic"  # config default flowed through
    assert result == {"query": "hello", "results": [{"url": "https://x", "content": "y"}]}


@pytest.mark.asyncio
async def test_tavily_search_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    async with WorkflowBuilder() as builder:
        with pytest.raises(ValueError, match="Tavily API key"):
            await builder.add_function(
                name="tavily_search",
                config=TavilySearchConfig(),
            )
