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

import inspect
import logging
from typing import Any
from typing import Callable
from typing import Union

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import create_model
from tavily import AsyncTavilyClient

from nat.builder.builder import Builder
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.common import SerializableSecretStr
from nat.data_models.function import FunctionGroupBaseConfig

from ._client import build_async_client

logger = logging.getLogger(__name__)

# SDK params we never expose to the LLM. `self` is the bound instance, `kwargs` is the
# SDK's forward-compat escape hatch, and `timeout` is an HTTP-transport concern.
_HIDDEN_PARAMS: frozenset[str] = frozenset({"self", "kwargs", "timeout"})


def _build_input_schema(method: Callable, name: str) -> type[BaseModel]:
    """Generate a pydantic input model from a live SDK method signature."""
    sig = inspect.signature(method)
    fields: dict[str, tuple[Any, Any]] = {}
    for param_name, param in sig.parameters.items():
        if param_name in _HIDDEN_PARAMS:
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        annotation = param.annotation if param.annotation is not inspect.Parameter.empty else Any

        if param.default is inspect.Parameter.empty:
            default = ...
        else:
            default = param.default
            if default is None:
                # SDK uses `T = None` as "use server default" — make Optional so pydantic
                # accepts the None default against a non-Optional Literal/scalar annotation.
                annotation = Union[annotation, None]

        fields[param_name] = (annotation, Field(default=default))

    return create_model(
        name,
        __config__=ConfigDict(arbitrary_types_allowed=True),
        **fields,
    )


# Schemas are a static reflection of the SDK signature; build once at module import so
# test-time monkeypatches of the bound methods don't poison the introspection.
TavilySearchInput: type[BaseModel] = _build_input_schema(AsyncTavilyClient.search, "TavilySearchInput")
TavilyExtractInput: type[BaseModel] = _build_input_schema(AsyncTavilyClient.extract, "TavilyExtractInput")
TavilyCrawlInput: type[BaseModel] = _build_input_schema(AsyncTavilyClient.crawl, "TavilyCrawlInput")
TavilyMapInput: type[BaseModel] = _build_input_schema(AsyncTavilyClient.map, "TavilyMapInput")

_DESCRIPTIONS: dict[str, str] = {
    "search":
        "Search the web with Tavily for up-to-date information. Returns ranked source documents "
        "(URL, title, content snippet, score) and optionally a synthesized answer.",
    "extract":
        "Extract clean text or markdown content from one or more URLs. Use this when you already "
        "have URLs (e.g. from a prior search) and need their full contents.",
    "crawl":
        "Crawl a website starting from a URL, following links to a configurable depth and breadth. "
        "Returns extracted content from each visited page. Use for bulk content retrieval from a "
        "single domain.",
    "map":
        "Discover and list URLs reachable from a starting URL on a domain, without extracting "
        "page content. Faster than crawl. Use to find specific pages on a known site.",
}


class TavilyToolsGroupConfig(FunctionGroupBaseConfig, name="tavily"):
    """Tavily tools group: search, extract, crawl, map.

    All four tools share a single AsyncTavilyClient and one API key. Per-call arguments
    come from the agent and are passed straight through to the SDK.
    """

    api_key: SerializableSecretStr = Field(
        default_factory=lambda: SerializableSecretStr(""),
        description="Tavily API key. Falls back to the TAVILY_API_KEY env var.",
    )


@register_function_group(config_type=TavilyToolsGroupConfig)
async def tavily_tools(config: TavilyToolsGroupConfig, _builder: Builder):
    """Register the `tavily` function group."""
    client = build_async_client(config.api_key)

    async def _search(value: TavilySearchInput) -> dict:
        return await client.search(**value.model_dump(exclude_none=True))

    async def _extract(value: TavilyExtractInput) -> dict:
        return await client.extract(**value.model_dump(exclude_none=True))

    async def _crawl(value: TavilyCrawlInput) -> dict:
        return await client.crawl(**value.model_dump(exclude_none=True))

    async def _map(value: TavilyMapInput) -> dict:
        return await client.map(**value.model_dump(exclude_none=True))

    group = FunctionGroup(config=config)
    group.add_function("search", _search, input_schema=TavilySearchInput, description=_DESCRIPTIONS["search"])
    group.add_function("extract", _extract, input_schema=TavilyExtractInput, description=_DESCRIPTIONS["extract"])
    group.add_function("crawl", _crawl, input_schema=TavilyCrawlInput, description=_DESCRIPTIONS["crawl"])
    group.add_function("map", _map, input_schema=TavilyMapInput, description=_DESCRIPTIONS["map"])

    yield group
