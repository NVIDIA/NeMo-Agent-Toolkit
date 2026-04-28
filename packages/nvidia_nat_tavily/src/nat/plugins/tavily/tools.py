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
import json
import logging
import types
import typing
from collections.abc import Sequence as AbcSequence
from typing import Annotated
from typing import Any
from typing import AsyncIterator
from typing import Callable
from typing import Union

from pydantic import BaseModel
from pydantic import BeforeValidator
from pydantic import ConfigDict
from pydantic import Field
from pydantic import create_model

from nat.builder.builder import Builder
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.common import SerializableSecretStr
from nat.data_models.function import FunctionGroupBaseConfig
from tavily import AsyncTavilyClient

from ._client import build_async_client
from .parse_streaming import _accumulate_research_stream

logger = logging.getLogger(__name__)

# SDK params we never expose to the LLM. `self` is the bound instance, `kwargs` is the
# SDK's forward-compat escape hatch, `timeout` is an HTTP-transport concern, and `stream`
# is locked to True internally for `research` (the wrapper consumes the SSE stream).
_HIDDEN_PARAMS: frozenset[str] = frozenset({"self", "kwargs", "timeout", "stream"})

_LIST_ORIGINS: tuple[Any, ...] = (list, tuple, set, frozenset, AbcSequence)


def _annotation_accepts_list(annotation: Any) -> bool:
    """True if `annotation` is (or contains, in a Union) a list/sequence type."""
    origin = typing.get_origin(annotation)
    if origin in _LIST_ORIGINS:
        return True
    if origin in (Union, types.UnionType):
        return any(_annotation_accepts_list(a) for a in typing.get_args(annotation))
    return False


def _coerce_json_list(value: Any) -> Any:
    """Decode a JSON-encoded list string into a Python list.

    LLMs (and some framework tool wrappers) sometimes emit array-typed args as JSON
    strings — e.g. `urls='["https://a", "https://b"]'`. This pre-validator unwraps
    them before pydantic dispatches Union arms; without it, pydantic accepts the
    string under the `str` arm of `Union[List[str], str]` and the SDK then fails
    URL validation downstream.
    """
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError:
                return value
            if isinstance(parsed, list):
                return parsed
    return value


def _build_input_schema(method: Callable, name: str) -> type[BaseModel]:
    """Generate a pydantic input model from a live SDK method signature.

    List-typed fields are wrapped in a BeforeValidator that JSON-decodes
    string-encoded lists from the LLM.
    """
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

        if _annotation_accepts_list(annotation):
            annotation = Annotated[annotation, BeforeValidator(_coerce_json_list)]

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
TavilyResearchInput: type[BaseModel] = _build_input_schema(AsyncTavilyClient.research, "TavilyResearchInput")

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
    "research":
        "Run a deep, multi-source research task with citations. Returns a synthesized report "
        "(markdown or structured per `output_schema`) plus a list of sources. Slow (30-120s) — "
        "use only when the user explicitly asks for research-depth output. For quick fact-finding "
        "use `search`.",
}


class TavilyToolsGroupConfig(FunctionGroupBaseConfig, name="tavily"):
    """Tavily tools group: search, extract, crawl, map, research.

    All tools share a single AsyncTavilyClient and one API key. Per-call arguments
    come from the agent and are passed straight through to the SDK.
    """

    api_key: SerializableSecretStr = Field(
        default_factory=lambda: SerializableSecretStr(""),
        description="Tavily API key. Falls back to the TAVILY_API_KEY env var.",
    )

    research_timeout_seconds: float | None = Field(
        default=900.0,
        gt=0,
        description="HTTP timeout for the research SSE stream. None disables the client-side cap.",
    )
    research_include_trace: bool = Field(
        default=False,
        description="If True, include intermediate tool_call/tool_response events from the research "
        "stream in the returned dict (under `trace`). Useful for debugging; noisy for agents.",
    )


@register_function_group(config_type=TavilyToolsGroupConfig)
async def tavily_tools(config: TavilyToolsGroupConfig, _builder: Builder) -> AsyncIterator[FunctionGroup]:
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

    async def _research(value: TavilyResearchInput) -> dict:
        # `client.research` is async-def; awaiting it returns an AsyncGenerator[bytes] when
        # stream=True (or a dict when stream=False, which we don't use here).
        stream = await client.research(
            stream=True,
            timeout=config.research_timeout_seconds,
            **value.model_dump(exclude_none=True),
        )
        return await _accumulate_research_stream(stream, include_trace=config.research_include_trace)

    group = FunctionGroup(config=config)
    group.add_function("search", _search, input_schema=TavilySearchInput, description=_DESCRIPTIONS["search"])
    group.add_function("extract", _extract, input_schema=TavilyExtractInput, description=_DESCRIPTIONS["extract"])
    group.add_function("crawl", _crawl, input_schema=TavilyCrawlInput, description=_DESCRIPTIONS["crawl"])
    group.add_function("map", _map, input_schema=TavilyMapInput, description=_DESCRIPTIONS["map"])
    group.add_function("research", _research, input_schema=TavilyResearchInput, description=_DESCRIPTIONS["research"])

    yield group
