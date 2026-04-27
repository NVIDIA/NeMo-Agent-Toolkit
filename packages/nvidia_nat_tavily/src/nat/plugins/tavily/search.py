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
from typing import Union

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import create_model
from tavily import AsyncTavilyClient

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.common import SerializableSecretStr
from nat.data_models.function import FunctionBaseConfig

from ._client import build_async_client

logger = logging.getLogger(__name__)

# Parameters of AsyncTavilyClient.search that should NOT be exposed to the LLM.
# `self` is the bound instance, `**kwargs` is for forward-compat extras, and `timeout`
# is an HTTP-transport concern unrelated to the agent's intent.
_HIDDEN_PARAMS: frozenset[str] = frozenset({"self", "kwargs", "timeout"})

_DEFAULT_DESCRIPTION = (
    "Search the web with Tavily for up-to-date information. Returns a structured "
    "result set including a synthesized answer (when requested) and a list of "
    "ranked source documents with URLs, titles, and content snippets.")


def _build_input_schema() -> type[BaseModel]:
    """Build a pydantic input schema from the live AsyncTavilyClient.search signature.

    Annotations and defaults are pulled directly from the SDK so the agent surface
    tracks the SDK without manual maintenance.
    """
    sig = inspect.signature(AsyncTavilyClient.search)
    fields: dict[str, tuple[Any, Any]] = {}
    for name, param in sig.parameters.items():
        if name in _HIDDEN_PARAMS:
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        annotation = param.annotation if param.annotation is not inspect.Parameter.empty else Any

        if param.default is inspect.Parameter.empty:
            default = ...  # required
        else:
            default = param.default
            # SDK uses `T = None` as the "use server default" convention. Make the type
            # Optional so pydantic accepts None without erroring on the literal/scalar.
            if default is None:
                annotation = Union[annotation, None]

        fields[name] = (annotation, Field(default=default))

    return create_model(
        "TavilySearchInput",
        __config__=ConfigDict(arbitrary_types_allowed=True),
        **fields,
    )


# Build once at import. The schema is a static reflection of the SDK signature; rebuilding per
# registration would re-introspect a possibly mutated (e.g. monkeypatched in tests) bound method.
TavilySearchInput: type[BaseModel] = _build_input_schema()


class TavilySearchConfig(FunctionBaseConfig, name="tavily_search"):
    """Tavily web search tool. Thin wrapper over `tavily.AsyncTavilyClient.search`.

    Any field accepted by the SDK can be set per-call by the agent and will override
    the corresponding default set on this config.
    """

    model_config = ConfigDict(extra="allow")

    api_key: SerializableSecretStr = Field(
        default_factory=lambda: SerializableSecretStr(""),
        description="Tavily API key. Falls back to the TAVILY_API_KEY env var.",
    )
    description: str | None = Field(
        default=None,
        description="Optional override of the agent-facing tool description.",
    )


@register_function(config_type=TavilySearchConfig)
async def tavily_search(config: TavilySearchConfig, _builder: Builder):
    """Register the `tavily_search` NAT function.

    The input schema is generated dynamically from the SDK signature; per-call kwargs
    from the agent override defaults pulled from the config.
    """
    client = build_async_client(config.api_key)

    # Defaults set on the config (anything beyond the explicit fields, due to extra="allow")
    # become per-call defaults that the agent can override.
    config_defaults: dict[str, Any] = {
        k: v
        for k, v in config.model_dump(exclude_none=True).items()
        if k not in {"type", "api_key", "description"} and k in TavilySearchInput.model_fields
    }

    async def _search(value: TavilySearchInput) -> dict:
        agent_kwargs = value.model_dump(exclude_none=True)
        merged = {**config_defaults, **agent_kwargs}
        return await client.search(**merged)

    yield FunctionInfo.from_fn(
        fn=_search,
        input_schema=TavilySearchInput,
        description=config.description or _DEFAULT_DESCRIPTION,
    )
