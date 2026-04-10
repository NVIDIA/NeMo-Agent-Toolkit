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
"""Converts NAT Function objects to AG2 Tool objects."""

import logging
from dataclasses import is_dataclass
from typing import Any

from autogen.tools import Tool
from pydantic import TypeAdapter

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import Function
from nat.cli.register_workflow import register_tool_wrapper

logger = logging.getLogger(__name__)


def _get_json_schema(input_schema: Any) -> dict[str, Any] | None:
    """Extract JSON schema from a Pydantic model, pydantic dataclass, or stdlib dataclass."""
    if input_schema is None:
        return None
    if hasattr(input_schema, "model_json_schema"):
        return input_schema.model_json_schema()
    if is_dataclass(input_schema):
        return TypeAdapter(input_schema).json_schema()
    return None


@register_tool_wrapper(wrapper_type=LLMFrameworkEnum.AG2)
def ag2_tool_wrapper(
    name: str,
    fn: Function,
    _builder: Builder,
) -> Tool:
    """Convert a NAT Function to an AG2 Tool."""

    async def callable_ainvoke(**kwargs: Any) -> Any:
        """Async function to invoke the NAT function.

        Args:
            **kwargs: Keyword arguments to pass to the NAT function.

        Returns:
            Any: The result of invoking the NAT function.
        """
        return await fn.acall_invoke(**kwargs)

    async def callable_astream_collected(**kwargs: Any) -> Any:
        """Async function to collect all streamed results from the NAT function.

        AG2 tools return a single value, so streaming results are collected
        and joined into a single response.

        Args:
            **kwargs: Keyword arguments to pass to the NAT function.

        Returns:
            Any: The collected streaming results.
        """
        chunks: list[Any] = []
        async for item in fn.acall_stream(**kwargs):
            chunks.append(item)
        if not chunks:
            return ""
        if all(isinstance(c, str) for c in chunks):
            return "".join(chunks)
        return chunks

    if fn.has_streaming_output and not fn.has_single_output:
        logger.debug("Creating stream-collected AG2 Tool for: %s", name)
        call_function = callable_astream_collected
    else:
        logger.debug("Creating async AG2 Tool for: %s", name)
        call_function = callable_ainvoke

    call_function.__name__ = name
    call_function.__doc__ = fn.description or name

    return Tool(
        name=name,
        description=fn.description or name,
        func_or_tool=call_function,
        parameters_json_schema=_get_json_schema(fn.input_schema),
    )
