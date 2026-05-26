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

import logging

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import Function
from nat.cli.register_workflow import register_tool_wrapper

logger = logging.getLogger(__name__)


@register_tool_wrapper(wrapper_type=LLMFrameworkEnum.LANGCHAIN)
def langchain_tool_wrapper(name: str, fn: Function, builder: Builder):

    import asyncio
    import json
    import typing

    from langchain_core.tools.structured import StructuredTool

    from nat.data_models.api_server import ChatRequest

    assert fn.input_schema is not None, "Tool must have input schema"

    def _normalize_messages(messages: list) -> list:
        normalized_messages = []
        for message in messages:
            if isinstance(message, dict) and isinstance(message.get("content"), dict):
                message = {**message, "content": [message["content"]]}
            normalized_messages.append(message)
        return normalized_messages

    class NATStructuredTool(StructuredTool):

        def _parse_input(self, tool_input: str | dict, tool_call_id: str | None) -> str | dict:
            if isinstance(tool_input, str):
                schema = self.args_schema
                if schema is not None and "input_message" in getattr(schema, "model_fields", {}):
                    tool_input = {"input_message": tool_input}
                elif isinstance(schema, type) and issubclass(schema, ChatRequest):
                    tool_input = ChatRequest.from_string(tool_input).model_dump(exclude_none=True)
            elif isinstance(tool_input, dict):
                schema = self.args_schema
                if schema is not None and "messages" in getattr(schema, "model_fields", {}):
                    messages = tool_input.get("messages")
                    if isinstance(messages, str):
                        try:
                            parsed_messages = json.loads(messages)
                        except json.JSONDecodeError:
                            pass
                        else:
                            if isinstance(parsed_messages, list):
                                tool_input = {**tool_input, "messages": _normalize_messages(parsed_messages)}
                    elif isinstance(messages, list):
                        tool_input = {**tool_input, "messages": _normalize_messages(messages)}

            return typing.cast(str | dict, super()._parse_input(tool_input, tool_call_id))

    loop = asyncio.get_running_loop()

    # Provide a sync wrapper for the tool to support synchronous tool calls
    def _sync_fn(*args, **kwargs):
        logger.warning("Invoking a synchronous tool call, performance may be degraded: `%s`", fn.instance_name)
        return loop.run_until_complete(fn.acall_invoke(*args, **kwargs))

    if fn.description is None:
        logger.warning("No description set for `%s` falling back to instance name: `%s`",
                       type(fn).__name__,
                       fn.instance_name)
        _sync_fn.__doc__ = fn.instance_name

    return NATStructuredTool.from_function(coroutine=fn.acall_invoke,
                                           func=_sync_fn,
                                           name=name,
                                           description=fn.description,
                                           args_schema=fn.input_schema)
