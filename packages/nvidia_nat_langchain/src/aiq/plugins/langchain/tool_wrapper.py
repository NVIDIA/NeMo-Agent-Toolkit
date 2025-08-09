# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function import Function
from aiq.cli.register_workflow import register_tool_wrapper

logger = logging.getLogger(__name__)


@register_tool_wrapper(wrapper_type=LLMFrameworkEnum.LANGCHAIN)
def langchain_tool_wrapper(name: str, fn: Function, builder: Builder):  # pylint: disable=unused-argument

    import asyncio

    from langchain_core.tools.structured import StructuredTool

    assert fn.input_schema is not None, "Tool must have input schema"

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

    return StructuredTool.from_function(coroutine=fn.acall_invoke,
                                        func=_sync_fn,
                                        name=name,
                                        description=fn.description,
                                        args_schema=fn.input_schema)
