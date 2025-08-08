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

import asyncio

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import Function
from nat.cli.register_workflow import register_tool_wrapper


@register_tool_wrapper(wrapper_type=LLMFrameworkEnum.CREWAI)
def crewai_tool_wrapper(name: str, fn: Function, builder: Builder):

    from crewai.tools.base_tool import Tool

    # Capture the loop at the time this is called
    loop = asyncio.get_event_loop()

    # Capture the coroutine at the time this is called
    runnable = fn.acall_invoke

    # Because CrewAI tools are not async, we need to wrap the coroutine in a normal function
    def wrapper(*args, **kwargs):

        return asyncio.run_coroutine_threadsafe(runnable(*args, **kwargs), loop).result()

    return Tool(name=name, description=fn.description or "", args_schema=fn.input_schema, func=wrapper)
