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
"""Runtime tool wrapper — produces SDK Tool instances for the Builder framework."""

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import Function
from nat.cli.register_workflow import register_tool_wrapper
from nat.sdk.tool.tool import Tool

# Backward-compatible alias so existing imports continue to work.
RuntimeToolWrapper = Tool


@register_tool_wrapper(wrapper_type=LLMFrameworkEnum.RUNTIME)
def runtime_tool_wrapper(name: str, fn: Function, _builder: Builder) -> Tool:
    """Create a runtime tool wrapper for a toolkit function.

    Args:
        name: Tool name exposed to the OpenAI-compatible interface.
        fn: Toolkit `Function` to wrap.
        _builder: Builder instance (unused).

    Returns:
        Configured `Tool` instance.
    """
    return Tool.from_function(fn)
