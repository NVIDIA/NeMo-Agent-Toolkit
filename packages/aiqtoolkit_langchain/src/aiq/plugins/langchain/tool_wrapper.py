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

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function import Function
from aiq.cli.register_workflow import register_tool_wrapper


@register_tool_wrapper(wrapper_type=LLMFrameworkEnum.LANGCHAIN)
def langchain_tool_wrapper(name: str, fn: Function, builder: Builder):

    from langchain_core.tools.structured import StructuredTool

    assert fn.input_schema is not None, "Tool must have input schema"

    return StructuredTool.from_function(coroutine=fn.acall_invoke,
                                        name=name,
                                        description=fn.description,
                                        args_schema=fn.input_schema)
