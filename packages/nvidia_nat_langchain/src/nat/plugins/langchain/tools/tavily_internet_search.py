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

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

_MIGRATION_MESSAGE = (
    "`tavily_internet_search` was removed from `nvidia-nat[langchain]` in NeMo Agent Toolkit 1.8. "
    "Install `nemo-agent-toolkit-tavily` and migrate the workflow config to a Tavily function group, for example:\n\n"
    "function_groups:\n"
    "  internet_search:\n"
    "    _type: tavily\n"
    "    include: [search]\n\n"
    "workflow:\n"
    "  tool_names: [internet_search__search]\n")


class TavilyInternetSearchToolConfig(FunctionBaseConfig, name="tavily_internet_search"):
    """Migration stub for the removed LangChain-backed Tavily search tool."""


@register_function(config_type=TavilyInternetSearchToolConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def tavily_internet_search(tool_config: TavilyInternetSearchToolConfig, builder: Builder):
    raise RuntimeError(_MIGRATION_MESSAGE)
    yield
