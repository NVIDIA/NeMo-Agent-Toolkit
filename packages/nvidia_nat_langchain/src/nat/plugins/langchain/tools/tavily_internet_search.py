# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


# Internet Search tool
class TavilyInternetSearchToolConfig(FunctionBaseConfig, name="tavily_internet_search"):
    """
    Tool that retrieves relevant contexts from web search (using Tavily) for the given question.
    Requires a TAVILY_API_KEY.
    """
    max_results: int = 3
    api_key: str = ""


@register_function(config_type=TavilyInternetSearchToolConfig)
async def tavily_internet_search(tool_config: TavilyInternetSearchToolConfig, builder: Builder):
    import os

    from langchain_community.tools import TavilySearchResults

    if not os.environ.get("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = tool_config.api_key
    # This tavily tool requires an API Key and it must be set as an environment variable (TAVILY_API_KEY)
    # Refer to create_customize_workflow.md for instructions of getting the API key

    async def _tavily_internet_search(question: str) -> str:
        # Search the web and get the requested amount of results
        tavily_search = TavilySearchResults(max_results=tool_config.max_results)
        search_docs = await tavily_search.ainvoke({'query': question})
        # Format
        web_search_results = "\n\n---\n\n".join(
            [f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>' for doc in search_docs])
        return web_search_results

    # Create a Generic NAT tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _tavily_internet_search,
        description=("""This tool retrieves relevant contexts from web search (using Tavily) for the given question.

                        Args:
                            question (str): The question to be answered.
                    """),
    )
