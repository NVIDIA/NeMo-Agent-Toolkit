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
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class HaystackWebSearchConfig(FunctionBaseConfig, name="haystack_web_search_tool"):
    top_k: int = 10
    timeout: int = 3
    retry_attempts: int = 2


@register_function(config_type=HaystackWebSearchConfig)
async def haystack_web_search_tool(tool_config: HaystackWebSearchConfig, builder: Builder):
    """
    Web search tool using Haystack's SerperDevWebSearch and LinkContentFetcher
    Returns a SuperComponent that can be used with ComponentTool
    """
    from haystack.components.converters.html import HTMLToDocument
    from haystack.components.converters.output_adapter import OutputAdapter
    from haystack.components.fetchers.link_content import LinkContentFetcher
    from haystack.components.websearch.serper_dev import SerperDevWebSearch
    from haystack.core.pipeline import Pipeline
    from haystack.core.super_component import SuperComponent

    logger.info(f"Creating web search tool with config: {tool_config}")

    # Create the search pipeline exactly as in the notebook
    search_pipeline = Pipeline()

    search_pipeline.add_component("search", SerperDevWebSearch(top_k=tool_config.top_k))
    search_pipeline.add_component(
        "fetcher",
        LinkContentFetcher(
            timeout=tool_config.timeout,
            raise_on_failure=False,
            retry_attempts=tool_config.retry_attempts
        )
    )
    search_pipeline.add_component("converter", HTMLToDocument())
    search_pipeline.add_component(
        "output_adapter",
        OutputAdapter(
            template="""
{%- for doc in docs -%}
  {%- if doc.content -%}
  <search-result url="{{ doc.meta.url }}">
  {{ doc.content|truncate(25000) }}
  </search-result>
  {%- endif -%}
{%- endfor -%}
""",
            output_type=str,
        ),
    )

    search_pipeline.connect("search.links", "fetcher.urls")
    search_pipeline.connect("fetcher.streams", "converter.sources")
    search_pipeline.connect("converter.documents", "output_adapter.docs")

    # Create SuperComponent exactly as in the notebook
    search_component = SuperComponent(
        pipeline=search_pipeline,
        input_mapping={"query": ["search.query"]},
        output_mapping={"output_adapter.output": "search_result"},
    )

    async def _arun(query: str) -> str:
        """
        Search the web for information on a given topic

        Args:
            query: The search query

        Returns:
            Formatted search results with URLs and content
        """
        try:
            logger.info(f"Performing web search for query: {query}")
            result = search_component.run({"query": query})
            output = result["search_result"]
            logger.info(f"Web search completed successfully for query: {query}")
            return output
        except Exception as e:
            logger.error(f"Web search failed for query '{query}': {str(e)}")
            return f"Search failed: {str(e)}"

    # Store the component for access by the workflow
    _arun.search_component = search_component

    yield FunctionInfo.from_fn(
        _arun,
        description="Use this tool to search for information on the internet."
    )