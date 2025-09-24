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

from haystack.components.converters.html import HTMLToDocument
from haystack.components.converters.output_adapter import OutputAdapter
from haystack.components.fetchers.link_content import LinkContentFetcher
from haystack.components.websearch.serper_dev import SerperDevWebSearch
from haystack.core.pipeline import Pipeline
from haystack.core.super_component import SuperComponent
from haystack.tools import ComponentTool


def create_search_tool(top_k: int = 10) -> ComponentTool:
    """
    Build a Haystack web search tool pipeline.

    Args:
        top_k: Number of search results to retrieve from Serper.

    Returns:
        ComponentTool: A Haystack tool that executes web search and returns formatted text.
    """
    search_pipeline = Pipeline()
    search_pipeline.add_component("search", SerperDevWebSearch(top_k=top_k))
    search_pipeline.add_component(
        "fetcher",
        LinkContentFetcher(timeout=3, raise_on_failure=False, retry_attempts=2),
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

    search_component = SuperComponent(
        pipeline=search_pipeline,
        input_mapping={"query": ["search.query"]},
        output_mapping={"output_adapter.output": "search_result"},
    )

    return ComponentTool(
        name="search",
        description="Use this tool to search for information on the Internet.",
        component=search_component,
        outputs_to_string={"source": "search_result"},
    )
