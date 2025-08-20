# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from haystack.components.converters.html import HTMLToDocument
from haystack.components.converters.output_adapter import OutputAdapter
from haystack.components.fetchers.link_content import LinkContentFetcher
from haystack.components.websearch.serper_dev import SerperDevWebSearch
from haystack.core.pipeline import Pipeline
from haystack.core.super_component import SuperComponent
from haystack.tools import ComponentTool


def create_search_tool() -> ComponentTool:
	search_pipeline = Pipeline()
	search_pipeline.add_component("search", SerperDevWebSearch(top_k=10))
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
		description="Use this tool to search for information on the internet.",
		component=search_component,
		outputs_to_string={"source": "search_result"},
	)
