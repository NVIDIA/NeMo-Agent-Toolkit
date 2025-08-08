<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Add Tools to a Workflow

The [Customizing a Workflow](./customize-a-workflow.md) tutorial demonstrates how to customize a workflow by overriding parameters. This tutorial will demonstrate how to add new tools to a workflow. Adding a new tool to a workflow requires copying and modifying the workflow configuration file, which, in effect, creates a new customized workflow.

NeMo Agent toolkit includes several built-in tools (functions) that can be used in any workflow. To query for a list of installed tools, run the following command:
```bash
aiq info components -t function
```

The `examples/getting_started/simple_web_query/configs/config.yml` workflow defines a tool to query the [LangSmith User Guide](https://docs.smith.langchain.com). This is defined in the `functions` section of the configuration file:
```yaml
functions:
  webpage_query:
    _type: webpage_query
    webpage_url: https://docs.smith.langchain.com
    description: "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!"
    embedder_name: nv-embedqa-e5-v5
    chunk_size: 512
```

However, the workflow is unaware of some related technologies, such as LangChain, if you run:
```bash
aiq run --config_file examples/getting_started/simple_web_query/configs/config.yml --input "How do I trace only specific parts of my LangChain application?"
```

The output may be similar to the following:
```
Workflow Result:
["Unfortunately, the provided webpages do not provide specific instructions on how to trace only specific parts of a LangChain application using LangSmith. However, they do provide information on how to set up LangSmith tracing with LangChain and how to use LangSmith's observability features to analyze traces and configure metrics, dashboards, and alerts. It is recommended to refer to the how-to guide for setting up LangSmith with LangChain or LangGraph for more information."]
```

You can solve this by updating the workflow to also query the [LangGraph Quickstart](https://langchain-ai.github.io/langgraph/tutorials/introduction) guide.

To do this, create a copy of the original workflow configuration file. To add the LangGraph query tool to the workflow, update the YAML file updating the `functions` section from:
```yaml
functions:
  webpage_query:
    _type: webpage_query
    webpage_url: https://docs.smith.langchain.com
    description: "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!"
    embedder_name: nv-embedqa-e5-v5
    chunk_size: 512
```

to:
```yaml
functions:
  langsmith_query:
    _type: webpage_query
    webpage_url: https://docs.smith.langchain.com
    description: "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!"
    embedder_name: nv-embedqa-e5-v5
    chunk_size: 512
  langchain_query:
    _type: webpage_query
    webpage_url: https://docs.smith.langchain.com/observability/how_to_guides/trace_with_langchain
    description: "Search for information about LangChain. For any questions about LangChain, you must use this tool!"
    embedder_name: nv-embedqa-e5-v5
    chunk_size: 512
```

Since you now have two instances of the `webpage_query` tool, you need to update the name of the first tool to `langsmith_query`.

Finally, update the `workflow.tool_names` section to include the new tool from:
```yaml
workflow:
  _type: react_agent
  tool_names: [webpage_query, current_datetime]
```

to:
```yaml
workflow:
  _type: react_agent
  tool_names: [langsmith_query, langchain_query, current_datetime]
```

:::{note}
The resulting YAML is located at `examples/documentation_guides/workflows/custom_workflow/custom_config.yml` in the NeMo Agent toolkit repository.
:::

When you rerun the workflow with the updated configuration file:
```bash
aiq run --config_file examples/documentation_guides/workflows/custom_workflow/custom_config.yml \
  --input "How do I trace only specific parts of my LangChain application?"
```

We should receive output similar to:
```
Workflow Result:
['To trace only specific parts of a LangChain application, you can either manually pass in a LangChainTracer instance as a callback or use the tracing_v2_enabled context manager. Additionally, you can configure a LangChainTracer instance to trace a specific invocation.']
```

## Alternate Method Using a Web Search Tool
Adding individual web pages to a workflow can be cumbersome, especially when dealing with multiple web pages. An alternative method is to use a web search tool. One of the tools available in NeMo Agent toolkit is the `tavily_internet_search` tool, which utilizes the [Tavily Search API](https://tavily.com/).

The `tavily_internet_search` tool is part of the `aiqtoolkit[langchain]` package, to install the package run:
```bash
# local package install from source
uv pip install -e '.[langchain]'
```

Prior to using the `tavily_internet_search` tool, create an account at [`tavily.com`](https://tavily.com/) and obtain an API key. Once obtained, set the `TAVILY_API_KEY` environment variable to the API key:
```bash
export TAVILY_API_KEY=<YOUR_TAVILY_API_KEY>
```

We will now update the `functions` section of the configuration file replacing the two `webpage_query` tools with a single `tavily_internet_search` tool entry:
```yaml
functions:
  internet_search:
    _type: tavily_internet_search
  current_datetime:
    _type: current_datetime
```

Next, update the `workflow.tool_names` section to include the new tool:
```yaml
workflow:
  _type: react_agent
  tool_names: [internet_search, current_datetime]
```

The resulting configuration file is located at `examples/documentation_guides/workflows/custom_workflow/search_config.yml` in the NeMo Agent toolkit repository.

When you re-run the workflow with the updated configuration file:
```bash
aiq run --config_file examples/documentation_guides/workflows/custom_workflow/search_config.yml \
  --input "How do I trace only specific parts of my LangChain application?"
```

Which will then yield a slightly different result to the same question:
```
Workflow Result:
['To trace only specific parts of a LangChain application, users can use the `@traceable` decorator to mark specific functions or methods as traceable. Additionally, users can configure the tracing functionality to log traces to a specific project, add metadata and tags to traces, and customize the run name and ID. Users can also use the `LangChainTracer` class to trace specific invocations or parts of their application. Furthermore, users can use the `tracing_v2_enabled` context manager to trace a specific block of code.']
```
