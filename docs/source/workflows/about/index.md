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

# About NVIDIA NeMo Agent Toolkit Workflows

Workflows are the heart of NeMo Agent toolkit because they define which agentic tools and models are used to perform a given task or series of tasks.

## Understanding the Workflow Configuration File

The workflow configuration file is a YAML file that specifies the tools and models to use in a workflow, along with general configuration settings. This section examines the configuration of the `examples/getting_started/simple_web_query` workflow to show how they're organized.

`examples/getting_started/simple_web_query/configs/config.yml`:
```yaml
functions:
  webpage_query:
    _type: webpage_query
    webpage_url: https://docs.smith.langchain.com
    description: "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!"
    embedder_name: nv-embedqa-e5-v5
    chunk_size: 512
  current_datetime:
    _type: current_datetime

llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    temperature: 0.0

embedders:
  nv-embedqa-e5-v5:
    _type: nim
    model_name: nvidia/nv-embedqa-e5-v5

workflow:
  _type: react_agent
  tool_names: [webpage_query, current_datetime]
  llm_name: nim_llm
  verbose: true
  parse_agent_response_max_retries: 3
```

This workflow configuration is divided into four sections: `functions`, `llms`, `embedders`, and `workflow`. The `functions` section contains the tools used in the workflow, while `llms` and `embedders` define the models used in the workflow, and lastly the `workflow` section ties the other sections together and defines the workflow itself.

In this workflow the `webpage_query` tool is used to query the LangSmith User Guide, and the `current_datetime` tool is used to get the current date and time. The `description` entry is what is used to instruct the LLM when and how to use the tool. In this case, the `description` is explicitly defined for the `webpage_query` tool.

The `webpage_query` tool makes use of the `nv-embedqa-e5-v5` embedder, which is defined in the `embedders` section.

For details on workflow configuration, including sections not utilized in the above example, refer to the [Workflow Configuration](../workflow-configuration.md) document.

## Workflow Types

```{toctree}

ReAct Agent <./react-agent.md>
Reasoning Agent <./reasoning-agent.md>
./rewoo-agent.md
Tool Calling Agent <./tool-calling-agent.md>
```
