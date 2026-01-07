<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# About Building NVIDIA NeMo Agent Toolkit Workflows

In NeMo Agent toolkit, a workflow defines which [functions](./functions-and-function-groups/functions.md) and [models](./llms/index.md) are used to perform a given task or series of tasks. A workflow definition is specified in a [YAML configuration file](#understanding-the-workflow-configuration-file). The `workflow` section of the configuration file defines the workflow itself, and specifies a function, typically an [agent](../components/agents/index.md), which will orchestrate which functions and models are called to complete the given task.

## Understanding the Workflow Configuration File

The workflow configuration file is a YAML file that specifies the [tools](./functions-and-function-groups/functions.md#agents-and-tools) and models to use in a workflow, along with general configuration settings. This section examines the configuration of the `examples/getting_started/simple_web_query` workflow to show how they are organized.

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

The workflow itself is typically an agent, however any NeMo Agent toolkit function can be used as a workflow. Refer to the [Agents](../components/agents/index.md) documentation for more details on the agents that are included in NeMo Agent toolkit.

In this workflow, the `webpage_query` tool queries the LangSmith User Guide, and the `current_datetime` tool gets the current date and time. The `description` entry instructs the LLM when and how to use the tool. In this case, the workflow explicitly defines `description` for the `webpage_query` tool.

The `webpage_query` tool uses the `nv-embedqa-e5-v5` embedder, which is defined in the `embedders` section.

For details on workflow configuration, including sections not utilized in the above example, refer to the [Workflow Configuration](./workflow-configuration.md) document.

## Using Agents With Workflows

The following are [agents](../components/agents/index.md) offered by NeMo Agent toolkit:

- [Automatic Memory Wrapper Agent](../components/agents/auto-memory-wrapper/index.md)
- [ReAct Agent](../components/agents/react-agent/index.md)
- [Reasoning Agent](../components/agents/reasoning-agent/index.md)
- [ReWOO Agent](../components/agents/rewoo-agent/index.md)
- [Responses API and Agent](../components/agents/responses-api-and-agent/index.md)
- [Tool Calling Agent](../components/agents/tool-calling-agent/index.md)

## Using Control Flow Components With Workflows

The following are control flow components offered by NeMo Agent toolkit:

- [Router Agent](../components/agents/router-agent/index.md)
- [Sequential Executor](../components/agents/sequential-executor/index.md)
