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

# Create and Customize NVIDIA Agent Intelligence Toolkit Workflows

Workflows are the heart of AIQ Toolkit because they define which agentic tools and models are used to perform a given task or series of tasks. This document will walk through the process of running an existing workflow, customizing an existing workflow, adding tools to a workflow, creating a new tool, and creating a new workflows.

## Prerequisites

1. Set up your environment by following the instructions in the [Install From Source](../quick-start/installing.md#install-from-source) section of the install guide.
1. Install AIQ Toolkit and the AIQ Toolkit Simple example workflow.
    ```bash
    uv pip install -e .
    uv pip install -e examples/simple
    ```

## Running a Workflow

A workflow is defined by a YAML configuration file that specifies the tools and models to use. AIQ Toolkit provides the following ways to run a workflow:
- Using the `aiq run` command.
   - This is the simplest and most common way to run a workflow.
- Using the `aiq serve` command.
   - This starts a web server that listens for incoming requests and runs the specified workflow.
- Using the `aiq eval` command.
   - In addition to running the workflow, it also evaluates the accuracy of the workflow.
- Using the Python API
   - This is the most flexible way to run a workflow.

![Running Workflows](../_static/running_workflows.png)

### Using the `aiq run` Command
The `aiq run` command is the simplest way to run a workflow. `aiq run` receives a configuration file as specified by the `--config_file` flag, along with input that can be specified either directly with the `--input` flag or by providing a file path with the `--input_file` flag.

A typical invocation of the `aiq run` command follows this pattern:
```
aiq run --config_file <path/to/config.yml> [--input "question?" | --input_file <path/to/input.txt>]
```

The following command runs the `examples/simple` workflow with a single input question "What is LangSmith?":
```bash
aiq run --config_file examples/simple/configs/config.yml --input "What is LangSmith?"
```

The following command runs the same workflow with the input question provided in a file:
```bash
echo "What is LangSmith?" > .tmp/input.txt
aiq run --config_file examples/simple/configs/config.yml --input_file .tmp/input.txt
```

### Using the `aiq eval` Command
The `aiq eval` command is similar to the `aiq run` command, however in addition to running the workflow it also evaluates the accuracy of the workflow, refer to [Evaluating AIQ Toolkit Workflows](../guides/evaluate.md) for more information.

### Using the `aiq serve` Command
The `aiq serve` command starts a web server that listens for incoming requests and runs the specified workflow. The server can be accessed with a web browser or by sending a POST request to the server's endpoint. Similar to the `aiq run` command, the `aiq serve` command requires a configuration file specified by the `--config_file` flag.

The following command runs the `examples/simple` workflow on a web server listening to the default port `8000` and default endpoint of `/generate`:
```bash
aiq serve --config_file examples/simple/configs/config.yml
```

In a separate terminal, run the following command to send a POST request to the server:
```bash
curl --request POST \
  --url http://localhost:8000/generate \
  --header 'Content-Type: application/json' \
  --data '{
    "input_message": "What is LangSmith?"
}'
```

Refer to `aiq serve --help` for more information on how to customize the server.

### Using the Python API

Using the Python API for running workflows is outside the scope of this document. Refer to the Python API documentation for the {py:class}`~aiq.runtime.runner.AIQRunner` class for more information.

## Understanding the Workflow Configuration File

The workflow configuration file is a YAML file that specifies the tools and models to use in a workflow, along with general configuration settings. To illustrate how these are organized, we will examine the configuration of the simple workflow that we used in the previous section.

`examples/simple/configs/config.yml`:
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
  retry_parsing_errors: true
  max_retries: 3
```

In the previous example, note that it is divided into four sections: `functions`, `llms`, `embedders`, and `workflow`. The `functions` section contains the tools used in the workflow, while `llms` and `embedders` define the models used in the workflow, and lastly the `workflow` section ties defines the workflow itself.

In the example workflow the `webpage_query` tool is used to query the LangSmith User Guide, and the `current_datetime` tool is used to get the current date and time. The questions we have asked the workflow have not involved time and the workflow would still run without the `current_datetime` tool.

The `description` entry is what is used to instruct the LLM when and how to use the tool. In this case, we explicitly defined the `description` for the `webpage_query` tool.

The `webpage_query` tool makes use of the `nv-embedqa-e5-v5` embedder, which is defined in the `embedders` section.

For details on workflow configuration, including sections not utilized in the above example, refer to the [Workflow Configuration](../concepts/workflow-configuration.md) document.
