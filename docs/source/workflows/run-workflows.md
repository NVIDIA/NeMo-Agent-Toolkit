<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Run NVIDIA NeMo Agent Toolkit Workflows

A workflow is defined by a YAML configuration file that specifies the tools and models to use. NeMo Agent toolkit provides the following ways to run a workflow:
- Using the `aiq run` command.
   - This is the simplest and most common way to run a workflow.
- Using the `aiq serve` command.
   - This starts a web server that listens for incoming requests and runs the specified workflow.
- Using the `aiq eval` command.
   - In addition to running the workflow, it also evaluates the accuracy of the workflow.
- Using the Python API
   - This is the most flexible way to run a workflow.

![Running Workflows](../_static/running_workflows.png)

## Prerequisites

Ensure that you have followed the instructions in the [Install Guide](../quick-start/installing.md#install-from-source) to create the development environment and install NeMo Agent toolkit.

The examples in this document utilize the `examples/getting_started/simple_web_query` workflow, install it by running the following commands from the root directory of the NeMo Agent toolkit library:
```bash
uv pip install -e examples/getting_started/simple_web_query
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```


## Using the `aiq run` Command
The `aiq run` command is the simplest way to run a workflow. `aiq run` receives a configuration file as specified by the `--config_file` flag, along with input that can be specified either directly with the `--input` flag or by providing a file path with the `--input_file` flag.

A typical invocation of the `aiq run` command follows this pattern:
```
aiq run --config_file <path/to/config.yml> [--input "question?" | --input_file <path/to/input.txt>]
```

The following command runs the `examples/getting_started/simple_web_query` workflow with a single input question "What is LangSmith?":
```bash
aiq run --config_file examples/getting_started/simple_web_query/configs/config.yml --input "What is LangSmith?"
```

The following command runs the same workflow with the input question provided in a file:
```bash
echo "What is LangSmith?" > .tmp/input.txt
aiq run --config_file examples/getting_started/simple_web_query/configs/config.yml --input_file .tmp/input.txt
```

## Using the `aiq eval` Command
The `aiq eval` command is similar to the `aiq run` command. However, in addition to running the workflow, it also evaluates the accuracy of the workflow, refer to [Evaluating NeMo Agent toolkit Workflows](../workflows/evaluate.md) for more information.

## Using the `aiq serve` Command
The `aiq serve` command starts a web server that listens for incoming requests and runs the specified workflow. The server can be accessed with a web browser or by sending a POST request to the server's endpoint. Similar to the `aiq run` command, the `aiq serve` command requires a configuration file specified by the `--config_file` flag.

The following command runs the `examples/getting_started/simple_web_query` workflow on a web server listening to the default port `8000` and default endpoint of `/generate`:
```bash
aiq serve --config_file examples/getting_started/simple_web_query/configs/config.yml
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

## Using the Python API

The toolkit offers a programmatic way to execute workflows through its Python API, allowing you to integrate workflow execution directly into your Python code. Here's how to use it:

```python
import asyncio

from aiq.runtime.loader import load_workflow
from aiq.utils.type_utils import StrPath


async def run_workflow(config_file: StrPath, input_str: str) -> str:
    async with load_workflow(config_file) as workflow:
        async with workflow.run(input_str) as runner:
            return await runner.result(to_type=str)


result = asyncio.run(
    run_workflow(config_file='examples/getting_started/simple_web_query/configs/config.yml',
                 input_str='What is LangSmith?'))

print(result)
```

In this example:
- `config_file`: A string or {py:class}`~pathlib.Path` pointing to your workflow YAML file
- `input_str`: A string containing the input for your workflow
- The `workflow.run(input_str)` method returns an instance of {py:class}`~aiq.runtime.runner.AIQRunner`

For detailed information about the `AIQRunner` class and its capabilities, please refer to the Python API documentation for the {py:class}`~aiq.runtime.runner.AIQRunner` class.
