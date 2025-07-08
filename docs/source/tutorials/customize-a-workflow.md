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

# Customize a Workflow

## Prerequisites

1. Set up your environment by following the instructions in the [Install From Source](../quick-start/installing.md#install-from-source) section of the install guide.
1. Install AIQ toolkit and the AIQ toolkit Simple example workflow.
    ```bash
    uv pip install -e .
    uv pip install -e examples/basic/functions/simple
    ```

This tutorial assumes familiarity with [workflows](../workflows/about/index.md) and the [command line interface](../reference/cli.md) of AIQ toolkit.

## Customizing the `examples/basic/functions/simple` Workflow

The `examples/basic/functions/simple` workflow is defined by the `examples/basic/functions/simple/configs/config.yml` configuration file, which you can examine in the configuration file contents.

`examples/basic/functions/simple/configs/config.yml`:
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

The  workflow file contains two tools: one that queries the LangSmith User Guide, and another that returns the current date and time. It also contains two models: an embedding model and an LLM model. After running the workflow, you can ask it questions about LangSmith. This tutorial demonstrates how to customize this workflow.

Each workflow contains several configuration parameters that can be modified to customize the workflow. While copying and modifying the file is possible, it is not always necessary as some parameters can be overridden using the `--override` flag.

Examining the `examples/basic/functions/simple/configs/config.yml` file, the `llms` section is as follows:
```yaml
llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    temperature: 0.0
```

To override the `temperature` parameter for the `nim_llm`, the following command can be used:
```bash
aiq run --config_file examples/basic/functions/simple/configs/config.yml --input "What is LangSmith?"  \
  --override llms.nim_llm.temperature 0.7
```

When successful, the output contains the following line:
```
aiq.cli.cli_utils.config_override - INFO - Successfully set override for llms.nim_llm.temperature with value: 0.7
```

The `--override` flag can be specified multiple times, allowing the ability to override multiple parameters. For example, the `llama-3.1-70b-instruct` model can be replaced with the `llama-3.3-70b-instruct` using:
```bash
aiq run --config_file examples/basic/functions/simple/configs/config.yml --input "What is LangSmith?"  \
  --override llms.nim_llm.temperature 0.7 \
  --override llms.nim_llm.model_name meta/llama-3.3-70b-instruct
```

:::{note}
Not all parameters are specified in the workflow YAML. For each tool, there are potentially multiple optional parameters with default values that can be overridden. The `aiq info components` command can be used to list all available parameters. In this case, to list all available parameters for the LLM `nim` type run:
```bash
aiq info components -t llm_provider -q nim
```
:::
