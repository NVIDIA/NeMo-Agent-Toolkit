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

# Get Started with NVIDIA AgentIQ

This guide will help you set up your development environment, run existing workflows, and create your own custom workflows using the `aiq` command-line interface.

### Supported LLM APIs:
- NIM (such as Llama-3.1-70b-instruct and Llama-3.3-70b-instruct)
- OpenAI

### Supported LLM Frameworks:
- LangChain
- LlamaIndex
- CrewAI
- Semantic Kernel

## Installing AgentIQ
To run the examples, you need to install AgentIQ from source. For more information on installing AgentIQ from source, refer to [Install From Source](./install.md#install-from-source)

## Obtaining API Keys
Depending which workflows you are running, you may need to obtain API keys from the respective services. Most AgentIQ workflows require an NVIDIA API key defined with the `NVIDIA_API_KEY` environment variable. An API key can be obtained by visiting [`build.nvidia.com`](https://build.nvidia.com/) and creating an account.

## Running Example Workflows

Before running any of the AgentIQ examples, set your NVIDIA API key as an
environment variable to access NVIDIA AI services.

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

:::{note}
Replace `<YOUR_API_KEY>` with your actual NVIDIA API key.
:::

### Running the Simple Workflow

1. Install the `aiq_simple` Workflow

    ```bash
    uv pip install -e examples/simple
    ```

2. Run the `aiq_simple` Workflow

    ```bash
    aiq run --config_file=examples/simple/configs/config.yml --input "What is LangSmith"
    ```

3. **Run and evaluate the `aiq_simple` Workflow**

    The `eval_config.yml` YAML is a super-set of the `config.yml` containing additional fields for evaluation. To evaluate the `aiq_simple` workflow, run the following command:
    ```bash
    aiq eval --config_file=examples/simple/configs/eval_config.yml
    ```


## Next Steps

* AgentIQ contains several examples which demonstrate how AgentIQ can be used to build custom workflows and tools. These examples are located in the `examples` directory of the AgentIQ repository.
* Refer to the [AgentIQ Guides](../guides/index.md) for more detailed information on how to use AgentIQ.
