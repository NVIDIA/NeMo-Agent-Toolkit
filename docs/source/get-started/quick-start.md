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

# Quick Start with NVIDIA NeMo Agent Toolkit

This guide will walk you through [running](../run-workflows/about-running-workflows.md) and [evaluating](../improve-workflows/evaluate.md) existing [workflows](../build-workflows/about-building-workflows.md). If you have not yet installed the NeMo Agent toolkit, follow the instructions in the [Install Guide](./installation.md) first.

## Obtaining API Keys

Depending on which workflows you are running, you may need to obtain API keys from the respective services. Most NeMo Agent toolkit workflows require an NVIDIA API key defined with the `NVIDIA_API_KEY` environment variable. An API key can be obtained by creating an account on [`build.nvidia.com`](https://build.nvidia.com/).

### Optional OpenAI API Key

Some workflows may also require an OpenAI API key. Create an account on [OpenAI](https://openai.com/). Navigate to your account settings to obtain your OpenAI API key. Copy the key and set it as an environment variable using the following command:

```bash
export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
```

## Running Example Workflows

Before running any of the NeMo Agent toolkit examples, set your NVIDIA API key as an
environment variable to access NVIDIA AI services.

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

:::{note}
Replace `<YOUR_API_KEY>` with your actual NVIDIA API key.
:::

### Running the Simple Workflow

1. Install the `nat_simple_web_query` Workflow

   ```bash
   uv pip install -e examples/getting_started/simple_web_query
   ```

2. Run the `nat_simple_web_query` Workflow

   ```bash
   nat run --config_file=examples/getting_started/simple_web_query/configs/config.yml --input "What is LangSmith"
   ```

3. **Run and evaluate the `nat_simple_web_query` Workflow**

   The `eval_config.yml` YAML is a super-set of the `config.yml` containing additional fields for [evaluation](../improve-workflows/evaluate.md). To evaluate the `nat_simple_web_query` workflow, run the following command:

   ```bash
   nat eval --config_file=examples/evaluation_and_profiling/simple_web_query_eval/configs/eval_config.yml
   ```

## NeMo Agent Toolkit Packages

Once a NeMo Agent toolkit workflow is ready for deployment to production, the deployed workflow will need to declare a dependency on the `nvidia-nat` package, along with the needed plugins. When declaring a dependency on NeMo Agent toolkit, we recommend using the first two digits of the version number. For example if the version is `1.0.0`, then the dependency would be `1.0`.

For more information on the available plugins, refer to [Framework Integrations](./installation.md#framework-integrations).

Example of a dependency for NeMo Agent toolkit using the LangChain/LangGraph plugin for projects using a `pyproject.toml` file:

```toml
dependencies = [
"nvidia-nat[langchain]~=1.0",
# Add any additional dependencies your workflow needs
]
```

For projects using a `requirements.txt` file:

```
nvidia-nat[langchain]==1.0.*
```

## Next Steps

- Review the NeMo Agent toolkit [tutorials](./tutorials/index.md) for detailed guidance on using the toolkit.
- Explore the examples in the `examples` directory to learn how to build custom workflows and [tools](../build-workflows/functions-and-function-groups/functions.md#agents-and-tools) with NeMo Agent toolkit.
