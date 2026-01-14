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

# Install NVIDIA NeMo Agent Toolkit

This guide will help you set up your NVIDIA NeMo Agent toolkit development environment.

## Supported LLM APIs

The following [LLM](../build-workflows/llms/index.md) API providers are supported:

- NIM (such as Llama-3.1-70b-instruct and Llama-3.3-70b-instruct)
- OpenAI
- AWS Bedrock
- Azure OpenAI

## Framework Integrations

To keep the library lightweight, many of the first-party plugins supported by NeMo Agent toolkit are located in separate distribution packages. For example, the `nvidia-nat-langchain` distribution contains all the LangChain-specific and LangGraph-specific plugins, and the `nvidia-nat-mem0ai` distribution contains the Mem0-specific plugins.

To install these first-party plugin libraries, you can use the full distribution name (for example, `nvidia-nat-langchain`) or use the `nvidia-nat[langchain]` extra distribution. The following extras are supported:

- `nvidia-nat[adk]` or `nvidia-nat-adk` - [Google ADK](https://github.com/google/adk-python)
- `nvidia-nat[agno]` or `nvidia-nat-agno` - [Agno](https://agno.com/)
- `nvidia-nat[all]` or `nvidia-nat-all` - Pseudo-package for installing **almost all** optional dependencies
- `nvidia-nat[crewai]` or `nvidia-nat-crewai` - [CrewAI](https://www.crewai.com/).  Conflicts with `nvidia-nat[openpipe-art]`.
- `nvidia-nat[data-flywheel]` or `nvidia-nat-data-flywheel` - [NeMo DataFlywheel](https://github.com/NVIDIA-AI-Blueprints/data-flywheel)
- `nvidia-nat[huggingface]` - [HuggingFace](https://huggingface.co/) local model integration
- `nvidia-nat[ingestion]` or `nvidia-nat-ingestion` - Additional dependencies needed for data ingestion
- `nvidia-nat[langchain]` or `nvidia-nat-langchain` - [LangChain](https://www.langchain.com/), [LangGraph](https://www.langchain.com/langgraph)
- `nvidia-nat[llama-index]` or `nvidia-nat-llama-index` - [LlamaIndex](https://www.llamaindex.ai/)
- `nvidia-nat[mcp]` or `nvidia-nat-mcp` - [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
- `nvidia-nat[mem0ai]` or `nvidia-nat-mem0ai` - [Mem0](https://mem0.ai/)
- `nvidia-nat[mysql]` or `nvidia-nat-mysql` - [MySQL](https://www.mysql.com/)
- `nvidia-nat[openpipe-art]` or `nvidia-nat-openpipe-art` - [Agent Reinforcement Trainer](https://art.openpipe.ai/getting-started/about) **Not part of `nvidia-nat[all]` or `nvidia-nat-all`**. Conflicts with `nvidia-nat[crewai]`.
- `nvidia-nat[opentelemetry]` or `nvidia-nat-opentelemetry` - [OpenTelemetry](https://opentelemetry.io/)
- `nvidia-nat[phoenix]` or `nvidia-nat-phoenix` - [Arize Phoenix](https://arize.com/docs/phoenix)
- `nvidia-nat[profiling]` or `nvidia-nat-profiling` - Additional dependencies needed for [profiling](../improve-workflows/profiler.md)
- `nvidia-nat[ragaai]` or `nvidia-nat-ragaai` - [RagaAI Catalyst](https://raga.ai/) **Not part of `nvidia-nat[all]` or `nvidia-nat-all`**. Conflicts with `nvidia-nat[strands]`.
- `nvidia-nat[redis]` or `nvidia-nat-redis` - [Redis](https://redis.io/)
- `nvidia-nat[s3]` or `nvidia-nat-s3` - [Amazon S3](https://aws.amazon.com/s3/)
- `nvidia-nat[semantic-kernel]` or `nvidia-nat-semantic-kernel` - [Microsoft Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/)
- `nvidia-nat[strands]` or `nvidia-nat-strands` - [Strands Agents](https://github.com/strands-agents/sdk-python). Conflicts with `nvidia-nat[ragaai]`.
- `nvidia-nat[test]` or `nvidia-nat-test` - NeMo Agent toolkit test
- `nvidia-nat[vanna]` or `nvidia-nat-vanna` - [Vanna](https://vanna.ai/) text-to-SQL with Databricks support
- `nvidia-nat[weave]` or `nvidia-nat-weave` - [Weights & Biases Weave](https://weave-docs.wandb.ai)
- `nvidia-nat[zep-cloud]` or `nvidia-nat-zep-cloud` - [Zep](https://www.getzep.com/)


## Supported Platforms

| Operating System | Architecture | Python Version | Supported |
|------------------|--------------|---------------|-----------|
| Linux | x86_64 | 3.11, 3.12, 3.13 | ✅ Tested, Validated in CI |
| Linux | aarch64 | 3.11, 3.12, 3.13 | ✅ Tested, Validated in CI |
| macOS | x86_64 | 3.11, 3.12, 3.13 | ❓ Untested, Should Work |
| macOS | aarch64 | 3.11, 3.12, 3.13 | ✅ Tested |
| Windows | x86_64 | 3.11, 3.12, 3.13 | ❓ Untested, Should Work |
| Windows | aarch64 | 3.11, 3.12, 3.13 | ❌ Unsupported |

## Software Prerequisites

NVIDIA NeMo Agent toolkit is a Python library that doesn't require a GPU to run by default. Before you begin using NeMo Agent toolkit, ensure that you meet the following software prerequisites:

- [Python](https://www.python.org/) 3.11, 3.12, or 3.13

### Additional Prerequisites for Development
- [Git](https://git-scm.com/)
- [Git Large File Storage](https://git-lfs.github.com/) (LFS)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (version 0.5.4 or later, latest version is recommended)

## Install from Package

The package installation is recommended for production use.

:::{note}
To run any examples, you need to install the NeMo Agent toolkit from source.
:::

To install the latest stable version of NeMo Agent toolkit, run the following command:

```bash
pip install nvidia-nat
```

NeMo Agent toolkit has many optional dependencies which can be installed with the core package. Optional dependencies are grouped by framework and can be installed with the core package. For example, to install the LangChain/LangGraph plugin, run the following:

```bash
pip install "nvidia-nat[langchain]"
```

Or for all optional dependencies:

```bash
pip install "nvidia-nat[all]"
```

The full list of optional dependencies can be found [here](#framework-integrations).

## Install From Source

Installing from source is required to run any examples provided in the repository or to contribute to the project.

1. Clone the NeMo Agent toolkit repository to your local machine.
    ```bash
    git clone -b main https://github.com/NVIDIA/NeMo-Agent-Toolkit.git nemo-agent-toolkit
    cd nemo-agent-toolkit
    ```

1. Initialize, fetch, and update submodules in the Git repository.
    ```bash
    git submodule update --init --recursive
    ```

1. Fetch the data sets by downloading the LFS files.
    ```bash
    git lfs install
    git lfs fetch
    git lfs pull
    ```

1. Create a Python environment.
    ```bash
    uv venv --python 3.13 --seed .venv
    source .venv/bin/activate
    ```
    :::{note}
    Python 3.11 and 3.12 are also supported simply replace `3.13` with `3.11` or `3.12` in the `uv` command above.
    :::

1. Install the NeMo Agent toolkit library.
    To install the NeMo Agent toolkit library along with all of the optional dependencies. Including developer tools (`--all-groups`) and all of the dependencies needed for profiling and plugins (`--all-extras`) in the source repository, run the following:
    ```bash
    uv sync --all-groups --all-extras
    ```

    Alternatively to install just the core NeMo Agent toolkit without any optional plugins, run the following:
    ```bash
    uv sync
    ```

    At this point individual plugins, which are located under the `packages` directory, can be installed with the following command `uv pip install -e '.[<plugin_name>]'`.
    For example, to install the LangChain/LangGraph plugin, run the following:
    ```bash
    uv pip install -e '.[langchain]'
    ```

    :::{note}
    Many of the example workflows require plugins, and following the documented steps in one of these examples will in turn install the necessary plugins. For example following the steps in the `examples/getting_started/simple_web_query/README.md` guide will install the `nvidia-nat-langchain` plugin if you haven't already done so.
    :::

    In addition to plugins, there are optional dependencies needed for profiling. Installing the `profiling` sub-package is required for [evaluation](../improve-workflows/evaluate.md) and profiling workflows using `nat eval`. To install these dependencies, run the following:
    ```bash
    uv pip install -e '.[profiling]'
    ```
1. Verify that you've installed the NeMo Agent toolkit library.

     ```bash
     nat --help
     nat --version
     ```

     If the installation succeeded, the `nat` command will log the help message and its current version.

## Next Steps

* Follow the [Quick Start Guide](./quick-start.md) to get started running workflows with NeMo Agent toolkit.
