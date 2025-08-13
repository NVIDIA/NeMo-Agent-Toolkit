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

# Installing NVIDIA NeMo Agent Toolkit

This guide will help you set up your NVIDIA NeMo Agent toolkit development environment, run existing workflows, and create your own custom workflows using the `aiq` command-line interface.

## Supported LLM APIs

The following LLM APIs are supported:

- NIM (such as Llama-3.1-70b-instruct and Llama-3.3-70b-instruct)
- OpenAI
- AWS Bedrock

## Framework Integrations

To keep the library lightweight, many of the first-party plugins supported by NeMo Agent toolkit are located in separate distribution packages. For example, the `aiqtoolkit-langchain` distribution contains all the LangChain-specific plugins and the `aiqtoolkit-mem0ai` distribution contains the Mem0-specific plugins.

To install these first-party plugin libraries, you can use the full distribution name (for example, `aiqtoolkit-langchain`) or use the `aiqtoolkit[langchain]` extra distribution. The following extras are supported:

- `aiqtoolkit[agno]` or `aiqtoolkit-agno` - [Agno](https://agno.com/)
- `aiqtoolkit[crewai]` or `aiqtoolkit-crewai` - [CrewAI](https://www.crewai.com/)
- `aiqtoolkit[langchain]` or `aiqtoolkit-langchain` - [LangChain](https://www.langchain.com/)
- `aiqtoolkit[llama-index]` or `aiqtoolkit-llama-index` - [LlamaIndex](https://www.llamaindex.ai/)
- `aiqtoolkit[mem0ai]` or `aiqtoolkit-mem0ai` - [Mem0](https://mem0.ai/)
- `aiqtoolkit[mysql]` or `aiqtoolkit-mysql` - [MySQL](https://www.mysql.com/)
- `aiqtoolkit[opentelemetry]` or `aiqtoolkit-opentelemetry` - [OpenTelemetry](https://opentelemetry.io/)
- `aiqtoolkit[phoenix]` or `aiqtoolkit-phoenix` - [Arize Phoenix](https://arize.com/docs/phoenix)
- `aiqtoolkit[ragaai]` or `aiqtoolkit-ragaai` - [RagaAI Catalyst](https://raga.ai/)
- `aiqtoolkit[redis]` or `aiqtoolkit-redis` - [Redis](https://redis.io/)
- `aiqtoolkit[s3]` or `aiqtoolkit-s3` - [Amazon S3](https://aws.amazon.com/s3/)
- `aiqtoolkit[semantic-kernel]` or `aiqtoolkit-semantic-kernel` - [Microsoft Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/)
- `aiqtoolkit[test]` or `aiqtoolkit-test` - NeMo Agent toolkit test
- `aiqtoolkit[weave]` or `aiqtoolkit-weave` - [Weights & Biases Weave](https://weave-docs.wandb.ai)
- `aiqtoolkit[zep-cloud]` or `aiqtoolkit-zep-cloud` - [Zep](https://www.getzep.com/)


## Prerequisites

NVIDIA NeMo Agent toolkit is a Python library that doesn't require a GPU to run by default. Before you begin using NeMo Agent toolkit, ensure that you meet the following software prerequisites:

- Install [Git](https://git-scm.com/)
- Install [Git Large File Storage](https://git-lfs.github.com/) (LFS)
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (version 0.5.4 or later, latest version is recommended)

## Install From Source

1. Clone the NeMo Agent toolkit repository to your local machine.
    ```bash
    git clone -b main git@github.com:NVIDIA/NeMo-Agent-Toolkit.git aiqtoolkit
    cd aiqtoolkit
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
    uv venv --python 3.12 --seed .venv
    source .venv/bin/activate
    ```
    :::{note}
    Python 3.11 is also supported simply replace `3.12` with `3.11` in the `uv` command above.
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
    For example, to install the `langchain` plugin, run the following:
    ```bash
    uv pip install -e '.[langchain]'
    ```

    :::{note}
    Many of the example workflows require plugins, and following the documented steps in one of these examples will in turn install the necessary plugins. For example following the steps in the `examples/getting_started/simple_web_query/README.md` guide will install the `aiqtoolkit-langchain` plugin if you haven't already done so.
    :::

    In addition to plugins, there are optional dependencies needed for profiling. To install these dependencies, run the following:
    ```bash
    uv pip install -e '.[profiling]'
    ```
1. Verify that you've installed the NeMo Agent toolkit library.

     ```bash
     aiq --help
     aiq --version
     ```

     If the installation succeeded, the `aiq` command will log the help message and its current version.


## Obtaining API Keys
Depending on which workflows you are running, you may need to obtain API keys from the respective services. Most NeMo Agent toolkit workflows require an NVIDIA API key defined with the `NVIDIA_API_KEY` environment variable. An API key can be obtained by visiting [`build.nvidia.com`](https://build.nvidia.com/) and creating an account.

### Optional OpenAI API Key
Some workflows may also require an OpenAI API key. Visit [OpenAI](https://openai.com/) and create an account. Navigate to your account settings to obtain your OpenAI API key. Copy the key and set it as an environment variable using the following command:

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

1. Install the `aiq_simple_web_query` Workflow

    ```bash
    uv pip install -e examples/getting_started/simple_web_query
    ```

2. Run the `aiq_simple_web_query` Workflow

    ```bash
    aiq run --config_file=examples/getting_started/simple_web_query/configs/config.yml --input "What is LangSmith"
    ```

3. **Run and evaluate the `aiq_simple_web_query` Workflow**

    The `eval_config.yml` YAML is a super-set of the `config.yml` containing additional fields for evaluation. To evaluate the `aiq_simple_web_query` workflow, run the following command:
    ```bash
    aiq eval --config_file=examples/evaluation_and_profiling/simple_web_query_eval/configs/eval_config.yml
    ```


## NeMo Agent Toolkit Packages
Once a NeMo Agent toolkit workflow is ready for deployment to production, the deployed workflow will need to declare a dependency on the `aiqtoolkit` package, along with the needed plugins. When declaring a dependency on NeMo Agent toolkit it is recommended to use the first two digits of the version number. For example if the version is `1.0.0` then the dependency would be `1.0`.

For more information on the available plugins, refer to [Framework Integrations](#framework-integrations).

Example dependency for NeMo Agent toolkit using the `langchain` plugin for projects using a `pyproject.toml` file:
```toml
dependencies = [
"aiqtoolkit[langchain]~=1.0",
# Add any additional dependencies your workflow needs
]
```

Alternately for projects using a `requirements.txt` file:
```
aiqtoolkit[langchain]==1.0.*
```

## Next Steps

* Explore the examples in the `examples` directory to learn how to build custom workflows and tools with NeMo Agent toolkit.
* Review the NeMo Agent toolkit tutorials for detailed guidance on using the toolkit.
