<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http:/www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

![NVIDIA Agent Intelligence Toolkit](./docs/source/_static/aiqtoolkit_banner.png "AIQ Toolkit banner image")

# NVIDIA Agent Intelligence Toolkit

Agent Intelligence Toolkit (AIQ Toolkit) is a flexible library designed to seamlessly integrate your enterprise agents—regardless of framework—with various data sources and tools. By treating agents, tools, and agentic workflows as simple function calls, AIQ Toolkit enables true composability: build once and reuse anywhere.

## Key Features

- [**Framework Agnostic:**](https://docs.nvidia.com/aiqtoolkit/latest/concepts/plugins.html) Works with any agentic framework, so you can use your current technology stack without replatforming.
- [**Reusability:**](https://docs.nvidia.com/aiqtoolkit/latest/guides/sharing-workflows-and-tools.html) Every agent, tool, or workflow can be combined and repurposed, allowing developers to leverage existing work in new scenarios.
- [**Rapid Development:**](https://docs.nvidia.com/aiqtoolkit/latest/guides/create-customize-workflows.html) Start with a pre-built agent, tool, or workflow, and customize it to your needs.
- [**Profiling:**](https://docs.nvidia.com/aiqtoolkit/latest/guides/profiler.html) Profile entire workflows down to the tool and agent level, track input/output tokens and timings, and identify bottlenecks.
- [**Observability:**](https://docs.nvidia.com/aiqtoolkit/latest/guides/observe-workflow-with-phoenix.html) Monitor and debug your workflows with any OpenTelemetry-compatible observability tool.
- [**Evaluation System:**](https://docs.nvidia.com/aiqtoolkit/latest/guides/evaluate.html) Validate and maintain accuracy of agentic workflows with built-in evaluation tools.
- [**User Interface:**](https://docs.nvidia.com/aiqtoolkit/latest/guides/using-aiqtoolkit-ui-and-server.html) Use the AIQ Toolkit UI chat interface to interact with your agents, visualize output, and debug workflows.
- [**MCP Compatibility**](https://docs.nvidia.com/aiqtoolkit/latest/components/mcp.html) Compatible with Model Context Protocol (MCP), allowing tools served by MCP Servers to be used as AIQ Toolkit functions.

With AIQ Toolkit, you can move quickly, experiment freely, and ensure reliability across all your agent-driven projects.

## Component Overview

The following diagram illustrates the key components of AIQ Toolkit and how they interact. It provides a high-level view of the architecture, including agents, plugins, workflows, and user interfaces. Use this as a reference to understand how to integrate and extend AIQ Toolkit in your projects.

![AIQ Toolkit Components Diagram](docs/source/_static/aiqtoolkit_gitdiagram.png)

## Links

 * [Documentation](https://docs.nvidia.com/aiqtoolkit/latest/index.html): Explore the full documentation for AIQ Toolkit.
 * [About AIQ Toolkit](https://docs.nvidia.com/aiqtoolkit/latest/intro/why-aiqtoolkit.html): Learn more about the benefits of using AIQ Toolkit.
 * [Get Started Guide](https://docs.nvidia.com/aiqtoolkit/latest/intro/get-started.html): Set up your environment and start building with AIQ Toolkit.
 * [Examples](https://github.com/NVIDIA/AIQToolkit/tree/main/examples#readme): Explore examples of AIQ Toolkit workflows.
 * [Create and Customize AIQ Toolkit Workflows](https://docs.nvidia.com/aiqtoolkit/latest/guides/create-customize-workflows.html): Learn how to create and customize AIQ Toolkit workflows.
 * [Evaluate with AIQ Toolkit](https://docs.nvidia.com/aiqtoolkit/latest/guides/evaluate.html): Learn how to evaluate your AIQ Toolkit workflows.
 * [Troubleshooting](https://docs.nvidia.com/aiqtoolkit/latest/troubleshooting.html): Get help with common issues.


## Get Started

### Prerequisites

Before you begin using AIQ Toolkit, ensure that you meet the following software prerequisites.

- Install [Git](https://git-scm.com/)
- Install [Git Large File Storage](https://git-lfs.github.com/) (LFS)
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Install [Python (3.11 or 3.12)](https://www.python.org/downloads/)

### Install From Source

1. Clone the AIQ Toolkit repository to your local machine.
    ```bash
    git clone git@github.com:NVIDIA/AIQToolkit.git aiqtoolkit
    cd aiqtoolkit
    ```

2. Initialize, fetch, and update submodules in the Git repository.
    ```bash
    git submodule update --init --recursive
    ```

3. Fetch the data sets by downloading the LFS files.
    ```bash
    git lfs install
    git lfs fetch
    git lfs pull
    ```

4. Create a Python environment.
    ```bash
    uv venv --seed .venv
    source .venv/bin/activate
    ```
    Make sure the environment is built with Python version `3.11` or `3.12`. If you have multiple Python versions installed,
    you can specify the desired version using the `--python` flag. For example, to use Python 3.11:
    ```bash
    uv venv --seed .venv --python 3.11
    ```
    You can replace `--python 3.11` with any other Python version (`3.11` or `3.12`) that you have installed.

5. Install the AIQ Toolkit library.
    To install the AIQ Toolkit library along with all of the optional dependencies. Including developer tools (`--all-groups`) and all of the dependencies needed for profiling and plugins (`--all-extras`) in the source repository, run the following:
    ```bash
    uv sync --all-groups --all-extras
    ```

    Alternatively to install just the core AIQ Toolkit without any plugins, run the following:
    ```bash
    uv sync
    ```

    At this point individual plugins, which are located under the `packages` directory, can be installed with the following command `uv pip install -e '.[<plugin_name>]'`.
    For example, to install the `langchain` plugin, run the following:
    ```bash
    uv pip install -e '.[langchain]'
    ```

    > [!NOTE]
    > Many of the example workflows require plugins, and following the documented steps in one of these examples will in turn install the necessary plugins. For example following the steps in the `examples/simple/README.md` guide will install the `aiqtoolkit-langchain` plugin if you haven't already done so.


    In addition to plugins, there are optional dependencies needed for profiling. To install these dependencies, run the following:
    ```bash
    uv pip install -e '.[profiling]'
    ```

6. Verify the installation using the AIQ Toolkit CLI

   ```bash
   aiq --version
   ```

   This should output the AIQ Toolkit version which is currently installed.

## Hello World Example

1. Ensure you have set the `NVIDIA_API_KEY` environment variable to allow the example to use NVIDIA NIMs. An API key can be obtained by visiting [`build.nvidia.com`](https://build.nvidia.com/) and creating an account.

   ```bash
   export NVIDIA_API_KEY=<your_api_key>
   ```

2. Create the AIQ Toolkit workflow configuration file. This file will define the agents, tools, and workflows that will be used in the example. Save the following as `workflow.yaml`:

   ```yaml
   functions:
      # Add a tool to search wikipedia
      wikipedia_search:
         _type: wiki_search
         max_results: 2

   llms:
      # Tell AIQ Toolkit which LLM to use for the agent
      nim_llm:
         _type: nim
         model_name: meta/llama-3.1-70b-instruct
         temperature: 0.0

   workflow:
      # Use an agent that 'reasons' and 'acts'
      _type: react_agent
      # Give it access to our wikipedia search tool
      tool_names: [wikipedia_search]
      # Tell it which LLM to use
      llm_name: nim_llm
      # Make it verbose
      verbose: true
      # Retry parsing errors because LLMs are non-deterministic
      retry_parsing_errors: true
      # Retry up to 3 times
      max_retries: 3
   ```

3. Run the Hello World example using the `aiq` CLI and the `workflow.yaml` file.

   ```bash
   aiq run --config_file workflow.yaml --input "List five subspecies of Aardvarks"
   ```

   This will run the workflow and output the results to the console.

   ```console
   Workflow Result:
   ['Here are five subspecies of Aardvarks:\n\n1. Orycteropus afer afer (Southern aardvark)\n2. O. a. adametzi  Grote, 1921 (Western aardvark)\n3. O. a. aethiopicus  Sundevall, 1843\n4. O. a. angolensis  Zukowsky & Haltenorth, 1957\n5. O. a. erikssoni  Lönnberg, 1906']
   ```

## Feedback

We would love to hear from you! Please file an issue on [GitHub](https://github.com/NVIDIA/AIQToolkit/issues) if you have any feedback or feature requests.

## Acknowledgements

We would like to thank the following open source projects that made AIQ Toolkit possible:

- [CrewAI](https://github.com/crewAIInc/crewAI)
- [FastAPI](https://github.com/tiangolo/fastapi)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Llama-Index](https://github.com/run-llama/llama_index)
- [Mem0ai](https://github.com/mem0ai/mem0)
- [Ragas](https://github.com/explodinggradients/ragas)
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel)
- [uv](https://github.com/astral-sh/uv)
