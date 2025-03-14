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

# Install NVIDIA AgentIQ
AgentIQ can be installed in the following ways, depending on your needs:
- **`pip install`**: Quickly install AgentIQ.
- **Install from the source**: Installs AgentIQ and includes examples of using AgentIQ.

AgentIQ is a Python library that doesn’t require a GPU to run the workflow by default. You can deploy the core workflows using one of the following:
- Ubuntu or other Linux distributions, including WSL, in a Python virtual environment.

## Install AgentIQ with `pip`
This is suitable for users who want to quickly install and start building agentic workflows.
**Note:** Ensure you have a Python environment with Python 3.12 or higher

To install AgentIQ with `pip install`:

1. Install AgentIQ and include the framework that you're using.

   ```bash
   pip install agentiq[<your framework>]
   ```

   For example, to install the LangChain integration, run the following:
   ```bash
   pip install agentiq[langchain]
   ```
   For example, to install both LangChain and LlamaIndex, run the following:
   ```bash
   pip install agentiq[langchain, llama-index]
   ```
   :::{note}
   This installs the core AgentIQ package and the plugin of your framework. AgentIQ also supports other LLM frameworks. For more information, refer to [Framework Integrations](../concepts/plugins.md#framework-integrations).
   :::

2. Verify the installation

   ```bash
   aiq --version
   ```

   This should output the current AgentIQ version installed.


## Install From Source

### Prerequisites

Before you begin using AgentIQ, ensure that you meet the following software prerequisites.

- Install [Git](https://git-scm.com/)
- Install [Git Large File Storage](https://git-lfs.github.com/) (LFS)
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/)


1. Clone the AgentIQ repository to your local machine.
    ```bash
    git clone git@github.com:NVIDIA/AgentIQ.git agentiq
    cd agentiq
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
    uv venv --seed .venv
    source .venv/bin/activate
    ```

1. Install the AgentIQ library.
    To install the AgentIQ library along with all of the optional dependencies. Including developer tools (`--all-groups`) and all of the dependencies needed for profiling and plugins (`--all-extras`) in the source repository, run the following:
    ```bash
    uv sync --all-groups --all-extras
    ```

    Alternatively to install just the core AgentIQ without any plugins, run the following:
    ```bash
    uv sync
    ```

    At this point individual plugins, which are located under the `packages` directory, can be installed with the following command `uv pip install -e '.[<plugin_name>]'`.
    For example, to install the `langchain` plugin, run the following:
    ```bash
    uv pip install -e '.[langchain]'
    ```

    :::{note}
    Many of the example workflows require plugins, and following the documented steps in one of these examples will in turn install the necessary plugins. For example following the steps in the `examples/simple/README.md` guide will install the `agentiq-langchain` plugin if you haven't already done so.
    :::

    In addition to plugins, there are optional dependencies needed for profiling. To install these dependencies, run the following:
    ```bash
    uv pip install -e .[profiling]
    ```


2. Verify that you've installed the AgentIQ library.

     ```bash
     aiq --help
     aiq --version
     ```

     If the installation succeeded, the `aiq` command will log the help message and its current version.

## Next Steps
After installing AgentIQ, you can start using AgentIQ agentic workflows. For more information, refer to [Get Started with AgentIQ](get-started.md).
