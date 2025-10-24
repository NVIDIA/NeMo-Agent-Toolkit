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

# Building an Agentic System using NeMo Agent Toolkit

Through this series of notebooks, we demonstrate how you can use the NVIDIA NeMo Agent toolkit to build, connect, evaluate, profile, and deploy an agentic system.

We showcase the building blocks that make up the agentic system, including tools, agents, workflows, and observability.

1. [Getting Started](1_getting_started_with_nat.ipynb) - Getting started with the NeMo Agent toolkit
2. [Bringing Your Own Agent](2_bringing_your_own_agent.ipynb) - Bringing your own agent to the NeMo Agent toolkit
3. [Adding Tools and Agents](3_adding_tools_to_agents.ipynb) - Adding tools to your agentic workflow
4. [Multi-Agent Orchestration](4_multi_agent_orchestration.ipynb) - Setting up a multi-agent orchestration workflow
5. [Observability, Evaluation, and Profiling](5_observability_evaluation_and_profiling.ipynb) - Instrumenting with observability, evaluation and profiling tools
   
We recommend opening these notebooks in a Jupyter Lab environment or Google Colab environment.

We also have a set of notebooks that are designed to be run in a Brev environment. See the [Brev Launchables](./launchables/README.md) for more details.

## Google Colab

To open these notebooks in a Google Colab environment, you can click the following link: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/NeMo-Agent-Toolkit/)

## Jupyter Lab
If you want to run these notebooks locally, you can clone the repository and open the notebooks in a Jupyter Lab environment. To install the necessary dependencies, you can run the following command:

```bash
uv venv --seed .venv
source .venv/bin/activate
uv pip install jupyterlab
```

Assuming you have cloned the repository and are in the root directory, you can open the notebooks in a Jupyter Lab environment by running the following command:

```bash
jupyter lab examples/notebooks
```
