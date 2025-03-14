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

<!--
  SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# A Simple LangSmith-Documentation Agent

A minimal example demonstrating a simple LangSmith-Documentation agent. This agent leverages the AgentIQ plugin system and `Builder` to integrate pre-built and custom tools into the workflow to answer questions about LangSmith. Key elements are summarized below:

## Table of Contents

* [Key Features](#key-features)
* [Installation and Usage](#installation-and-setup)
* [Deployment-Oriented Setup](#docker-quickstart)

---

## Key Features

- **Pre-built Tools:** Leverages core AgentIQ library tools.
- **Custom Plugin System:** Developers can bring in new tools using plugins.
- **High-level API:** Enables defining functions that transform into asynchronous LangChain tools.
- **Agentic Workflows:** Fully configurable via YAML for flexibility and productivity.
- **Ease of Use:** Simplifies developer experience and deployment.

---

## Installation and Setup

### Setup Virtual Environment and Install AgentIQ

From the root directory of the AgentIQ library, run the following commands. Skip this if you've already installed AgentIQ.

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install uv
uv sync --all-extras
```

### Install this Workflow:

From the root directory of the AgentIQ library, run the following commands:

```bash
uv pip install -e examples/simple
```

### Set Up API Keys

You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

### Run the Workflow

Run the following command from the root of the AgentIQ repo to execute this workflow with the specified input:

```bash
aiq run --config_file examples/simple/configs/config.yml --input "What is LangSmith?"
```

**Expected Output**

```console
$ aiq run --config_file=examples/simple/configs/config.yml --input "What is LangSmith"
2025-01-29 15:33:57,665 - aiq.cli.run - INFO - Loading configuration from: examples/simple/configs/config.yml
2025-01-29 15:33:59,375 - langchain_community.utils.user_agent - WARNING - USER_AGENT environment variable not set, consider setting it to identify your requests.
2025-01-29 15:33:59,410 - aiq_simple.register - INFO - Generating docs for the webpage: https://docs.smith.langchain.com/user_guide
2025-01-29 15:34:00,725 - faiss.loader - INFO - Loading faiss.
2025-01-29 15:34:00,816 - faiss.loader - INFO - Successfully loaded faiss.
2025-01-29 15:34:01,255 - aiq.workflows.react_agent.agent - INFO - Initializing ReAct Agent Graph
2025-01-29 15:34:01,255 - aiq.workflows.react_agent.agent - INFO - Filling the prompt variables "tools" and "tool_names", using the tools provided in the config.
2025-01-29 15:34:01,255 - aiq.workflows.react_agent.agent - INFO - Initialized ReAct Agent Graph
2025-01-29 15:34:01,255 - aiq.workflows.react_agent.agent - INFO - Building and compiling the ReAct Agent Graph
2025-01-29 15:34:01,256 - aiq.workflows.react_agent.agent - INFO - ReAct Graph built and compiled successfully

Configuration Summary:
--------------------
Workflow Type: react_agent
Number of Tools: 3
Number of LLMs: 1
Number of Embedders: 0
Number of Memory: 0

2025-01-29 15:34:01,256 - aiq.cli.run - INFO - Processing input: ('What is LangSmith',)
2025-01-29 15:34:02,028 - aiq.workflows.react_agent.agent - INFO - The user's question was: What is LangSmith
2025-01-29 15:34:02,029 - aiq.workflows.react_agent.agent - INFO - The agent's thoughts are:
Thought: I need to find information about LangSmith to answer this question.
Action: webpage_query
Action Input: LangSmith
Observation: Waiting for the search results...
2025-01-29 15:34:02,034 - aiq.workflows.react_agent.agent - INFO - Calling tool webpage_query with input: LangSmith
2025-01-29 15:34:04,385 - aiq.workflows.react_agent.agent - INFO -

The agent's thoughts are:
Thought: I now know the final answer
Final Answer: LangSmith is a platform for LLM (Large Language Model) application development, monitoring, and testing. It supports various workflows throughout the application development lifecycle, including annotating traces, adding runs to datasets, creating initial test sets, automations, and threads. LangSmith allows developers to create datasets, run tests, and score test results, as well as automate actions on traces in near real-time. It also provides a threads view to group traces from a single conversation together, making it easier to track performance and annotate applications across multiple turns.
2025-01-29 15:34:04,387 - aiq.cli.run - INFO - --------------------------------------------------
Workflow Result:
['LangSmith is a platform for LLM (Large Language Model) application development, monitoring, and testing. It supports various workflows throughout the application development lifecycle, including annotating traces, adding runs to datasets, creating initial test sets, automations, and threads. LangSmith allows developers to create datasets, run tests, and score test results, as well as automate actions on traces in near real-time. It also provides a threads view to group traces from a single conversation together, making it easier to track performance and annotate applications across multiple turns.']
--------------------------------------------------
Cleaning up react_agent workflow.
2025-01-29 15:34:04,408 - aiq.cli.entrypoint - INFO - Total time: 6.75 sec
2025-01-29 15:34:04,408 - aiq.cli.entrypoint - INFO - Pipeline runtime: 3.15 sec
```

## Docker Quickstart

Prior to building the Docker image ensure that you have followed the steps in the [Installation and Setup](#installation-and-setup) section, and you are currently in the AgentIQ virtual environment.

Set your NVIDIA API Key in the `NVIDIA_API_KEY` environment variable.

```bash
export NVIDIA_API_KEY="your_nvidia_api_key"
```

From the git repository root, run the following command to build AgentIQ and the simple agent into a Docker image.

```bash
docker build --build-arg AIQ_VERSION=$(python -m setuptools_scm) -f examples/simple/Dockerfile -t simple-agent .
```

Then, run the following command to run the simple agent.

```bash
docker run -p 8000:8000 -e NVIDIA_API_KEY simple-agent
```

After the container starts, you can access the agent at http://localhost:8000.

```bash
curl -X 'POST' \
  'http://localhost:8000/generate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"input_message": "What is LangSmith?"}'
```
