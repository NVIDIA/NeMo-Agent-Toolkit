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

If you have not already done so, follow the instructions in the [Install Guide](../../docs/source/intro/install.md) to create the development environment and install AgentIQ.

### Install this Workflow:

From the root directory of the AgentIQ library, run the following commands:

```bash
uv pip install -e examples/simple
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../docs/source/intro/get-started.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

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
$ aiq run --config_file examples/simple/configs/config.yml --input "What is LangSmith?"
2025-04-23 15:53:15,873 - aiq.runtime.loader - WARNING - Loading module 'aiq_automated_description_generation.register' from entry point 'aiq_automated_description_generation' took a long time (446.926117 ms). Ensure all imports are inside your registered functions.
2025-04-23 15:53:16,192 - aiq.cli.commands.start - INFO - Starting AgentIQ from config file: 'examples/simple/configs/config.yml'
2025-04-23 15:53:16,197 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.
2025-04-23 15:53:16,243 - aiq.profiler.utils - WARNING - Discovered frameworks: {<LLMFrameworkEnum.LANGCHAIN: 'langchain'>} in function webquery_tool by inspecting source. It is recommended and more reliable to instead add the used LLMFrameworkEnum types in the framework_wrappers argument when calling @register_function.
2025-04-23 15:53:16,251 - langchain_community.utils.user_agent - WARNING - USER_AGENT environment variable not set, consider setting it to identify your requests.
2025-04-23 15:53:16,262 - aiq_simple.register - INFO - Generating docs for the webpage: https://docs.smith.langchain.com
Fetching pages: 100%|#########################################################################################| 1/1 [00:00<00:00, 13.51it/s]
2025-04-23 15:53:16,769 - faiss.loader - INFO - Loading faiss with AVX2 support.
2025-04-23 15:53:16,873 - faiss.loader - INFO - Successfully loaded faiss with AVX2 support.

Configuration Summary:
--------------------
Workflow Type: react_agent
Number of Functions: 2
Number of LLMs: 1
Number of Embedders: 1
Number of Memory: 0
Number of Retrievers: 0

2025-04-23 15:53:18,031 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: What is LangSmith?
Agent's thoughts:
Thought: To answer this question, I need to find information about LangSmith.

Action: webpage_query
Action Input: {"query": "LangSmith"}


------------------------------
2025-04-23 15:53:18,290 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Calling tools: webpage_query
Tool's input: {"query": "LangSmith"}
Tool's response:
Get started with LangSmith | 🦜️🛠️ LangSmith

LangSmith is a platform for building production-grade LLM applications.
It allows you to closely monitor and evaluate your application, so you can ship quickly and with confidence.
ObservabilityAnalyze traces in LangSmith and configure metrics, dashboards, alerts based on these.EvalsEvaluate your application over production traffic — score application performance and get human feedback on your data.Prompt EngineeringIterate on prompts, with automatic version control and collaboration features.

Skip to main contentWe are growing and hiring for multiple roles for LangChain, LangGraph and LangSmith. Join our team!API ReferenceRESTPythonJS/TSSearchRegionUSEUGo to AppGet StartedObservabilityEvaluationPrompt EngineeringDeployment (LangGraph Platform)AdministrationSelf-hostingPricingReferenceCloud architecture and scalabilityAuthz and AuthnAuthentication methodsdata_formatsEvaluationDataset transformationsRegions FAQsdk_referenceGet StartedOn this...
------------------------------
2025-04-23 15:53:19,303 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: What is LangSmith?
Agent's thoughts:
Thought: I now know the final answer

Final Answer: LangSmith is a platform for building production-grade LLM (Large Language Model) applications, allowing users to monitor and evaluate their applications, and providing features such as observability, evaluation, prompt engineering, and deployment.
------------------------------
2025-04-23 15:53:19,307 - aiq.front_ends.console.console_front_end_plugin - INFO -
--------------------------------------------------
Workflow Result:
['LangSmith is a platform for building production-grade LLM (Large Language Model) applications, allowing users to monitor and evaluate their applications, and providing features such as observability, evaluation, prompt engineering, and deployment.']
--------------------------------------------------
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
