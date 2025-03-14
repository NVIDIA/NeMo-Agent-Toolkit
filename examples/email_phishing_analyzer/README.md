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

# Email phishing analyzer

## Table of Contents

- [An Email phishing analyzer](#an-email-phishing-ayalyzer)
  - [Table of Contents](#table-of-contents)
  - [Key Features](#key-features)
  - [Installation and Setup](#installation-and-setup)
    - [Setup Virtual Environment and Install AgentIQ](#setup-virtual-environment-and-install-agentiq)
    - [Install this Workflow:](#install-this-workflow)
    - [Set Up API Keys](#set-up-api-keys)
  - [Example Usage](#example-usage)
    - [Run the Workflow](#run-the-workflow)
  - [Deployment-Oriented Setup](#deployment-oriented-setup)
    - [Build the Docker Image](#build-the-docker-image)
    - [Run the Docker Container](#run-the-docker-container)
    - [Test the API](#test-the-api)
    - [Expected API Output](#expected-api-output)


## Key Features

- **Pre-built Tools:** Leverages core AgentIQ library agent and tools.
- **ReAct Agent:** Performs reasoning between tool call; utilizes tool names and descriptions to appropriately route to the correct tool
- **Custom Plugin System:** Developers can bring in new tools using plugins.
- **High-level API:** Enables defining functions that transform into asynchronous LangChain tools.
- **Agentic Workflows:** Fully configurable via YAML for flexibility and productivity.
- **Ease of Use:** Simplifies developer experience and deployment.


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
uv pip install -e examples/email_phishing_analyzer
```

### Set Up API Keys

You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

## Example Usage

### Run the Workflow

Run the following command from the root of the AgentIQ repo to execute this workflow with the specified input:

```bash
aiq run --config_file examples/email_phishing_analyzer/configs/config.yml --input "Dear [Customer], Thank you for your purchase on [Date]. We have processed a refund of $[Amount] to your account. Please provide your account and routing numbers so we can complete the transaction. Thank you, [Your Company]"
```

**Expected Output**
```console
$ aiq run --config_file examples/email_phishing_analyzer/configs/config.yml --input "Dear [Customer], Thank you for your purchase on [Date]. We have processed a refund of $[Amount] to your account. Please provide your account and routing numbers so we can complete the transaction. Thank you, [Your Company]"
2025-03-10 21:00:37,349 - aiq.cli.commands.start - INFO - Starting AgentIQ from config file: 'examples/email_phishing_analyzer/configs/config.yml'
2025-03-10 21:00:37,355 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.
2025-03-10 21:00:37,504 - aiq.profiler.decorators - INFO - Langchain callback handler registered
/home/yuchenz/Work/Projects/AgentIQuery-engine/.venv/lib/python3.12/site-packages/langchain_nvidia_ai_endpoints/chat_models.py:591: UserWarning: Model 'meta/llama-3.1-405b-instruct' is not known to support tools. Your tool binding may fail at inference time.
  warnings.warn(
2025-03-10 21:00:37,809 - aiq.agent.tool_calling_agent.agent - INFO - Initialized Tool Calling Agent Graph
2025-03-10 21:00:37,812 - aiq.agent.tool_calling_agent.agent - INFO - Tool Calling Agent Graph built and compiled successfully

Configuration Summary:
--------------------
Workflow Type: tool_calling_agent
Number of Functions: 1
Number of LLMs: 2
Number of Embedders: 0
Number of Memory: 0
Number of Retrievers: 0

2025-03-10 21:00:37,814 - aiq.front_ends.console.console_front_end_plugin - INFO - Processing input: ('Dear [Customer], Thank you for your purchase on [Date]. We have processed a refund of 0 to your account. Please provide your account and routing numbers so we can complete the transaction. Thank you, [Your Company]',)
2025-03-10 21:00:37,817 - aiq.agent.tool_calling_agent.agent - INFO - Calling agent
2025-03-10 21:02:47,001 - aiq.agent.tool_calling_agent.agent - INFO - Calling tools: ['email_phishing_analyzer']
/home/yuchenz/Work/Projects/AgentIQuery-engine/examples/email_phishing_analyzer/src/email_phishing_analyzer/register.py:42: LangChainDeprecationWarning: The method `BaseChatModel.apredict` was deprecated in langchain-core 0.1.7 and will be removed in 1.0. Use :meth:`~ainvoke` instead.
  response = await llm.apredict(config.prompt.format(body=text))
2025-03-10 21:03:00,747 - aiq.agent.tool_calling_agent.agent - INFO - Calling agent
2025-03-10 21:03:13,356 - aiq.observability.async_otel_listener - INFO - Intermediate step stream completed. No more events will arrive.
2025-03-10 21:03:13,356 - aiq.front_ends.console.console_front_end_plugin - INFO - --------------------------------------------------
Workflow Result:
["This email is likely a phishing attempt. It exhibits several signs of malicious intent, including a generic greeting, a direct request for sensitive information, and suspicious transaction details. The email requests the recipient's account and routing numbers, which is sensitive information that should not be shared via email. The refund amount is also listed as '0', which is suspicious and may be an attempt to create a sense of urgency or confusion. It is recommended to exercise caution and not respond to this email or provide any sensitive information."]
--------------------------------------------------
```
---

## Deployment-Oriented Setup

For a production deployment, use Docker:

### Build the Docker Image

Prior to building the Docker image ensure that you have followed the steps in the [Installation and Setup](#installation-and-setup) section, and you are currently in the AgentIQ virtual environment.

From the root directory of the Simple Calculator repository, build the Docker image:

```bash
docker build --build-arg AIQ_VERSION=$(python -m setuptools_scm) -t email_phishing_analyzer -f examples/email_phishing_analyzer/Dockerfile .
```

### Run the Docker Container
Deploy the container:

```bash
docker run -p 8000:8000 -e NVIDIA_API_KEY email_phishing_analyzer
```

### Test the API
Use the following curl command to test the deployed API:

```bash
curl -X 'POST' \
  'http://localhost:8000/generate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"input_message": "Dear [Customer], Thank you for your purchase on [Date]. We have processed a refund of $[Amount] to your account. Please provide your account and routing numbers so we can complete the transaction. Thank you, [Your Company]"}'
  ```

### Expected API Output
The API response should look like this:

```bash
{"value":"This email is likely a phishing attempt. It requests sensitive information, such as account and routing numbers, which is a common tactic used by scammers. The email also lacks specific details about the purchase, which is unusual for a refund notification. Additionally, the greeting is impersonal, which suggests a lack of personalization. It is recommended to be cautious when responding to such emails and to verify the authenticity of the email before providing any sensitive information."}
```
