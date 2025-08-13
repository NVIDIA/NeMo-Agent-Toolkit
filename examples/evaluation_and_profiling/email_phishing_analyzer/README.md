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


# Email phishing analyzer

## Table of Contents

- [Key Features](#key-features)
- [Installation and Setup](#installation-and-setup)
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

- **Email Security Analysis:** Demonstrates an `email_phishing_analyzer` tool that examines email content for suspicious patterns, social engineering tactics, and phishing indicators using LLM-based analysis.
- **ReAct Agent Integration:** Uses a `react_agent` that can reason about email content and determine when to invoke the phishing analysis tool based on suspicious characteristics.
- **Phishing Detection Workflow:** Shows how to analyze emails for common phishing techniques including requests for sensitive information, urgency tactics, and suspicious sender patterns.
- **Security-Focused LLM Application:** Demonstrates how to apply AI reasoning to cybersecurity use cases with specialized prompting and analysis workflows.
- **Threat Assessment Pipeline:** Provides a foundation for building automated email security screening systems that can classify potential threats.


## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install NeMo Agent toolkit.

### Install this Workflow:

From the root directory of the NeMo Agent toolkit library, run the following commands:

```bash
uv pip install -e examples/evaluation_and_profiling/email_phishing_analyzer
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

## Example Usage

### Run the Workflow

Run the following command from the root of the NeMo Agent toolkit repo to execute this workflow with the specified input:

```bash
aiq run --config_file examples/evaluation_and_profiling/email_phishing_analyzer/configs/config.yml --input "Dear [Customer], Thank you for your purchase on [Date]. We have processed a refund of $[Amount] to your account. Please provide your account and routing numbers so we can complete the transaction. Thank you, [Your Company]"
```

The configuration file specified above contains configurations for the NeMo Agent Toolkit `evaluation` and `profiler` capabilities. Additional documentation for evaluation configuration can be found in the [evaluation guide](../../../docs/source/workflows/evaluate.md). Furthermore, similar documentation for profiling configuration can be found in the [profiling guide](../../../docs/source/workflows/profiler.md).

**Expected Workflow Output**
```console
2025-04-23 15:24:54,183 - aiq.runtime.loader - WARNING - Loading module 'aiq_automated_description_generation.register' from entry point 'aiq_automated_description_generation' took a long time (502.501011 ms). Ensure all imports are inside your registered functions.
2025-04-23 15:24:54,483 - aiq.cli.commands.start - INFO - Starting AIQ toolkit from config file: 'examples/evaluation_and_profiling/email_phishing_analyzer/configs/config.yml'
2025-04-23 15:24:54,495 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.

Configuration Summary:
--------------------
Workflow Type: react_agent
Number of Functions: 1
Number of LLMs: 3
Number of Embedders: 0
Number of Memory: 0
Number of Retrievers: 0

2025-04-23 15:24:58,017 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: Dear [Customer], Thank you for your purchase on [Date]. We have processed a refund of 0 to your account. Please provide your account and routing numbers so we can complete the transaction. Thank you, [Your Company]
Agent's thoughts:
Thought: This email seems suspicious as it asks for sensitive information such as account and routing numbers. I should analyze it for signs of phishing.

Action: email_phishing_analyzer
Action Input: {'text': 'Dear [Customer], Thank you for your purchase on [Date]. We have processed a refund of 0 to your account. Please provide your account and routing numbers so we can complete the transaction. Thank you, [Your Company]'}
Observation
------------------------------
/AIQToolkit/examples/evaluation_and_profiling/email_phishing_analyzer/src/nat_email_phishing_analyzer/register.py:56: LangChainDeprecationWarning: The method `BaseChatModel.apredict` was deprecated in langchain-core 0.1.7 and will be removed in 1.0. Use :meth:`~ainvoke` instead.
  response = await llm.apredict(config.prompt.format(body=text))
2025-04-23 15:25:07,477 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Calling tools: email_phishing_analyzer
Tool's input: {"text": "Dear [Customer], Thank you for your purchase on [Date]. We have processed a refund of 0 to your account. Please provide your account and routing numbers so we can complete the transaction. Thank you, [Your Company]"}
Tool's response:
{"is_likely_phishing": true, "explanation": "The email exhibits suspicious signals that may indicate phishing. Specifically, the email requests sensitive personal information (account and routing numbers) under the guise of completing a refund transaction. Legitimate companies typically do not request such information via email, as it is a security risk. Additionally, the refund amount of '0' is unusual and may be an attempt to create a sense of urgency or confusion. The tone of the email is also somewhat generic and lacks personalization, which is another common trait of phishing emails."}
------------------------------
2025-04-23 15:25:08,862 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: Dear [Customer], Thank you for your purchase on [Date]. We have processed a refund of 0 to your account. Please provide your account and routing numbers so we can complete the transaction. Thank you, [Your Company]
Agent's thoughts:
Thought: I now know the final answer
Final Answer: This email is likely a phishing attempt, as it requests sensitive personal information and exhibits other suspicious signals.
------------------------------
2025-04-23 15:25:08,866 - aiq.front_ends.console.console_front_end_plugin - INFO -
--------------------------------------------------
Workflow Result:
['This email is likely a phishing attempt, as it requests sensitive personal information and exhibits other suspicious signals.']
```

---

## Deployment-Oriented Setup

For a production deployment, use Docker:

### Build the Docker Image

Prior to building the Docker image ensure that you have followed the steps in the [Installation and Setup](#installation-and-setup) section, and you are currently in the NeMo Agent toolkit virtual environment.

From the root directory of the Simple Calculator repository, build the Docker image:

```bash
docker build --build-arg AIQ_VERSION=$(python -m setuptools_scm) -t email_phishing_analyzer -f examples/evaluation_and_profiling/email_phishing_analyzer/Dockerfile .
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

```json
{"value":"This email is likely a phishing attempt. It requests sensitive information, such as account and routing numbers, which is a common tactic used by scammers. The email also lacks specific details about the purchase, which is unusual for a refund notification. Additionally, the greeting is impersonal, which suggests a lack of personalization. It is recommended to be cautious when responding to such emails and to verify the authenticity of the email before providing any sensitive information."}
```
