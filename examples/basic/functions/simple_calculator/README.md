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

# A Simple LLM Calculator

This example demonstrates an end-to-end (E2E) agentic workflow using the AIQ toolkit library, fully configured through a YAML file. It showcases the AIQ toolkit plugin system and `Builder` to seamlessly integrate pre-built and custom tools into workflows.

## Table of Contents

- [A Simple LLM Calculator](#a-simple-llm-calculator)
  - [Table of Contents](#table-of-contents)
  - [Key Features](#key-features)
  - [Installation and Setup](#installation-and-setup)
    - [Install this Workflow:](#install-this-workflow)
    - [Set Up API Keys](#set-up-api-keys)
    - [Run the Workflow](#run-the-workflow)
  - [Deployment-Oriented Setup](#deployment-oriented-setup)
    - [Build the Docker Image](#build-the-docker-image)
    - [Run the Docker Container](#run-the-docker-container)
    - [Test the API](#test-the-api)
    - [Expected API Output](#expected-api-output)


---

## Key Features

- **Pre-built Tools:** Leverages core AIQ toolkit library tools.
- **Custom Plugin System:** Developers can bring in new tools using plugins.
- **High-level API:** Enables defining functions that transform into asynchronous LangChain tools.
- **Agentic Workflows:** Fully configurable via YAML for flexibility and productivity.
- **Ease of Use:** Simplifies developer experience and deployment.

---

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install AIQ toolkit.

### Install this Workflow:

From the root directory of the AIQ toolkit library, run the following commands:

```bash
uv pip install -e examples/basic/functions/simple_calculator
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

### Run the Workflow

Return to your original terminal, and run the following command from the root of the AIQ toolkit repo to execute this workflow with the specified input:

```bash
aiq run --config_file examples/basic/functions/simple_calculator/configs/config.yml --input "Is the product of 2 * 4 greater than the current hour of the day?"
```

**Expected Output**
The workflow output can be quite lengthy, the end of the workflow output should contain something similar to the following (the final answer will depend on the time of day the workflow is run):
```console
$ aiq run --config_file examples/basic/functions/simple_calculator/configs/config.yml --input "Is the product of 2 * 4 greater than the current hour of the day?"
2025-04-23 15:58:34,877 - aiq.runtime.loader - WARNING - Loading module 'aiq_automated_description_generation.register' from entry point 'aiq_automated_description_generation' took a long time (440.151215 ms). Ensure all imports are inside your registered functions.
2025-04-23 15:58:35,193 - aiq.cli.commands.start - INFO - Starting AIQ toolkit from config file: 'examples/basic/functions/simple_calculator/configs/config.yml'
2025-04-23 15:58:35,199 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.

Configuration Summary:
--------------------
Workflow Type: react_agent
Number of Functions: 5
Number of LLMs: 2
Number of Embedders: 0
Number of Memory: 0
Number of Retrievers: 0

2025-04-23 15:58:36,674 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: Is the product of 2 * 4 greater than the current hour of the day?
Agent's thoughts:
Thought: To answer this question, I need to calculate the product of 2 and 4, and then compare it to the current hour of the day.

Action: calculator_multiply
Action Input: {'text': '2 * 4'}


------------------------------
2025-04-23 15:58:36,682 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Calling tools: calculator_multiply
Tool's input: {"text": "2 * 4"}
Tool's response:
The product of 2 * 4 is 8
------------------------------
2025-04-23 15:58:37,704 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: Is the product of 2 * 4 greater than the current hour of the day?
Agent's thoughts:
Thought: Now that I have the product of 2 and 4, I need to get the current hour of the day to compare it with the product.

Action: current_datetime
Action Input: None
------------------------------
2025-04-23 15:58:37,710 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Calling tools: current_datetime
Tool's input: None
Tool's response:
The current time of day is 2025-04-23 15:58:37
------------------------------
2025-04-23 15:58:38,865 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: Is the product of 2 * 4 greater than the current hour of the day?
Agent's thoughts:
Thought: Now that I have the current time of day, I can extract the hour and compare it with the product of 2 and 4.

Action: calculator_inequality
Action Input: {'text': '8 > 15'}
------------------------------
2025-04-23 15:58:38,871 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Calling tools: calculator_inequality
Tool's input: {"text": "8 > 15"}
Tool's response:
First number 8 is less than the second number 15
------------------------------
2025-04-23 15:58:39,978 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: Is the product of 2 * 4 greater than the current hour of the day?
Agent's thoughts:
Thought: I now know the final answer

Final Answer: No, the product of 2 * 4 (which is 8) is less than the current hour of the day (which is 15).
------------------------------
2025-04-23 15:58:39,981 - aiq.front_ends.console.console_front_end_plugin - INFO -
--------------------------------------------------
Workflow Result:
['No, the product of 2 * 4 (which is 8) is less than the current hour of the day (which is 15).']
```


## Deployment-Oriented Setup

For a production deployment, use Docker:

### Build the Docker Image

Prior to building the Docker image ensure that you have followed the steps in the [Installation and Setup](#installation-and-setup) section, and you are currently in the AIQ toolkit virtual environment.

From the root directory of the Simple Calculator repository, build the Docker image:

```bash
docker build --build-arg AIQ_VERSION=$(python -m setuptools_scm) -t simple_calculator -f examples/basic/functions/simple_calculator/Dockerfile .
```

### Run the Docker Container
Deploy the container:

```bash
docker run -p 8000:8000 -p 6006:6006 -e NVIDIA_API_KEY simple_calculator
```

Note, a phoenix telemetry service will be exposed at port 6006.

### Test the API
Use the following curl command to test the deployed API:

```bash
curl -X 'POST' \
  'http://localhost:8000/generate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"input_message": "Is the product of 2 * 4 greater than the current hour of the day?"}'
```

### Expected API Output
The API response should be similar to the following:

```bash
{
  "input": "Is the product of 2 * 4 greater than the current hour of the day?",
  "output": "No, the product of 2 * 4 (which is 8) is less than the current hour of the day (which is 16)."
}
```
