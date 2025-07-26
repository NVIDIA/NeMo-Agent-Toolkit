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

This example demonstrates an end-to-end (E2E) agentic workflow using the NeMo Agent toolkit library, fully configured through a YAML file. It showcases the NeMo Agent toolkit plugin system and `Builder` to seamlessly integrate pre-built and custom tools into workflows.

## Table of Contents

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

- **Custom Calculator Tools:** Demonstrates five mathematical tools - `calculator_multiply`, `calculator_inequality`, `calculator_divide`, `calculator_subtract`, and `current_datetime` for mathematical operations and time-based comparisons.
- **ReAct Agent Integration:** Uses a `react_agent` that performs reasoning between tool calls to solve complex mathematical queries requiring multiple steps.
- **Multi-step Problem Solving:** Shows how an agent can break down complex questions like "Is the product of 2 * 4 greater than the current hour?" into sequential tool calls.
- **Custom Function Registration:** Demonstrates the NeMo Agent toolkit plugin system for registering custom mathematical functions with proper validation and error handling.
- **YAML-based Configuration:** Fully configurable workflow that showcases how to orchestrate multiple tools through simple configuration.

---

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install NeMo Agent toolkit.

### Install this Workflow:

From the root directory of the NeMo Agent toolkit library, run the following commands:

```bash
uv pip install -e examples/getting_started/simple_calculator
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
export OPENAI_API_KEY=<YOUR_API_KEY>  # OPTIONAL
```

### Run the Workflow

Return to your original terminal, and run the following command from the root of the NeMo Agent toolkit repo to execute this workflow with the specified input:

```bash
aiq run --config_file examples/getting_started/simple_calculator/configs/config.yml --input "Is the product of 2 * 4 greater than the current hour of the day?"
```

**Expected Workflow Output**
Note that the output is subject to the time of day when the workflow was run. For this example output, it was run in the afternoon.
```
No, the product of 2 * 4 (which is 8) is less than the current hour of the day (which is 15).
```


## Deployment-Oriented Setup

For a production deployment, use Docker:

### Build the Docker Image

Prior to building the Docker image ensure that you have followed the steps in the [Installation and Setup](#installation-and-setup) section, and you are currently in the NeMo Agent toolkit virtual environment.

From the root directory of the Simple Calculator repository, build the Docker image:

```bash
docker build --build-arg AIQ_VERSION=$(python -m setuptools_scm) -t simple_calculator -f examples/getting_started/simple_calculator/Dockerfile .
```

### Run the Docker Container
Deploy the container:

```bash
docker run -p 8000:8000 -p 6006:6006 -e NVIDIA_API_KEY -e OPENAI_API_KEY simple_calculator
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
  "value": "No, the product of 2 * 4 (which is 8) is less than the current hour of the day (which is 16)."
}
```
