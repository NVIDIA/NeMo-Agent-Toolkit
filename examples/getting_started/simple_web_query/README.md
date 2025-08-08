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

A minimal example demonstrating a simple LangSmith-Documentation agent. This agent leverages the NeMo Agent toolkit plugin system and `Builder` to integrate pre-built and custom tools into the workflow to answer questions about LangSmith. Key elements are summarized below:

## Table of Contents

* [Key Features](#key-features)
* [Prerequisites](#prerequisites)
* [Installation and Setup](#installation-and-setup)
* [Running the Workflow](#running-the-workflow)
* [Deployment-Oriented Setup](#docker-quickstart)

---

## Key Features

- **Webpage Query Tool:** Demonstrates a `webpage_query` tool that retrieves and processes documentation from LangSmith's website (https://docs.smith.langchain.com) using web scraping and vector search.
- **ReAct Agent Integration:** Uses a `react_agent` that reasons about user queries and determines when to retrieve relevant documentation from the web.
- **Document Retrieval and Embedding:** Shows how to automatically generate embeddings from web content and perform semantic search to answer questions about LangSmith.
- **End-to-End Web RAG:** Complete example of Retrieval-Augmented Generation (RAG) using web-scraped content as the knowledge source.
- **YAML-based Configuration:** Fully configurable workflow demonstrating integration of web scraping, embeddings, and agent reasoning through simple configuration.

## Prerequisites

Ensure that Docker is installed and the Docker service is running before proceeding.

- Install Docker: Follow the official installation guide for your platform: [Docker Installation Guide](https://docs.docker.com/engine/install/)
- Start Docker Service:
  - Linux: Run`sudo systemctl start docker` (ensure your user has permission to run Docker).
  - Mac & Windows: Docker Desktop should be running in the background.
- Verify Docker Installation: Run the following command to verify that Docker is installed and running correctly:
  ```bash
  docker info
  ```

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install NeMo Agent toolkit.

### Install this Workflow:

From the root directory of the NeMo Agent toolkit library, run the following commands:

```bash
uv pip install -e examples/getting_started/simple_web_query
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

## Running the Workflow

Run the following command from the root of the NeMo Agent toolkit repo to execute this workflow with the specified input:

```bash
aiq run --config_file examples/getting_started/simple_web_query/configs/config.yml --input "What is LangSmith?"
```

**Expected Workflow Output**
```console
<snipped for brevity>

Workflow Result:
['LangSmith is a platform for building production-grade LLM (Large Language Model) applications, allowing users to monitor and evaluate their applications, and providing features such as observability, evaluation, and prompt engineering. It is framework-agnostic and can be used with or without LangChain's open source frameworks.']
```

## Docker Quickstart

Prior to building the Docker image ensure that you have followed the steps in the [Installation and Setup](#installation-and-setup) section, and you are currently in the NeMo Agent toolkit virtual environment.

Set your NVIDIA API Key in the `NVIDIA_API_KEY` environment variable.

```bash
export NVIDIA_API_KEY="your_nvidia_api_key"
```

From the git repository root, run the following command to build NeMo Agent toolkit and the simple agent into a Docker image.

```bash
docker build --build-arg AIQ_VERSION=$(python -m setuptools_scm) -f examples/getting_started/simple_web_query/Dockerfile -t simple-web-query-agent .
```

Then, run the following command to run the simple agent.

```bash
docker run -p 8000:8000 -e NVIDIA_API_KEY simple-web-query-agent
```

After the container starts, you can access the agent at http://localhost:8000.

```bash
curl -X 'POST' \
  'http://localhost:8000/generate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"input_message": "What is LangSmith?"}'
```
