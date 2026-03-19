<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# MemMachine Memory Example

**Complexity:** 🟨 Intermediate

This example demonstrates how to use [MemMachine](https://docs.memmachine.ai/) as a long-term memory backend for NeMo Agent Toolkit agents. MemMachine provides unified episodic and semantic memory management backed by PostgreSQL (pgvector) and Neo4j.

## Table of Contents

- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
  - [Configure MemMachine](#configure-memmachine)
  - [Start Services](#start-services)
- [Run the Example](#run-the-example)

## Key Features

- **Episodic and Semantic Memory:** Stores memories from conversations and direct facts into both episodic and semantic memory layers, enabling rich retrieval by context.
- **ReAct Agent with Memory Tools:** Demonstrates a ReAct agent equipped with `get_memory` and `add_memory` tools that can recall and persist user preferences across interactions.
- **Docker-Based MemMachine Service:** MemMachine runs as a Docker service (PostgreSQL + Neo4j + MemMachine app), so no local server process is needed in the notebook.

## Prerequisites

- Docker installed and running — see the [Docker Installation Guide](https://docs.docker.com/engine/install/)
- An NVIDIA API key from [NVIDIA Build](https://build.nvidia.com/explore/discover)
- An OpenAI API key (or AWS Bedrock credentials) for MemMachine's embedding and language models

## Installation and Setup

If you have not already done so, follow the [Install Guide](../../../docs/source/get-started/installation.md#install-from-source) to create the development environment and install NeMo Agent Toolkit.

Install the required packages from the **repository root**:

```bash
uv pip install -e ".[langchain]"
uv pip install -e "packages/nvidia_nat_memmachine"
```

### Configure MemMachine

Edit `examples/memory/memmachine/configuration.yml` and replace the API key placeholders with your actual credentials:

- `<OPENAI_API_KEY>` — your OpenAI API key
- `<AWS_ACCESS_KEY_ID>` / `<AWS_SECRET_ACCESS_KEY>` — only needed if using AWS Bedrock models

The file is pre-configured to connect to the PostgreSQL and Neo4j containers using the default credentials in `docker-compose.memmachine.yml`. No changes to the database section are needed unless you override the defaults.

### Start Services

Start MemMachine (PostgreSQL, Neo4j, and the MemMachine app) from the **repository root**:

```bash
docker compose -f examples/deploy/docker-compose.memmachine.yml up -d
```

By default, the MemMachine API is available at `http://localhost:8095`. See `examples/deploy/README.md` for full start/stop instructions.

## Run the Example

Open and run the notebook from the **repository root**:

```bash
jupyter lab examples/memory/memmachine/memmachine_memory_example.ipynb
```

The notebook walks through:

1. Connecting to the running MemMachine Docker service
2. Adding memories from conversations and directly
3. Searching memories
4. Running a ReAct agent that uses memory tools to recall and store user preferences
