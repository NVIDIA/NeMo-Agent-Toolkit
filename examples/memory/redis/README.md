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

# Redis Examples

These examples use the redis memory backend.

## Table of Contents

- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
  - [Start Services](#start-services)
- [Run the Workflow](#run-the-workflow)
  - [Create Memory](#create-memory)
  - [Recall Memory](#recall-memory)

## Key Features

- **Redis Memory Backend Integration:** Demonstrates how to integrate Redis as a memory backend for NeMo Agent toolkit workflows, enabling persistent memory storage and retrieval across agent interactions.
- **Chat Memory Management:** Shows implementation of simple chat functionality with the ability to create, store, and recall memories using Redis as the underlying storage system.
- **Embeddings-Based Memory Search:** Uses embeddings models to create vector representations of queries and stored memories, implementing HNSW indexing with L2 distance metrics for efficient similarity search.

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

### Start Services

Run redis on `localhost:6379` and Redis Insight on `localhost:5540` with:

```bash
docker compose -f examples/deploy/docker-compose.redis.yml up
```

The examples are configured to use the Phoenix observability tool. Start phoenix on `localhost:6006` with:

```bash
docker compose -f examples/deploy/docker-compose.phoenix.yml up
```

## Run the Workflow

This examples shows how to have a simple chat that uses a redis memory backend for creating and retrieving memories.

An embeddings model is used to create embeddings for queries and for stored memories. Uses HNSW and L2 distance metric.

### Create Memory

Here we will add a memory for the workflow to use in following invocations. The memory tool will automatically determine the intent as to whether or not an input should be stored as a "fact" or if the input should be used to query the memory.

```bash
aiq run --config_file=examples/memory/redis/configs/config.yml --input "my favorite flavor is strawberry"
```

**Expected Workflow Output**
```console
<snipped for brevity>

Workflow Result:
['The user's favorite flavor has been stored as strawberry.']
```

### Recall Memory

Once we have established something in the memory, we can use the workflow to give us a response based on its input.

```bash
aiq run --config_file=examples/memory/redis/configs/config.yml --input "what flavor of ice-cream should I get?"
```

**Expected Workflow Output**
```console
<snipped for brevity>

Workflow Result:
['You should get strawberry ice cream, as it is your favorite flavor.']
```
