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
- [Installation and Setup](#installation-and-setup)
  - [Prerequisites](#prerequisites)
  - [Install Redis Dependencies](#install-redis-dependencies)
  - [Start Services](#start-services)
- [Run the Workflow](#run-the-workflow)

## Key Features

- **Redis Memory Backend Integration:** Demonstrates how to integrate Redis as a memory backend for NeMo Agent toolkit workflows, enabling persistent memory storage and retrieval across agent interactions.
- **Chat Memory Management:** Shows implementation of simple chat functionality with the ability to create, store, and recall memories using Redis as the underlying storage system.
- **Embeddings-Based Memory Search:** Uses embeddings models to create vector representations of queries and stored memories, implementing HNSW indexing with L2 distance metrics for efficient similarity search.

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install NeMo Agent toolkit.

### Prerequisites

- Docker and Docker Compose installed
- NeMo Agent toolkit with Redis dependencies

### Install Redis Dependencies

```bash
uv pip install -e '.[redis]'
```

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

```bash
aiq run --config_file=examples/memory/redis/configs/config.yml --input "my favorite flavor is strawberry"
```

**Expected Workflow Result**
```
The user's favorite flavor has been stored as strawberry.
```

### Recall Memory

```bash
aiq run --config_file=examples/memory/redis/configs/config.yml --input "what flavor of ice-cream should I get?"
```

**Expected Workflow Result**
```
You should get strawberry ice cream, as it is your favorite flavor.
```
