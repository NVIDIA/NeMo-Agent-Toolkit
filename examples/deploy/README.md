<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Supporting services for NeMo Agent Toolkit examples

This directory contains configurations for running services used by the examples in this repo.

## Table of Contents

- [Key Features](#key-features)
- [Available Services](#available-services)
- [Installation and Setup](#installation-and-setup)
  - [Prerequisites](#prerequisites)
  - [Running Services](#running-services)
  - [Stopping Services](#stopping-services)

## Key Features

- **Docker Compose Services:** Provides pre-configured Docker Compose files for essential services used across NeMo Agent toolkit examples.
- **Example Support Infrastructure:** Simplifies setup of supporting services required by various examples in the repository.
- **MinIO:** Includes `docker-compose.minio.yml` for running a MinIO server for object store examples.
- **MySQL:** Includes `docker-compose.mysql.yml` for running a MySQL server for object store examples.
- **Phoenix Observability:** Includes `docker-compose.phoenix.yml` for running Phoenix observability server to monitor and debug workflows.
- **Redis Service:** Includes `docker-compose.redis.yml` for running Redis memory backend with Redis Insight for memory-based examples.


## Available Services

- **`minio`**: `docker-compose.minio.yml`
- **`mysql`**: `docker-compose.mysql.yml`
- **`phoenix`**: `docker-compose.phoenix.yml`
- **`redis`**: `docker-compose.redis.yml`

## Installation and Setup

### Prerequisites

Ensure that Docker is installed and the Docker service is running before proceeding.

- Install Docker: Follow the official installation guide for your platform: [Docker Installation Guide](https://docs.docker.com/engine/install/)
- Start Docker Service:
  - Linux: Run`sudo systemctl start docker` (ensure your user has permission to run Docker).
  - Mac & Windows: Docker Desktop should be running in the background.
- Verify Docker Installation: Run the following command to verify that Docker is installed and running correctly:
```bash
docker info
```

### Running Services

To start MinIO (for object store examples):
```bash
docker compose -f examples/deploy/docker-compose.minio.yml up -d
```

To start MySQL (for object store examples):
```bash
docker compose -f examples/deploy/docker-compose.mysql.yml up -d
```

To start Phoenix (for observability examples):
```bash
docker compose -f examples/deploy/docker-compose.phoenix.yml up -d
```

To start Redis (required for memory and object store examples):
```bash
docker compose -f examples/deploy/docker-compose.redis.yml up -d
```

### Stopping Services

To stop the MinIO service:
```bash
docker compose -f examples/deploy/docker-compose.minio.yml down
```

To stop the MySQL service:
```bash
docker compose -f examples/deploy/docker-compose.mysql.yml down
```

To stop the Phoenix service:
```bash
docker compose -f examples/deploy/docker-compose.phoenix.yml down
```

To stop the Redis service:
```bash
docker compose -f examples/deploy/docker-compose.redis.yml down
```
