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

# NVIDIA NeMo Agent Toolkit - MemMachine Integration

This package provides integration with MemMachine for memory management in NeMo Agent toolkit.

## Overview

MemMachine is a unified memory management system that supports both episodic and semantic memory through a single interface. This integration allows you to use MemMachine as a memory backend for your NeMo Agent toolkit workflows.

## Prerequisites

- Python 3.11+
- **memmachine-client** is installed automatically with this package. You need a running MemMachine instance (local or hosted) to connect to. To run a local instance, install and configure **memmachine-server** separately (see [MemMachine Server Setup](#memmachine-server-setup) below).

## Installation

Install the package:

```bash
pip install nvidia-nat-memmachine
```

Or for development:

```bash
uv pip install -e packages/nvidia_nat_memmachine
```

## MemMachine Server Setup

This section is optional. Only follow these steps if you want to run a **local** MemMachine instance. If you use a hosted MemMachine instance, configure `base_url` (and any auth) in your workflow config and skip this section.

### Step 1: Configure MemMachine

Before starting the server, edit `examples/memory/memmachine/configuration.yml` and replace the `<OPENAI_API_KEY>` (or AWS) placeholders with your actual API keys.

### Step 2: Start the MemMachine Server

Start MemMachine (along with its PostgreSQL and Neo4j dependencies) using Docker Compose:

```bash
docker compose -f examples/deploy/docker-compose.memmachine.yml up -d
```

This starts:
- **PostgreSQL** — vector and relational storage
- **Neo4j** — graph memory backend
- **MemMachine** — the memory server, exposed on `http://localhost:8095`

Ensure Docker is installed and running before executing this command. See the [Docker Installation Guide](https://docs.docker.com/engine/install/) if needed.

To stop the server:

```bash
docker compose -f examples/deploy/docker-compose.memmachine.yml down
```

For more details, see the [MemMachine Documentation](https://docs.memmachine.ai/).

## Usage in NeMo Agent toolkit

Add MemMachine memory to your workflow configuration:

```yaml
memory:
  memmachine_memory:
    base_url: "http://localhost:8095"  # MemMachine server URL
    org_id: "my_org"  # Optional: default organization ID
    project_id: "my_project"  # Optional: default project ID
```

## Additional Resources

- [Example Notebook](../../examples/memory/memmachine/memmachine_memory_example.ipynb)
- [MemMachine Documentation](https://docs.memmachine.ai/)
- [NeMo Agent toolkit Documentation](https://docs.nvidia.com/nemo/agent-toolkit/latest/)

