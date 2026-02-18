# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

### Step 1: Install MemMachine Server (local instance only)

```bash
pip install memmachine-server
```

### Step 2: Run the Configuration Wizard

Before starting the server, you need to configure MemMachine using the interactive configuration wizard:

```bash
memmachine-configure
```

The wizard will guide you through setting up:

- **Neo4j Database**: Option to install Neo4j automatically or provide connection details for an existing instance. If you enter nothing, Neo4j is installed on your local disk by default.
- **Large Language Model (LLM) Provider**: Choose from supported providers like OpenAI, AWS Bedrock, or Ollama
- **Model Selection**: Select specific LLM and embedding models. The default is OpenAI with `gpt-4o-mini` and `text-embedding-3-small`.
- **API Keys and Credentials**: Input necessary API keys for your selected LLM provider
- **Server Settings**: Configure server host and port. The default is `localhost:8080` but it is recommended to bind to port `8095` as `8080` is a commonly used port.

**Note**: 
- The wizard installs Neo4j and Java automatically when you choose to install Neo4j (platform-specific: Windows uses ZIP, macOS uses brew, Linux uses tar.gz)
- The wizard uses Neo4j as the vector database and SQLite as the relational database by default
- The configuration file will be generated at `<HOME>/.config/memmachine/cfg.yml`

### Step 3: Start the MemMachine Server

After completing the configuration wizard, start the server:

```bash
memmachine-server
```

The server will start on `http://localhost:8080` by default (or the port you configured with the configuration wizard).

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

- [MemMachine Documentation](https://docs.memmachine.ai/)
- [NeMo Agent toolkit Documentation](https://docs.nvidia.com/nemo/agent-toolkit/latest/)

