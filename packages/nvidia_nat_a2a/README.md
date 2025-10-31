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

# NVIDIA NAT A2A Integration

A2A Protocol integration for NeMo Agent Toolkit (NAT). This is a placeholder for evolving A2A client implementation.

## Overview

This package provides A2A (Agent-to-Agent) Protocol support for NAT, enabling:
- **A2A Client**: Connect to remote A2A agents and invoke their skills as NAT functions
- **Agent Discovery**: Fetch and parse Agent Cards to discover remote agent capabilities
- **Task Execution**: Submit tasks to remote agents with async execution and status polling
- **Authentication**: Integrate with NAT auth system for protected A2A agents

## Installation

```bash
# Install from source (during development)
cd packages/nvidia_nat_a2a
uv pip install -e .

# Or install as part of NAT with A2A support
uv pip install "nvidia-nat[a2a]"
```

## Quick Start

### Using A2A Client in NAT Workflow

```yaml
# config.yml
function_groups:
  research_agent:
    _type: a2a_client
    agent:
      url: https://research-agent.example.com
      task_timeout: 300

workflow:
  _type: react_agent
  llm_name: nim_llm
  tool_names: [research_agent.search_papers]
```

```bash
nat run --config_file config.yml --input "Find papers on quantum computing"
```

### Using CLI

```bash
# Discover agent capabilities
nat a2a client discover --url https://agent.example.com

# List available skills
nat a2a client skill list --url https://agent.example.com

# Submit a task
nat a2a client task submit search_papers \
  --url https://agent.example.com \
  --json-args '{"topic": "quantum computing"}'
```
