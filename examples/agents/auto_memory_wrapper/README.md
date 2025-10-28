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

# Automatic Memory Wrapper for NAT Agents

The `auto_memory_agent` wraps any NAT agent to provide **automatic memory capture and retrieval** without requiring the LLM to invoke memory tools.

## Why Use This?

**Problem with tool-based memory:**
- LLMs may forget to call memory tools
- Memory capture is inconsistent
- Requires adding memory tools to every agent configuration

**Solution with automatic memory:**
- **Guaranteed capture**: Every user message and agent response is automatically stored
- **Automatic retrieval**: Relevant context is retrieved before every agent call
- **Memory-agnostic**: Works with Zep, Mem0, Redis, or any `MemoryEditor` implementation
- **Universal compatibility**: Wraps ANY agent type (ReAct, ReWOO, Tool Calling, custom)

## Quick Start

### 1. Basic Usage (Zep)

```yaml
memory:
  zep_memory:
    _type: nat.plugins.zep_cloud/zep_memory

functions:
  my_react_agent:
    _type: react_agent
    llm_name: nim_llm
    tool_names: [calculator]
    use_openai_api: true  # REQUIRED

workflow:
  _type: auto_memory_agent
  inner_agent_name: my_react_agent
  memory_name: zep_memory
  llm_name: nim_llm
  user_id: "user123"  # Configure user ID
```

### 2. Custom Search Parameters (Zep Advanced)

```yaml
workflow:
  _type: auto_memory_agent
  inner_agent_name: my_agent
  memory_name: zep_memory
  llm_name: nim_llm
  user_id: "user123"
  search_params:
    mode: "summary"  # Zep-specific: use summary mode
    top_k: 10
  add_params:
    ignore_roles: ["assistant"]  # Don't add assistant messages to Zep graph
```

## Configuration Options

### Required Fields
- `inner_agent_name`: Name of the agent to wrap
- `memory_name`: Name of the memory backend
- `user_id`: User ID for memory isolation

### Feature Flags (all default to `true`)
- `save_user_messages_to_memory`: Automatically save user messages
- `retrieve_memory_for_every_response`: Automatically retrieve memory context
- `save_ai_messages_to_memory`: Automatically save AI responses

### Backend-Specific Parameters

**`search_params`** (passed to `memory_editor.search()`):
- **Zep**: `mode` ("basic" or "summary"), `top_k`
- **Mem0**: `threshold`, `rerank`, `top_k`
- **Redis**: `top_k`, `score_threshold`

**`add_params`** (passed to `memory_editor.add_items()`):
- **Zep**: `ignore_roles` (list of roles to exclude from graph: `["assistant"]`)

## How to Run

```bash
# Set environment variables
export ZEP_API_KEY="your_api_key"
export ZEP_PROJECT_ID="your_project_id"

# Run the example
nat run --config examples/agents/auto_memory_wrapper/configs/config_zep.yml
```

## Examples

See the `configs/` directory:
- `config_zep.yml` - Basic usage with Zep
- `config_zep_advanced.yml` - Zep with custom search parameters and ignore_roles
