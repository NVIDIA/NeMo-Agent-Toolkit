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

## How It Works

```
User Message
     ↓
[Capture Node]  ← Automatically stores user message
     ↓
[Retrieve Node] ← Fetches relevant context from memory
     ↓
[Inner Agent]   ← Your agent (ReAct, ReWOO, etc.) with injected context
     ↓
[Response Node] ← Automatically stores agent response
     ↓
Agent Response
```

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
    tool_names: [calculator]  # No memory tools needed!

workflow:
  _type: auto_memory_agent
  inner_agent_name: my_react_agent
  memory_name: zep_memory
  llm_name: nim_llm
```

### 2. Custom Search Parameters (Zep Advanced)

```yaml
workflow:
  _type: auto_memory_agent
  inner_agent_name: my_agent
  memory_name: zep_memory
  llm_name: nim_llm
  search_params:
    mode: "advanced"  # Zep-specific: use advanced search
    top_k: 10         # Return up to 10 context items
```

### 3. Save-Only Mode

```yaml
workflow:
  _type: auto_memory_agent
  inner_agent_name: my_agent
  memory_name: zep_memory
  llm_name: nim_llm
  retrieve_memory_for_every_response: false  # Only save, no automatic retrieval
```

Use this when you want automatic saving but prefer tool-based or manual retrieval.

## Configuration Options

### Feature Flags (all default to `true`)
- `save_user_messages_to_memory`: Automatically save user messages to memory
- `retrieve_memory_for_every_response`: Automatically retrieve memory context before agent
- `save_ai_messages_to_memory`: Automatically save AI agent responses to memory

### Search Parameters (`search_params`)
Backend-specific parameters passed to `memory_editor.retrieve_memory()`:

**Zep:**
- `mode`: "basic" (fast, P95 < 200ms) or "summary" (LLM-summarized context)
- `top_k`: Maximum context items to include

**Mem0:**
- `threshold`: Minimum relevance score (0.0-1.0)
- `rerank`: Enable reranking (bool)
- `top_k`: Maximum results (default: 5)

**Redis:**
- `top_k`: Maximum results to return from vector search
- `score_threshold`: Minimum similarity score

**Any backend can define custom parameters via `**kwargs`**

### Add Parameters (`add_params`)
Backend-specific parameters passed to `memory_editor.add_items()`:

**Zep:**
- `ignore_roles`: List of role types to exclude from graph memory (e.g., `["assistant"]`)
  Available roles: `norole`, `system`, `assistant`, `user`, `function`, `tool`

**Other backends can define custom parameters via `**kwargs`**

## Examples

See the `configs/` directory:
- `config_zep.yml` - Basic usage with Zep
- `config_zep_advanced.yml` - Zep with custom search parameters

## How to Run

```bash
# Set environment variables
export ZEP_API_KEY="your_api_key"
export ZEP_PROJECT_ID="your_project_id"

# Run the example
nat run --config examples/control_flow/auto_memory_wrapper/configs/config_zep.yml
```

## Comparison: Tool-Based vs Automatic

| Feature | Tool-Based Memory | Automatic Memory |
|---------|------------------|------------------|
| Reliability | ❌ Depends on LLM | ✅ Guaranteed |
| Setup | ❌ Add tools to config | ✅ Wrap any agent |
| LLM Overhead | ❌ Extra tool calls | ✅ No extra calls |
| Consistency | ❌ May miss memories | ✅ Saves everything |
| Selective Retrieval | ✅ LLM decides | ❌ Always retrieves* |

*Can use `retrieve_memory_for_every_response: false` for save-only mode

## When to Use Each Approach

**Use Automatic Memory (`auto_memory_agent`):**
- Production apps requiring consistent memory
- Memory backends designed for automatic operation (Zep, Mem0)
- When LLM reliability is a concern
- Multi-turn conversations requiring context continuity

**Use Tool-Based Memory:**
- Selective memory access based on context
- LLM-driven memory decisions
- Complex memory operations beyond add/retrieve

## Advanced: Hybrid Mode

Combine automatic saving with tool-based retrieval:

```yaml
functions:
  get_memory:
    _type: get_memory_tool
    memory_name: zep_memory

  my_agent:
    _type: react_agent
    tool_names: [calculator, get_memory]  # LLM can retrieve via tool

workflow:
  _type: auto_memory_agent
  inner_agent_name: my_agent
  memory_name: zep_memory
  retrieve_memory_for_every_response: false  # Disable automatic retrieval
  # Saving is still automatic!
```

This gives you:
- ✅ Guaranteed saving (automatic)
- ✅ Selective retrieval (LLM-driven via tools)
