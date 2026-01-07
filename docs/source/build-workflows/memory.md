<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Memory in NVIDIA NeMo Agent Toolkit

The NeMo Agent toolkit Memory subsystem is designed to store and retrieve a user's conversation history, preferences, and other "long-term memory." This is especially useful for building stateful [LLM-based](./llms/index.md) applications that recall user-specific data or interactions across multiple steps.

The memory module is designed to be extensible, allowing developers to create custom memory back-ends, providers in NeMo Agent toolkit terminology.

## Included Memory Modules
The NeMo Agent toolkit includes three memory module providers, all of which are available as plugins:
* [Mem0](https://mem0.ai/) which is provided by the [`nvidia-nat-mem0ai`](https://pypi.org/project/nvidia-nat-mem0ai/) plugin.
* [Redis](https://redis.io/) which is provided by the [`nvidia-nat-redis`](https://pypi.org/project/nvidia-nat-redis/) plugin.
* [Zep](https://www.getzep.com/) which is provided by the [`nvidia-nat-zep-cloud`](https://pypi.org/project/nvidia-nat-zep-cloud/) plugin ([Zep NVIDIA NeMo documentation](https://help.getzep.com/nvidia-nemo)).

## Automatic Memory Wrapper Agent

The NeMo Agent toolkit provides an [`auto_memory_agent`](../components/agents/auto-memory-wrapper/index.md) wrapper that adds automatic memory capture and retrieval to any agent without requiring the LLM to invoke memory tools explicitly.

### Why Use Automatic Memory?

**Traditional tool-based memory:**
- LLMs may forget to call memory tools
- Memory capture is inconsistent
- Requires explicit memory tool configuration

**Automatic memory wrapper agent:**
- **Guaranteed capture**: User messages and agent responses are automatically stored
- **Automatic retrieval**: Relevant context is injected before each agent call
- **Memory backend agnostic**: Works with Zep, Mem0, Redis, or any `MemoryEditor`
- **Universal compatibility**: Wraps any agent type (ReAct, ReWOO, Tool Calling, etc.)

### Quick Start

To use automatic memory, wrap any agent with the `auto_memory_agent` workflow type:

```yaml
memory:
  zep_memory:
    _type: nat.plugins.zep_cloud/zep_memory

functions:
  my_react_agent:
    _type: react_agent
    llm_name: nim_llm
    tool_names: [calculator]

workflow:
  _type: auto_memory_agent
  inner_agent_name: my_react_agent
  memory_name: zep_memory
  llm_name: nim_llm
```

### Configuration Options

The automatic memory wrapper agent supports several configuration parameters:

**Required Parameters:**
- `inner_agent_name`: Name of the agent to wrap with automatic memory
- `memory_name`: Name of the memory backend (from `memory:` section)
- `llm_name`: LLM to use (required by `AgentBaseConfig`)

**Optional Feature Flags** (all default to `true`):
- `save_user_messages_to_memory`: Automatically save user messages before agent processing
- `retrieve_memory_for_every_response`: Automatically retrieve and inject memory context
- `save_ai_messages_to_memory`: Automatically save agent responses after generation

**Memory Backend Parameters:**
- `search_params`: Passed to `memory_editor.search()` (e.g., `mode`, `top_k`)
- `add_params`: Passed to `memory_editor.add_items()` (e.g., `ignore_roles`)

### Multi-Tenant Memory Isolation

User ID is automatically extracted at runtime for memory isolation via:
1. `user_manager.get_id()` - For production with custom auth middleware (recommended)
2. `X-User-ID` HTTP header - For testing without middleware
3. `"default_user"` - Fallback for local development

For detailed configuration and usage examples, refer to the `examples/agents/auto_memory_wrapper/README.md` guide.

## Examples
The following examples in the [repository](https://github.com/NVIDIA/NeMo-Agent-Toolkit) demonstrate how to use the memory module in the NeMo Agent toolkit:
* `examples/agents/auto_memory_wrapper` - Automatic memory wrapper agent for any agent
* `examples/memory/redis` - Basic long-term memory using Redis
* `examples/frameworks/semantic_kernel_demo` - Multi-agent system with long-term memory
* `examples/RAG/simple_rag` - RAG system with Mem0 memory

## Additional Resources
For information on how to write a new memory module provider can be found in the [Adding a Memory Provider](../extend/custom-components/memory.md) document.
