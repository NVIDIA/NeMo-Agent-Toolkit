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

# NVIDIA NeMo Agent Toolkit Memory Module

The NeMo Agent toolkit Memory subsystem is designed to store and retrieve a user's conversation history, preferences, and other long-term memory. This is especially useful for building stateful LLM-based applications that recall user-specific data or interactions across multiple sessions.

The memory module is designed to be extensible, allowing developers to create custom memory backends (providers in NeMo Agent toolkit terminology).

## Included Memory Providers

The NeMo Agent toolkit includes three memory providers, all of which are available as plugins:
* [Mem0](https://mem0.ai/) provided by the [`nvidia-nat-mem0ai`](https://pypi.org/project/nvidia-nat-mem0ai/) plugin
* [Redis](https://redis.io/) provided by the [`nvidia-nat-redis`](https://pypi.org/project/nvidia-nat-redis/) plugin
* [Zep](https://www.getzep.com/) provided by the [`nvidia-nat-zep-cloud`](https://pypi.org/project/nvidia-nat-zep-cloud/) plugin

Each provider formats memory differently to optimize LLM understanding:
- **Zep**: Returns pre-formatted memory with timestamps, structured facts, and knowledge graph insights
- **Mem0**: Formats results as numbered lists with categories
- **Redis**: Includes similarity scores and tags to help weight information appropriately

## Using Memory in Your Workflows

### Automatic Memory with AutoMemoryWrapper (Recommended)

The recommended approach for implementing agent memory is to use the `AutoMemoryWrapper`, which automatically captures and retrieves memory for any agent:

```yaml
llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.3-70b-instruct

memory:
  zep_memory:
    _type: nat.plugins.zep_cloud/zep_memory

functions:
  my_react_agent:
    _type: react_agent
    llm_name: nim_llm
    tool_names: []
    use_openai_api: true

workflow:
  _type: auto_memory_agent
  inner_agent_name: my_react_agent
  memory_name: zep_memory
  llm_name: nim_llm
  # retrieve_memory_for_every_response: true  # Automatic retrieval (default)
```

This approach:
- Captures every message between the user and agent automatically
- Retrieves relevant memory context for every user message
- Operates deterministically for reduced latency
- Follows best practices for memory management

### Memory Tools (Alternative Approach)

For use cases requiring explicit memory control, you can use memory tools:

```yaml
memory:
  user_memory:
    _type: mem0_memory

functions:
  add_memory:
    _type: add_memory
    memory: user_memory
    description: Add facts about user preferences to long-term memory.
  
  get_memory:
    _type: get_memory
    memory: user_memory
    description: Retrieve user preferences from long-term memory.

workflow:
  _type: react_agent
  tool_names:
    - add_memory
    - get_memory
  llm: nim_llm
```

You can also combine both approaches by disabling automatic retrieval in `AutoMemoryWrapper` while keeping automatic capture:

```yaml
functions:
  get_memory:
    _type: get_memory
    memory: user_memory
    description: Retrieve relevant memories from long-term memory.
  
  my_react_agent:
    _type: react_agent
    llm_name: nim_llm
    tool_names: [get_memory]
    use_openai_api: true

workflow:
  _type: auto_memory_agent
  inner_agent_name: my_react_agent
  memory_name: user_memory
  llm_name: nim_llm
  retrieve_memory_for_every_response: false  # Disable automatic retrieval
```

## User Identity and Multi-Tenant Isolation

The NeMo Agent toolkit memory system automatically handles user identity through the `Context` object:

- **Production deployments**: User identity is set through HTTP headers (`user-id`, `user-first-name`, `user-last-name`, `user-email`) on incoming requests
- **Automatic isolation**: Memory operations are automatically scoped to the current user
- **Security**: User identity is managed by the system, not the agent, preventing cross-user memory access

Additional user attributes (`user_first_name`, `user_last_name`, `user_email`) enable memory providers such as Zep to build richer user profiles and knowledge graphs.

## Examples

The following examples demonstrate how to use the memory module in the NeMo Agent toolkit:
* `examples/memory/redis` - Memory tools with Redis
* `examples/memory/zep` - Memory tools and AutoMemoryWrapper with Zep
* `examples/control_flow/auto_memory_wrapper` - AutoMemoryWrapper examples with different providers
* `examples/frameworks/semantic_kernel_demo` - Memory integration with Semantic Kernel
* `examples/RAG/simple_rag` - Memory with RAG workflows

## Additional Resources

For information on how to create a custom memory provider, see the [Adding a Memory Provider](../extend/memory.md) document.
