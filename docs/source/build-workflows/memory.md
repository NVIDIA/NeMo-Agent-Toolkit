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

The NeMo Agent Toolkit Memory subsystem is designed to store and retrieve a user's conversation history, preferences, and other "long-term memory." This is especially useful for building stateful [LLM-based](./llms/index.md) applications that recall user-specific data or interactions across multiple steps.

The memory module is designed to be extensible, allowing developers to create custom memory back-ends, providers in NeMo Agent Toolkit terminology.

## User Identity for Memory Tools

The built-in `add_memory`, `get_memory`, and `delete_memory` tools bind every operation to an identity. Set an optional `user_id` in the tool configuration for a fixed service or single-user identity.

```yaml
functions:
  get_memory:
    _type: get_memory
    memory: user_memory
    user_id: service_user
```

## Included Memory Modules
The NeMo Agent Toolkit includes four memory module providers, all of which are available as plugins:
* [Mem0](https://mem0.ai/), provided by the [`nvidia-nat-mem0ai`](https://pypi.org/project/nvidia-nat-mem0ai/) plugin.
* [MemMachine](https://memmachine.ai/), provided by the [`nvidia-nat-memmachine`](https://pypi.org/project/nvidia-nat-memmachine/) plugin (**Experimental; not recommended for production use**).
* [Redis](https://redis.io/), provided by the Redis-maintained [`nemo-agent-toolkit-redis`](https://pypi.org/project/nemo-agent-toolkit-redis/) plugin.
* [Zep](https://www.getzep.com/), provided by the [`nvidia-nat-zep-cloud`](https://pypi.org/project/nvidia-nat-zep-cloud/) plugin ([Zep NVIDIA NeMo documentation](https://help.getzep.com/nvidia-nemo)).

## Third-Party Memory Plugins
Additional memory backends are available as community plugins:
* [Synap](https://maximem.ai) — managed memory layer with user and customer scoping, provided by the [`maximem-synap-nemo-agent-toolkit`](https://pypi.org/project/maximem-synap-nemo-agent-toolkit/) plugin. See `examples/memory/synap/` for usage. ([Open source integration package](https://github.com/maximem-ai/maximem_synap_sdk/tree/main/packages/integrations))

## Authenticating Memory Tool Users

Each of the built-in `add_memory`, `get_memory`, and `delete_memory` tools require configuring an identity source:

- Use `user_id` for a fixed, single-user memory namespace.
- Use `user_id_resolver` for a multi-user application. Its value is the import path of a trusted, zero-argument Python callable that returns the current authenticated user's stable ID. The callable can be synchronous or asynchronous and is invoked for every memory operation.

For example, application code can obtain a user that authentication middleware has already verified:

```python
from my_application.request_context import get_authenticated_user


def resolve_memory_user_id() -> str:
    user = get_authenticated_user()
    if user is None:
        raise RuntimeError("An authenticated user is required")
    return user.user_id
```

Reference that callable from each memory tool:

```yaml
functions:
  add_memory:
    _type: add_memory
    memory: user_memory
    user_id_resolver: my_application.auth.resolve_memory_user_id
  get_memory:
    _type: get_memory
    memory: user_memory
    user_id_resolver: my_application.auth.resolve_memory_user_id
  delete_memory:
    _type: delete_memory
    memory: user_memory
    user_id_resolver: my_application.auth.resolve_memory_user_id
```

## Automatic Memory Wrapper Agent

The NeMo Agent Toolkit provides an [`auto_memory_agent`](../components/agents/auto-memory-wrapper/index.md) wrapper that adds automatic memory capture and retrieval to any agent without requiring the LLM to invoke memory tools explicitly.

### Why Use Automatic Memory?

**Traditional tool-based memory:**
- LLMs may forget to call memory tools
- Memory capture is inconsistent
- Requires explicit memory tool configuration

**Automatic memory wrapper agent:**
- **Guaranteed capture**: User messages and agent responses are automatically stored
- **Automatic retrieval**: Relevant context is injected before each agent call
- **Memory backend agnostic**: Works with Zep, Mem0, MemMachine, Redis, or any `MemoryEditor`
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

The automatic memory wrapper reads only the identity resolved by the runtime session. Resolve identity through
authenticated front-end credentials, `SessionManager.session(user_id=...)`, or the console front end `user_id` (which
defaults to `"nat_run_user_id"` for `nat run`). Memory operations fail closed when no identity is available.

For local testing or an isolated deployment behind an authenticating reverse proxy, you can explicitly opt in to a
trusted upstream identity header:

```yaml
general:
  front_end:
    _type: fastapi
    identity_header: X-User-ID
```

Do not enable this setting unless clients cannot reach `nat serve` directly, the proxy authenticates every request and
overwrites the header, the toolkit port is not published outside the trusted backend network, and every container on
that network is trusted. A client-supplied identity header is not authentication.

Conversation-aware memory backends can also use `conversation_id` to isolate separate conversations for the same user.
For `nat run`, pass `--conversation_id` when testing independent memory conversations from the CLI.

When configured, the header is authoritative for HTTP and WebSocket requests. Missing, empty, and repeated values are
rejected, and other credentials cannot override it.

For detailed configuration and usage examples, refer to the `examples/agents/auto_memory_wrapper/README.md` guide.

## Examples
The following examples in the [repository](https://github.com/NVIDIA/NeMo-Agent-Toolkit) demonstrate how to use the memory module in the NeMo Agent Toolkit:
* `examples/agents/auto_memory_wrapper` - Automatic memory wrapper agent for any agent
* `examples/memory/memmachine` - MemMachine server setup and example notebook
* `examples/memory/redis` - Basic long-term memory using Redis
* `examples/frameworks/semantic_kernel_demo` - Multi-agent system with long-term memory
* `examples/RAG/simple_rag` - RAG system with Mem0 memory

## Additional Resources
For information on how to write a new memory module provider can be found in the [Adding a Memory Provider](../extend/custom-components/memory.md) document.
