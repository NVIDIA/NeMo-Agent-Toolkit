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

# Automatic Memory Wrapper Agent

The `auto_memory_agent` wraps any NAT agent to provide **automatic memory capture and retrieval** without requiring the LLM to invoke memory tools explicitly. This wrapper ensures consistent memory management across conversations while maintaining full compatibility with any agent type.

---

## Features

- **Guaranteed Memory Capture**: User messages and agent responses are automatically stored
- **Automatic Memory Retrieval**: Relevant context is injected before each agent call
- **Universal Compatibility**: Wraps any agent type (ReAct, ReWOO, Tool Calling, Reasoning, etc.)
- **Memory Backend Agnostic**: Works with Zep, Mem0, Redis, or any `MemoryEditor` implementation
- **Multi-Tenant Support**: Automatic user isolation via runtime user ID extraction
- **Configurable Behavior**: Fine-grained control over memory capture and retrieval

---

## Why Use Automatic Memory?

**Traditional tool-based memory** relies on the LLM to explicitly call memory tools:
- LLMs may forget to call memory tools
- Memory capture is inconsistent across interactions
- Requires careful prompt engineering to ensure memory tool usage
- Additional tokens consumed for memory tool invocation

**Automatic memory wrapper agent** handles memory transparently:
- Memory operations are guaranteed to occur on every interaction
- No additional prompt engineering needed
- Consistent memory behavior across all agent types
- Reduced cognitive load on the LLM

---

## Requirements

The automatic memory wrapper agent works with any memory backend that implements the `MemoryEditor` interface. The following memory plugins are available:

- [`nvidia-nat-zep-cloud`](https://pypi.org/project/nvidia-nat-zep-cloud/) - Zep Cloud memory backend
- [`nvidia-nat-mem0ai`](https://pypi.org/project/nvidia-nat-mem0ai/) - Mem0 memory backend
- [`nvidia-nat-redis`](https://pypi.org/project/nvidia-nat-redis/) - Redis memory backend

---

## Configuration

The automatic memory wrapper agent may be utilized as a workflow or a function.

### Example `config.yml`

To use the automatic memory wrapper agent as a workflow:

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

To use the automatic memory wrapper agent as a function:

```yaml
memory:
  zep_memory:
    _type: nat.plugins.zep_cloud/zep_memory

functions:
  my_agent_with_memory:
    _type: auto_memory_agent
    inner_agent_name: my_react_agent
    memory_name: zep_memory
    llm_name: nim_llm
    description: 'A ReAct agent with automatic memory'
```

### Configurable Options

**Required Parameters:**

| Parameter | Description |
|-----------|-------------|
| `inner_agent_name` | Name of the agent to wrap with automatic memory |
| `memory_name` | Name of the memory backend (from `memory:` section) |
| `llm_name` | LLM to use (required by `AgentBaseConfig`) |

**Optional Feature Flags** (all default to `true`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `save_user_messages_to_memory` | `true` | Automatically save user messages before agent processing |
| `retrieve_memory_for_every_response` | `true` | Automatically retrieve and inject memory context |
| `save_ai_messages_to_memory` | `true` | Automatically save agent responses after generation |

**Memory Backend Parameters:**

- `search_params`: Passed to `memory_editor.search()` for memory retrieval configuration
  ```yaml
  search_params:
    mode: "summary"  # Zep: "basic" or "summary"
    top_k: 10        # Maximum memories to retrieve
  ```

- `add_params`: Passed to `memory_editor.add_items()` for memory storage configuration
  ```yaml
  add_params:
    ignore_roles: ["assistant"]  # Zep: Exclude roles from graph memory
  ```

**Other Options:**

- `description`: Defaults to `"Auto Memory Agent Wrapper"`. When configured as a function, this allows control over the tool description.

- `verbose`: Defaults to `False` (useful to prevent logging of sensitive data). If set to `True`, the wrapper will log memory operations and intermediate steps.

---

## How the Automatic Memory Wrapper Agent Works

The automatic memory wrapper agent intercepts agent invocations and automatically handles memory operations:

### Step-by-Step Execution Flow

1. **User Message Reception** – The wrapper receives the user's input message
2. **Memory Retrieval** (if `retrieve_memory_for_every_response` is `true`)
   - Searches the memory backend for relevant context
   - Injects retrieved memories into the agent's context
3. **User Message Storage** (if `save_user_messages_to_memory` is `true`)
   - Stores the user's message in the memory backend
4. **Agent Invocation** – The wrapped agent processes the request with memory context
5. **Response Storage** (if `save_ai_messages_to_memory` is `true`)
   - Stores the agent's response in the memory backend
6. **Response Return** – Returns the agent's response to the user

### Example Walkthrough

Consider a conversation with automatic memory enabled:

**First Interaction:**
```
User: "My name is Alice and I prefer Python for data analysis."
Agent: "Nice to meet you, Alice! I'll remember your preference for Python."
```

The wrapper automatically:
- Stores the user message
- Invokes the inner agent
- Stores the agent response

**Later Interaction:**
```
User: "What programming language should I use for my data project?"
Agent: "Based on what you told me earlier, I recommend Python for your data analysis project since that's your preferred language."
```

The wrapper automatically:
- Retrieves relevant memories (Alice's name and Python preference)
- Injects them into the agent's context
- Agent can reference past conversations naturally

---

## Multi-Tenant Memory Isolation

The automatic memory wrapper agent provides multi-tenant support through automatic user ID extraction. User ID is **NOT** configured in YAML but extracted at runtime.

### User ID Extraction Priority

1. **`user_manager.get_id()`** - For production with custom auth middleware (recommended)
2. **`X-User-ID` HTTP header** - For testing without middleware
3. **`"default_user"`** - Fallback for local development

### Production: Custom Middleware

Create middleware that extracts user ID from your authentication system:

```python
from nat.runtime.session import SessionManager

class AuthenticatedUserManager:
    def __init__(self, user_id: str):
        self._user_id = user_id

    def get_id(self) -> str:
        return self._user_id

# In your request handler
async def handle_request(request):
    # Extract from JWT, OAuth, API key, etc.
    user_id = extract_user_from_jwt(request.headers["authorization"])

    async with session_manager.session(
        user_manager=AuthenticatedUserManager(user_id=user_id),
        http_connection=request,
    ) as session:
        result = await session.run(user_input)
    return result
```

### Testing: X-User-ID Header

For quick testing without custom middleware:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-User-ID: test_user_123" \
  -H "conversation-id: test_conv_001" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

### Local Development: No Authentication

Omit both `user_manager` and `X-User-ID` header to use `"default_user"`:

```bash
nat run --config examples/agents/auto_memory_wrapper/configs/config_zep.yml
```

---

## Advanced Configuration Example

Here's a comprehensive configuration showing all available options:

```yaml
memory:
  zep_memory:
    _type: nat.plugins.zep_cloud/zep_memory

llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.3-70b-instruct
    temperature: 0.7

function_groups:
  calculator:
    _type: calculator

functions:
  my_react_agent:
    _type: react_agent
    llm_name: nim_llm
    tool_names: [calculator]
    verbose: true

workflow:
  _type: auto_memory_agent
  inner_agent_name: my_react_agent
  memory_name: zep_memory
  llm_name: nim_llm
  verbose: true
  description: "A ReAct agent with automatic Zep memory"

  # Feature flags (optional - all default to true)
  save_user_messages_to_memory: true
  retrieve_memory_for_every_response: true
  save_ai_messages_to_memory: true

  # Memory retrieval configuration (optional)
  search_params:
    mode: "summary"  # Zep: "basic" (fast) or "summary" (comprehensive)
    top_k: 5         # Maximum number of memories to retrieve

  # Memory storage configuration (optional)
  add_params:
    ignore_roles: ["assistant"]  # Zep: Exclude assistant messages from graph
```

---

## Wrapping Different Agent Types

The automatic memory wrapper works with any agent type:

### Wrapping a ReAct Agent

```yaml
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

### Wrapping a ReWOO Agent

```yaml
functions:
  my_rewoo_agent:
    _type: rewoo_agent
    llm_name: nim_llm
    tool_names: [wikipedia_search, calculator]

workflow:
  _type: auto_memory_agent
  inner_agent_name: my_rewoo_agent
  memory_name: zep_memory
  llm_name: nim_llm
```

### Wrapping a Tool Calling Agent

```yaml
functions:
  my_tool_calling_agent:
    _type: tool_calling_agent
    llm_name: nim_llm
    tool_names: [weather_tool, calculator]

workflow:
  _type: auto_memory_agent
  inner_agent_name: my_tool_calling_agent
  memory_name: zep_memory
  llm_name: nim_llm
```

---

## Important Notes

1. **User ID is runtime-only** - Set via `user_manager` or `X-User-ID` header, not in configuration
2. **Memory backends are interchangeable** - Works with any implementation of `MemoryEditor` interface
3. **No memory tools needed** - The wrapped agent does not need explicit memory tools configured
4. **Transparent to inner agent** - The wrapped agent is unaware of memory operations

---

## Examples

For complete working examples, refer to:
- `examples/agents/auto_memory_wrapper` - Full example with Zep Cloud integration

For additional information on memory backends and configuration, see:
- [Memory Module Documentation](../../store-and-retrieve/memory.md)
- [Adding a Memory Provider](../../extend/memory.md)
