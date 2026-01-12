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

# Automatic Memory Wrapper for NeMo Agent Toolkit Agents

The `auto_memory_agent` wraps any agent to provide **automatic memory capture and retrieval** without requiring the LLM to invoke memory tools.

## Why Use This?

**Traditional tool-based memory:**
- LLMs may forget to call memory tools
- Memory capture is inconsistent
- Requires explicit memory tool configuration

**Automatic memory wrapper:**
- **Guaranteed capture**: User messages and agent responses are automatically stored
- **Automatic retrieval**: Relevant context is injected before each agent call
- **Memory backend agnostic**: Works with Zep, Mem0, Redis, or any `MemoryEditor`
- **Universal compatibility**: Wraps any agent type (ReAct, ReWOO, Tool Calling, etc.)

## Quick Start

### Basic Configuration

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

### Running the Example

```bash
# Set Zep credentials
export ZEP_API_KEY="your_api_key"

# Run the agent
nat run --config examples/agents/auto_memory_wrapper/configs/config_zep.yml
```

## Configuration Reference

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `inner_agent_name` | Name of the agent to wrap with automatic memory |
| `memory_name` | Name of the memory backend (from `memory:` section) |
| `llm_name` | LLM to use (required by `AgentBaseConfig`) |

### Optional Feature Flags

All default to `true`. Set to `false` to disable specific behaviors:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `save_user_messages_to_memory` | `true` | Automatically save user messages before agent processing |
| `retrieve_memory_for_every_response` | `true` | Automatically retrieve and inject memory context |
| `save_ai_messages_to_memory` | `true` | Automatically save agent responses after generation |

### Memory Backend Parameters

**`search_params`** - Passed to `memory_editor.search()`:
```yaml
search_params:
  mode: "summary"      # Zep: "basic" or "summary"
  top_k: 10           # Maximum memories to retrieve
```

**`add_params`** - Passed to `memory_editor.add_items()`:
```yaml
add_params:
  ignore_roles: ["assistant"]  # Zep: Exclude roles from graph memory
```

See `config_zep.yml` for comprehensive parameter examples.

## Multi-Tenant Memory Isolation

User ID is automatically extracted at runtime for memory isolation. It is **NOT** configured in YAML.

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

## Advanced Example

See `config_zep.yml` for a fully-commented configuration with all available parameters.

```yaml
workflow:
  _type: auto_memory_agent
  inner_agent_name: my_react_agent
  memory_name: zep_memory
  llm_name: nim_llm

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

## Important Notes

1. **User ID is runtime-only** - Set via `user_manager` or `X-User-ID` header, not in config
2. **Memory backends are interchangeable** - Works with any implementation of `MemoryEditor` interface

## Examples

See `configs/` directory:
- `config_zep.yml` - Comprehensive configuration with all parameters documented
