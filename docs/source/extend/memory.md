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

# Adding a Memory Provider

This documentation presumes familiarity with the NeMo Agent toolkit plugin architecture, the concept of "function registration" using `@register_function`, and how we define tool/workflow configurations in the NeMo Agent toolkit config described in the [Creating a New Tool and Workflow](../tutorials/create-a-new-workflow.md) tutorial.

## Key Memory Module Components

* **Memory Data Models**
   - **{py:class}`~nat.data_models.memory.MemoryBaseConfig`**: A Pydantic base class that all memory config classes must extend. This is used for specifying memory registration in the NeMo Agent toolkit config file.
   - **{py:class}`~nat.data_models.memory.MemoryBaseConfigT`**: A generic type alias for memory config classes.

* **Context Management**
   - **{py:class}`~nat.builder.context.Context`**: Manages request-scoped information including user identity. The `user_id` is retrieved from `Context.get().user_id` and passed to all memory operations.
   - **User attributes**: The Context also provides `user_first_name`, `user_last_name`, and `user_email`, which can be used by memory providers for enhanced features.
   - **HTTP headers**: In production deployments, user identity is set through HTTP headers (`user-id`, `user-first-name`, `user-last-name`, `user-email`) on incoming requests.

* **Memory Interfaces**
   - **{py:class}`~nat.memory.interfaces.MemoryEditor`** (abstract interface): The low-level API for adding, retrieving, and removing memory items. All methods require `user_id` as the first positional parameter.
   - **{py:class}`~nat.memory.interfaces.MemoryReader`** and **{py:class}`~nat.memory.interfaces.MemoryWriter`** (abstract classes): Provide structured read/write logic on top of the `MemoryEditor`.
   - **{py:class}`~nat.memory.interfaces.MemoryManager`** (abstract interface): Manages higher-level memory operations such as summarization or reflection if needed.

* **Memory Models**
   - **{py:class}`~nat.memory.models.MemoryItem`**: The main object representing a piece of memory. It includes:
     ```python
     conversation: list[dict[str, str]]  # user/assistant messages
     tags: list[str] = []
     metadata: dict[str, Any]
     memory: str | None  # optional textual memory
     ```
     > **Note**: The `user_id` is passed as a required positional parameter to memory editor methods, retrieved from the `Context` object to ensure proper security and multi-tenant isolation. It is not part of the `MemoryItem` object itself.
   - Helper models for search or deletion input: **{py:class}`~nat.memory.models.SearchMemoryInput`**, **{py:class}`~nat.memory.models.DeleteMemoryInput`**.


## Adding a Memory Module

In the NeMo Agent toolkit system, anything that extends {py:class}`~nat.data_models.memory.MemoryBaseConfig` and is declared with a `name="some_memory"` can be discovered as a *Memory type* by the NeMo Agent toolkit global type registry. This allows you to define a custom memory class to handle your own backends (Redis, custom database, a vector store, etc.). Then your memory class can be selected in the NeMo Agent toolkit config YAML via `_type: <your memory type>`.

### Basic Steps

1. **Create a config Class** that extends {py:class}`~nat.data_models.memory.MemoryBaseConfig`:
   ```python
   from nat.data_models.memory import MemoryBaseConfig

   class MyCustomMemoryConfig(MemoryBaseConfig, name="my_custom_memory"):
       # You can define any fields you want. For example:
       connection_url: str
       api_key: str
   ```
   > **Note**: The `name="my_custom_memory"` ensures that NeMo Agent toolkit can recognize it when the user places `_type: my_custom_memory` in the memory config.

2. **Implement a {py:class}`~nat.memory.interfaces.MemoryEditor`** that uses your backend**:
   ```python
   from nat.memory.interfaces import MemoryEditor, MemoryItem

   class MyCustomMemoryEditor(MemoryEditor):
       def __init__(self, config: MyCustomMemoryConfig):
           self._api_key = config.api_key
           self._conn_url = config.connection_url
           # Possibly set up connections here

       async def add_items(self, items: list[MemoryItem], user_id: str, **kwargs) -> None:
           # Insert into your custom DB or vector store
           # user_id is a required parameter (after items) for multi-tenant isolation
           ...

       async def retrieve_memory(self, query: str, user_id: str, **kwargs) -> str:
           # Perform your query in the DB or vector store
           # user_id is a required parameter (after query) for multi-tenant isolation
           # Return formatted memory as a string optimized for LLM consumption
           ...

       async def remove_items(self, user_id: str, **kwargs) -> None:
           # Implement your deletion logic
           # user_id is a required parameter for multi-tenant isolation
           ...
   ```
3. **Tell NeMo Agent toolkit how to build your MemoryEditor**. Typically, you do this by hooking into the builder system so that when `builder.get_memory_client("my_custom_memory")` is called, it returns an instance of `MyCustomMemoryEditor`.
   - For example, you might define a `@register_memory` or do it manually with the global type registry. The standard pattern is to see how `mem0`, `redis` or `zep` memory is integrated in the code. For instance, see `packages/nvidia_nat_mem0ai/src/nat/plugins/mem0ai/memory.py` to see how `mem0_memory` is integrated.

4. **Use in config**: Now in your NeMo Agent toolkit config, you can do something like:
   ```yaml
   memory:
     my_store:
       _type: my_custom_memory
       connection_url: "http://localhost:1234"
       api_key: "some-secret"
   ...
   ```

> The user can then reference `my_store` in their function or workflow config (for example, in a memory-based tool).

---

## Bringing Your Own Memory Client Implementation

A typical pattern is:
- You define a *config class* that extends {py:class}`~nat.data_models.memory.MemoryBaseConfig` (giving it a unique `_type` / name).
- You define the actual *runtime logic* in a "Memory Editor" or "Memory Client" class that implements {py:class}`~nat.memory.interfaces.MemoryEditor`.
- You connect them together (for example, by implementing a small factory function or a method in the builder that says: "Given `MyCustomMemoryConfig`, return `MyCustomMemoryEditor(config)`").

### Example: Minimal Skeleton

```python
# my_custom_memory_config.py
from nat.data_models.memory import MemoryBaseConfig

class MyCustomMemoryConfig(MemoryBaseConfig, name="my_custom_memory"):
    url: str
    token: str

# my_custom_memory_editor.py
from nat.memory.interfaces import MemoryEditor, MemoryItem

class MyCustomMemoryEditor(MemoryEditor):
    def __init__(self, cfg: MyCustomMemoryConfig):
        self._url = cfg.url
        self._token = cfg.token

    async def add_items(self, items: list[MemoryItem], user_id: str, **kwargs) -> None:
        # Insert memory items for the specified user
        # user_id is a required parameter (after items) for multi-tenant isolation
        pass

    async def retrieve_memory(self, query: str, user_id: str, top_k: int = 5, **kwargs) -> str:
        # Retrieve and format memory for the specified user
        # user_id is a required parameter (after query) for multi-tenant isolation
        # Return formatted memory as a string optimized for LLM consumption
        pass

    async def remove_items(self, user_id: str, **kwargs) -> None:
        # Remove memory items for the specified user
        # user_id is a required parameter for multi-tenant isolation
        pass
```

Then either:
- Write a small plugin method that `@register_memory` or `@register_function` with `framework_wrappers`, or
- Add a snippet to your plugin's `__init__.py` that calls the NeMo Agent toolkit TypeRegistry, passing your config.

---

## Best Practices for Agent Memory

### AutoMemoryWrapper: Recommended Pattern

The `AutoMemoryWrapper` (implemented as `auto_memory_agent` in workflow configurations) is the recommended approach for implementing agent memory in the NeMo Agent toolkit. This wrapper automatically handles memory capture and retrieval for any agent, following best practices for memory management:

- **Automatic capture**: Captures every raw message between the user and the agent, ensuring complete memory storage without relying on LLM decisions.
- **Automatic retrieval**: Retrieves relevant memory context automatically for every user message, using recent messages as the search query for optimal context.
- **Deterministic operation**: Memory operations happen deterministically, reducing latency compared to requiring the LLM to decide when to search memory.

To use the `AutoMemoryWrapper`, define your agent in the `functions` section and wrap it with the `auto_memory_agent` workflow:

```yaml
functions:
  my_agent:
    _type: <your_agent_type>
    llm_name: <llm_name>
    use_openai_api: true
    # ... your agent configuration

workflow:
  _type: auto_memory_agent
  inner_agent_name: my_agent
  memory_name: <memory_config_name>
  llm_name: <llm_name>
  # retrieve_memory_for_every_response: true  # Automatic retrieval (default)
```

### Memory Tools: Alternative Approach

For use cases requiring explicit memory search control, you can disable automatic retrieval and use memory tools instead:

```yaml
functions:
  get_memory:
    _type: get_memory
    memory: <memory_config_name>
    description: Retrieve relevant memories from long-term memory.
  
  my_agent:
    _type: <your_agent_type>
    llm_name: <llm_name>
    tool_names: [get_memory]
    use_openai_api: true

workflow:
  _type: auto_memory_agent
  inner_agent_name: my_agent
  memory_name: <memory_config_name>
  llm_name: <llm_name>
  retrieve_memory_for_every_response: false  # Disable automatic retrieval
```

This keeps automatic message capture while giving the agent explicit control over memory retrieval through tool calls.

---

## Using Memory in a Workflow

**At runtime**, you typically see code like:

```python
from nat.builder.context import Context

memory_client = builder.get_memory_client(<memory_config_name>)
user_id = Context.get().user_id
await memory_client.add_items(user_id, [MemoryItem(...), ...])
```

or

```python
from nat.builder.context import Context

user_id = Context.get().user_id
formatted_memory = await memory_client.retrieve_memory(user_id, query="What did user prefer last time?", top_k=3)
```

**Inside Tools**: Tools that read or write memory retrieve the `user_id` from the `Context` object and pass it to memory client methods. For example:

```python
from nat.builder.context import Context
from nat.memory.models import MemoryItem
from langchain_core.tools import ToolException

async def add_memory_tool_action(item: MemoryItem, memory_name: str):
    memory_client = builder.get_memory_client(memory_name)
    user_id = Context.get().user_id
    try:
        await memory_client.add_items(user_id, [item])
        return "Memory added successfully"
    except Exception as e:
        raise ToolException(f"Error adding memory: {e}")
```

### Example Configuration

#### Example 1: AutoMemoryWrapper with Automatic Retrieval (Recommended)

This example shows the recommended pattern using `AutoMemoryWrapper` with automatic memory capture and retrieval:

```yaml
llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.3-70b-instruct
    temperature: 0.7

memory:
  zep_memory:
    _type: nat.plugins.zep_cloud/zep_memory
    # API credentials loaded from environment variables:
    # ZEP_API_KEY and ZEP_PROJECT_ID

functions:
  calculator:
    _type: calculator_multiply
  
  my_react_agent:
    _type: react_agent
    llm_name: nim_llm
    tool_names: [calculator]
    use_openai_api: true
    verbose: true

workflow:
  _type: auto_memory_agent
  inner_agent_name: my_react_agent
  memory_name: zep_memory
  llm_name: nim_llm
  verbose: true
  # All feature flags default to true - memory save and retrieval are automatic!
  # save_user_messages_to_memory: true
  # retrieve_memory_for_every_response: true
  # save_ai_messages_to_memory: true
```

Explanation:

- We define a memory backend named `zep_memory` using the [Zep](https://www.getzep.com/) provider included in the [`nvidia-nat-zep-cloud`](https://pypi.org/project/nvidia-nat-zep-cloud/) plugin.
- The inner agent (`my_react_agent`) is defined in the `functions` section with its own tools.
- The `auto_memory_agent` workflow wraps the inner agent and automatically captures all messages between the user and agent.
- By default (or explicitly with `retrieve_memory_for_every_response: true`), relevant memory is automatically retrieved and provided to the agent for each user message.
- The agent can focus on its core tasks (such as calculation) without needing to manage memory explicitly.

#### Example 2: AutoMemoryWrapper with Memory Tools

This example shows `AutoMemoryWrapper` with automatic capture but manual retrieval through tools:

```yaml
llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    temperature: 0.0

memory:
  zep_memory:
    _type: nat.plugins.zep_cloud/zep_memory

functions:
  current_datetime:
    _type: current_datetime
  
  get_memory:
    _type: get_memory
    memory: zep_memory
    description: >-
      Retrieve relevant memories to personalize your response to the user.
      Call this tool once before generating each response.
  
  my_react_agent:
    _type: react_agent
    tool_names: [current_datetime, get_memory]
    llm_name: nim_llm
    use_openai_api: true
    verbose: true

workflow:
  _type: auto_memory_agent
  inner_agent_name: my_react_agent
  memory_name: zep_memory
  llm_name: nim_llm
  verbose: true
  save_user_messages_to_memory: true
  retrieve_memory_for_every_response: false  # Agent controls memory retrieval via tools
  save_ai_messages_to_memory: true
  use_openai_api: true
```

Explanation:

- We define a memory backend named `zep_memory` using the [Zep](https://www.getzep.com/) provider included in the [`nvidia-nat-zep-cloud`](https://pypi.org/project/nvidia-nat-zep-cloud/) plugin.
- The `get_memory` tool is defined in the `functions` section and included in the agent's `tool_names`.
- The `auto_memory_agent` workflow still captures all messages automatically, but with `retrieve_memory_for_every_response: false`, the agent must explicitly call the `get_memory` tool to retrieve memory.
- This approach gives the agent control over when to search memory, which can be useful for specific use cases where memory retrieval should be selective.

---


## Putting It All Together

To **bring your own memory**:

1. **Implement** a custom {py:class}`~nat.data_models.memory.MemoryBaseConfig` (with a unique `_type`).
2. **Implement** a custom {py:class}`~nat.memory.interfaces.MemoryEditor` that can handle `add_items`, `search`, `remove_items` calls.
3. **Register** your config class so that the NeMo Agent toolkit type registry is aware of `_type: <your memory>`.
4. In your `.yml` config, specify:
   ```yaml
   memory:
     user_store:
       _type: <your memory>
       # any other fields your config requires
   ```
5. Use `builder.get_memory_client("user_store")` to retrieve an instance of your memory in your code or tools.

---

## Summary

- The **Memory** module in NeMo Agent toolkit revolves around the {py:class}`~nat.memory.interfaces.MemoryEditor` interface and {py:class}`~nat.memory.models.MemoryItem` model.
- **User identity** is managed through the `Context` object and passed as a required positional parameter to memory editor methods, ensuring proper security and multi-tenant isolation.
- **Memory retrieval** is performed through the `retrieve_memory()` method, which returns formatted memory as a string optimized for LLM consumption, allowing each provider to control how memory is presented.
- **Configuration** is done through a subclass of {py:class}`~nat.data_models.memory.MemoryBaseConfig` that is discriminated by the `_type` field in the YAML config.
- **Registration** can be as simple as adding `name="my_custom_memory"` to your config class and letting NeMo Agent toolkit discover it.
- Tools and workflows then seamlessly **read/write** user memory by calling `builder.get_memory_client(...)` and passing the `user_id` from the `Context` object.

This modular design allows any developer to **plug in** a new memory backend—such as Zep, a custom embedding store, or a simple dictionary-based store—by following these steps. Once integrated, your **agent** (or tools) will treat it just like any other memory in the system.

---

**That's it!** You now know how to create, register, and use a **custom memory client** in NeMo Agent toolkit. Feel free to explore the existing memory clients in the `src/nat/memory` directory for reference and see how they are integrated into the overall framework.
