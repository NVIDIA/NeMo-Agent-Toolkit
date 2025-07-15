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
   - **{py:class}`~aiq.data_models.memory.MemoryBaseConfig`**: A Pydantic base class that all memory config classes must extend. This is used for specifying memory registration in the NeMo Agent toolkit config file.
   - **{py:class}`~aiq.data_models.memory.MemoryBaseConfigT`**: A generic type alias for memory config classes.

* **Memory Interfaces**
   - **{py:class}`~aiq.memory.interfaces.MemoryEditor`** (abstract interface): The low-level API for adding, searching, and removing memory items.
   - **{py:class}`~aiq.memory.interfaces.MemoryReader`** and **{py:class}`~aiq.memory.interfaces.MemoryWriter`** (abstract classes): Provide structured read/write logic on top of the `MemoryEditor`.
   - **{py:class}`~aiq.memory.interfaces.MemoryManager`** (abstract interface): Manages higher-level memory operations like summarization or reflection if needed.

* **Memory Models**
   - **{py:class}`~aiq.memory.models.MemoryItem`**: The main object representing a piece of memory. It includes:
     ```python
     conversation: list[dict[str, str]]  # user/assistant messages
     tags: list[str] = []
     metadata: dict[str, Any]
     user_id: str
     memory: str | None  # optional textual memory
     ```
   - Helper models for search or deletion input: **{py:class}`~aiq.memory.models.SearchMemoryInput`**, **{py:class}`~aiq.memory.models.DeleteMemoryInput`**.


## Adding a Memory Module

In the NeMo Agent toolkit system, anything that extends {py:class}`~aiq.data_models.memory.MemoryBaseConfig` and is declared with a `name="some_memory"` can be discovered as a *Memory type* by the NeMo Agent toolkit global type registry. This allows you to define a custom memory class to handle your own backends (Redis, custom database, a vector store, etc.). Then your memory class can be selected in the NeMo Agent toolkit config YAML via `_type: <your memory type>`.

### Basic Steps

1. **Create a config Class** that extends {py:class}`~aiq.data_models.memory.MemoryBaseConfig`:
   ```python
   from aiq.data_models.memory import MemoryBaseConfig

   class MyCustomMemoryConfig(MemoryBaseConfig, name="my_custom_memory"):
       # You can define any fields you want. For example:
       connection_url: str
       api_key: str
   ```
   > **Note**: The `name="my_custom_memory"` ensures that NeMo Agent toolkit can recognize it when the user places `_type: my_custom_memory` in the memory config.

2. **Implement a {py:class}`~aiq.memory.interfaces.MemoryEditor`** that uses your backend**:
   ```python
   from aiq.memory.interfaces import MemoryEditor, MemoryItem

   class MyCustomMemoryEditor(MemoryEditor):
       def __init__(self, config: MyCustomMemoryConfig):
           self._api_key = config.api_key
           self._conn_url = config.connection_url
           # Possibly set up connections here

       async def add_items(self, items: list[MemoryItem]) -> None:
           # Insert into your custom DB or vector store
           ...

       async def search(self, query: str, top_k: int = 5, **kwargs) -> list[MemoryItem]:
           # Perform your query in the DB or vector store
           ...

       async def remove_items(self, **kwargs) -> None:
           # Implement your deletion logic
           ...
   ```
3. **Tell NeMo Agent toolkit how to build your MemoryEditor**. Typically, you do this by hooking into the builder system so that when `builder.get_memory_client("my_custom_memory")` is called, it returns an instance of `MyCustomMemoryEditor`.
   - For example, you might define a `@register_memory` or do it manually with the global type registry. (The standard pattern is to see how `mem0_memory` or `zep` memory is integrated in the code under `aiq/memory/<provider>`.)

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
- You define a *config class* that extends {py:class}`~aiq.data_models.memory.MemoryBaseConfig` (giving it a unique `_type` / name).
- You define the actual *runtime logic* in a "Memory Editor" or "Memory Client" class that implements {py:class}`~aiq.memory.interfaces.MemoryEditor`.
- You connect them together (for example, by implementing a small factory function or a method in the builder that says: "Given `MyCustomMemoryConfig`, return `MyCustomMemoryEditor(config)`").

### Example: Minimal Skeleton

```python
# my_custom_memory_config.py
from aiq.data_models.memory import MemoryBaseConfig

class MyCustomMemoryConfig(MemoryBaseConfig, name="my_custom_memory"):
    url: str
    token: str

# my_custom_memory_editor.py
from aiq.memory.interfaces import MemoryEditor, MemoryItem

class MyCustomMemoryEditor(MemoryEditor):
    def __init__(self, cfg: MyCustomMemoryConfig):
        self._url = cfg.url
        self._token = cfg.token

    async def add_items(self, items: list[MemoryItem]) -> None:
        # ...
        pass

    async def search(self, query: str, top_k: int = 5, **kwargs) -> list[MemoryItem]:
        # ...
        pass

    async def remove_items(self, **kwargs) -> None:
        # ...
        pass
```

Then either:
- Write a small plugin method that `@register_memory` or `@register_function` with `framework_wrappers`, or
- Add a snippet to your plugin's `__init__.py` that calls the NeMo Agent toolkit TypeRegistry, passing your config.

---

## Using Memory in a Workflow

**At runtime**, you typically see code like:

```python
memory_client = builder.get_memory_client(<memory_config_name>)
await memory_client.add_items([MemoryItem(...), ...])
```

or

```python
memories = await memory_client.search(query="What did user prefer last time?", top_k=3)
```

**Inside Tools**: Tools that read or write memory simply call the memory client. For example:

```python
from aiq.memory.models import MemoryItem
from langchain_core.tools import ToolException

async def add_memory_tool_action(item: MemoryItem, memory_name: str):
    memory_client = builder.get_memory_client(memory_name)
    try:
        await memory_client.add_items([item])
        return "Memory added successfully"
    except Exception as e:
        raise ToolException(f"Error adding memory: {e}")
```

### Example Configuration

Here are the relevant sections from the `examples/intermediate/RAG/simple_rag/configs/milvus_memory_rag_config.yml` in the source code repository:

```yaml
memory:
  saas_memory:
    _type: mem0_memory
```
```yaml
functions:
  add_memory:
    _type: add_memory
    memory: saas_memory
    description: |
      Add any facts about user preferences to long term memory. Always use this if users mention a preference.
      The input to this tool should be a string that describes the user's preference, not the question or answer.
  get_memory:
    _type: get_memory
    memory: saas_memory
    description: |
      Always call this tool before calling any other tools, even if the user does not mention to use it.
      The question should be about user preferences which will help you format your response.
      For example: "How does the user like responses formatted?"
```
```yaml
workflow:
  _type: react_agent
  tool_names:
    - add_memory
    - get_memory
  llm: nim_llm
```

Explanation:

- We define a memory entry named `saas_memory` with `_type: mem0_memory`, using the [Mem0](https://mem0.ai/) provider included in the [`aiqtoolkit-mem0ai`](https://pypi.org/project/aiqtoolkit-mem0ai/) plugin.
- Then we define two tools (functions in NeMo Agent toolkit terminology) that reference `saas_memory`: `add_memory` and `get_memory`.
- Finally, the `agent_memory` workflow references these two tool names.

---


## Putting It All Together

To **bring your own memory**:

1. **Implement** a custom {py:class}`~aiq.data_models.memory.MemoryBaseConfig` (with a unique `_type`).
2. **Implement** a custom {py:class}`~aiq.memory.interfaces.MemoryEditor` that can handle `add_items`, `search`, `remove_items` calls.
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

- The **Memory** module in NeMo Agent toolkit revolves around the {py:class}`~aiq.memory.interfaces.MemoryEditor` interface and {py:class}`~aiq.memory.models.MemoryItem` model.
- **Configuration** is done via a subclass of {py:class}`~aiq.data_models.memory.MemoryBaseConfig` that is *discriminated* by the `_type` field in the YAML config.
- **Registration** can be as simple as adding `name="my_custom_memory"` to your config class and letting NeMo Agent toolkit discover it.
- Tools and workflows then seamlessly **read/write** user memory by calling `builder.get_memory_client(...)`.

This modular design allows any developer to **plug in** a new memory backend—like `Zep`, a custom embedding store, or even a simple dictionary-based store—by following these steps. Once integrated, your **agent** (or tools) will treat it just like any other memory in the system.

---

**That's it!** You now know how to create, register, and use a **custom memory client** in NeMo Agent toolkit. Feel free to explore the existing memory clients in the `aiq/memory` directory for reference and see how they are integrated into the overall framework.
