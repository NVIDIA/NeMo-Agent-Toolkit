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

# NVIDIA Agent Intelligence Toolkit Memory Module

The AIQ Toolkit Memory subsystem is designed to store and retrieve a user's conversation history, preferences, and other "long-term memory." This is especially useful for building stateful LLM-based applications that recall user-specific data or interactions across multiple steps.

## Key Components

* **Memory Data Models**
   - **{py:class}`~aiq.data_models.memory.MemoryBaseConfig`**: A Pydantic base class that all memory config classes must extend. This is used for specifying memory registration in the AIQ Toolkit config file.
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

## Included Memory Modules
The AIQ Toolkit includes two concrete memory module implementations:
* [Mem0](https://mem0.ai/) which is provided by the [aiqtoolkit-mem0ai](https://pypi.org/project/aiqtoolkit-mem0ai/) plugin.
* [Zep](https://www.getzep.com/) which is provided by the [aiqtoolkit-zep-cloud](https://pypi.org/project/aiqtoolkit-zep-cloud/) plugin.

## Examples
The following examples demonstrate how to use the memory module in the AIQ Toolkit:
* `examples/semantic_kernel_demo`
* `examples/simple_rag`
