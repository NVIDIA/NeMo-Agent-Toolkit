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

![NVIDIA NeMo Agent Toolkit](https://media.githubusercontent.com/media/NVIDIA/NeMo-Agent-Toolkit/refs/heads/main/docs/source/_static/banner.png "NeMo Agent Toolkit banner image")

# nvidia-nat-selfmemory

SelfMemory memory provider plugin for the NVIDIA NeMo Agent Toolkit.

This package provides a `MemoryEditor` implementation that uses [SelfMemory](https://github.com/selfmemory/selfmemory) as the memory backend, enabling AI agents built with NeMo Agent Toolkit to store and retrieve long-term memories through SelfMemory's multi-tenant, vector-based memory system.

## Features

- 29+ vector store backends (Qdrant, ChromaDB, Pinecone, Milvus, etc.)
- 15+ embedding providers (OpenAI, Ollama, HuggingFace, etc.)
- Multi-tenant user isolation
- Optional LLM-based intelligent fact extraction
- Built-in encryption support

## Usage

```yaml
memory:
  user_store:
    _type: selfmemory
    vector_store_provider: qdrant
    vector_store_config:
      host: localhost
      port: 6333
    embedding_provider: openai
    embedding_config:
      model: text-embedding-3-small
```
