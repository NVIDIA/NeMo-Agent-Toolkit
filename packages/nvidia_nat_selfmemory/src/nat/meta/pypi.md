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
