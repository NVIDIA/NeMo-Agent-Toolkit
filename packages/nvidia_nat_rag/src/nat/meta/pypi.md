# NVIDIA NeMo Agent Toolkit - RAG Plugin

This package provides integration between NVIDIA NeMo Agent Toolkit and the NVIDIA RAG library.

## Features

- **RAG Query**: Query documents using RAG with configurable LLM and embeddings
- **RAG Search**: Search for relevant documents in vector database collections

## Installation

```bash
pip install nvidia-nat-rag
```

## Usage

Add the RAG tools to your NAT workflow configuration:

```yaml
functions:
  rag_query:
    _type: nvidia_rag_query
    collection_names: ["my_collection"]
    vdb_endpoint: "http://localhost:19530"
    
workflow:
  _type: react_agent
  tool_names: [rag_query]
  llm_name: nim_llm
```

## Documentation

For more information, see the [NeMo Agent Toolkit documentation](https://docs.nvidia.com/nemo/agent-toolkit/latest/).
