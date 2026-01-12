# NVIDIA NAT RAG Plugin

This plugin integrates [NVIDIA RAG](https://github.com/NVIDIA-AI-Blueprints/rag) with NeMo Agent Toolkit, providing RAG query and search capabilities for your agent workflows.

## Prerequisites

- Python 3.11+
- NeMo Agent Toolkit installed
- Access to NVIDIA AI endpoints (API key required)
- Milvus vector database running (default: `localhost:19530`)

## Installation

### 1. Install the Plugin

From the NeMo Agent Toolkit root directory:

```bash
# Activate your virtual environment
source .venv/bin/activate

# Install the plugin in editable mode
uv pip install -e packages/nvidia_nat_rag
```

### 2. Set Environment Variables

```bash
# Required: NVIDIA API key for embeddings, reranking, and LLM
export NVIDIA_API_KEY="your-nvidia-api-key"

# Optional: If using custom endpoints
# export NVIDIA_BASE_URL="https://integrate.api.nvidia.com/v1"
```

### 3. Start Milvus (Vector Database)

The plugin requires a Milvus instance. You can start one using Docker:

```bash
# Using Milvus Lite (for development)
# The plugin will automatically use milvus-lite if installed

# Or start a full Milvus instance with Docker
docker run -d --name milvus \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:latest
```

## Configuration

### Sample Config File

The plugin includes a sample configuration at:
```
packages/nvidia_nat_rag/src/nat/plugins/rag/configs/config.yml
```

### Available Functions

#### `nvidia_rag_query`
Queries documents using NVIDIA RAG and returns an AI-generated response.

```yaml
functions:
  rag_query:
    _type: nvidia_rag_query
    config_file: config.yaml          # Path to nvidia_rag config
    collection_names: ["my_docs"]     # Milvus collection names
    vdb_endpoint: "http://localhost:19530"
    use_knowledge_base: true
    # embedding_endpoint: "localhost:9080"  # Optional: for on-prem embeddings
```

#### `nvidia_rag_search`
Searches for relevant document chunks without generating a response.

```yaml
functions:
  rag_search:
    _type: nvidia_rag_search
    config_file: config.yaml
    collection_names: ["my_docs"]
    vdb_endpoint: "http://localhost:19530"
    reranker_top_k: 3                 # Number of results after reranking
    vdb_top_k: 20                     # Number of results from vector search
```

## Usage

### Running a RAG Workflow

```bash
nat run \
  --config_file packages/nvidia_nat_rag/src/nat/plugins/rag/configs/config.yml \
  --input "What is the price of a hammer?"
```

### Example Workflow Config

```yaml
functions:
  rag_query:
    _type: nvidia_rag_query
    config_file: config.yaml
    collection_names: ["product_catalog"]
    vdb_endpoint: "http://localhost:19530"
    use_knowledge_base: true

  rag_search:
    _type: nvidia_rag_search
    config_file: config.yaml
    collection_names: ["product_catalog"]
    vdb_endpoint: "http://localhost:19530"
    reranker_top_k: 3
    vdb_top_k: 20

  current_datetime:
    _type: current_datetime

llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    temperature: 0.0

workflow:
  _type: react_agent
  tool_names:
    - rag_query
    - rag_search
    - current_datetime
  llm_name: nim_llm
  verbose: true
```

## Troubleshooting

### Error: Function type `nvidia_rag_query` not found

The plugin is not installed. Run:
```bash
uv pip install -e packages/nvidia_nat_rag
```

### Error: Token limit exceeded

If you get a token limit error, reduce the number of results returned:
```yaml
rag_search:
  _type: nvidia_rag_search
  reranker_top_k: 1    # Reduce from 3
  vdb_top_k: 10        # Reduce from 20
```

This often happens when documents contain large base64-encoded images (charts, figures).

### Error: Connection refused to Milvus

Ensure Milvus is running:
```bash
# Check if Milvus is running
docker ps | grep milvus

# Start Milvus if not running
docker start milvus
```

### Error: NVIDIA API key not set

```bash
export NVIDIA_API_KEY="your-api-key"
```

## Directory Structure

```
packages/nvidia_nat_rag/
├── LICENSE.md
├── README.md                 # This file
├── pyproject.toml           # Package configuration
├── src/
│   └── nat/
│       ├── meta/
│       │   └── pypi.md
│       └── plugins/
│           └── rag/
│               ├── __init__.py
│               ├── configs/
│               │   └── config.yml    # Sample config
│               ├── rag_functions.py  # RAG function implementations
│               └── register.py       # Plugin registration
└── vendor/
    └── nvidia_rag-2.4.0.dev0-py3-none-any.whl  # Vendored dependency
```

## License

Apache-2.0
