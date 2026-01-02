# NVIDIA NeMo Agent Toolkit - MemMachine Integration

This package provides integration with MemMachine for memory management in NeMo Agent Toolkit.

## Overview

MemMachine is a unified memory management system that supports both episodic and semantic memory through a single interface. This integration allows you to use MemMachine as a memory backend for your NeMo Agent Toolkit workflows.

## Prerequisites

- Python 3.11+
- MemMachine server (install via `pip install memmachine-server` and run with `python memmachine-server`)
- MemMachine configuration file (`cfg.yml`)

## Installation

Install the package:

```bash
pip install nvidia-nat-memmachine
```

Or for development:

```bash
uv pip install -e packages/memmachine
```

## MemMachine Server Setup

MemMachine requires a running server instance. Install and run the MemMachine server:

```bash
pip install memmachine-server
python memmachine-server
```

The server will start on `http://localhost:8080` by default. Make sure your `cfg.yml` configuration file is properly set up with database connections (PostgreSQL for semantic memory and Neo4j for episodic memory).

## Configuration

MemMachine requires a `cfg.yml` configuration file that specifies:

- **Databases**: PostgreSQL and Neo4j connection details
- **AI Models**: LLMs and embedders for processing
- **Other Resources**: Rerankers, session managers, etc.

### Example Configuration

Create a `cfg.yml` file with your database and model configurations:

```yaml
episodic_memory:
  long_term_memory:
    embedder: openai_embedder
    vector_graph_store: my_storage_id
  short_term_memory:
    llm_model: openai_model
    message_capacity: 500

semantic_memory:
  llm_model: openai_model
  embedding_model: openai_embedder
  database: profile_storage

resources:
  databases:
    profile_storage:
      provider: postgres
      config:
        host: localhost  # Use 'postgres' if running memmachine in Docker
        port: 5432
        user: memmachine
        password: memmachine_password
        db_name: memmachine
    my_storage_id:
      provider: neo4j
      config:
        uri: 'bolt://localhost:7687'  # Use 'bolt://neo4j:7687' if running in Docker
        username: neo4j
        password: neo4j_password
  language_models:
    openai_model:
      provider: openai-responses
      config:
        model: "gpt-4o-mini"
        api_key: <YOUR_OPENAI_API_KEY>
        base_url: "https://api.openai.com/v1"
  embedders:
    openai_embedder:
      provider: openai
      config:
        model: "text-embedding-3-small"
        api_key: <YOUR_OPENAI_API_KEY>
        base_url: "https://api.openai.com/v1"
        dimensions: 1536
```

**Note**: 
- Use `localhost` for database hosts when databases are running on your host machine
- Adjust hostnames in the configuration based on your database setup

For more details, see the [MemMachine Configuration Documentation](https://docs.memmachine.ai/open_source/configuration).

## Usage in NeMo Agent Toolkit

Add MemMachine memory to your workflow configuration:

```yaml
memory:
  memmachine_memory:
    base_url: "http://localhost:8080"  # MemMachine server URL
    org_id: "my_org"  # Optional: default organization ID
    project_id: "my_project"  # Optional: default project ID
```

## Additional Resources

- [MemMachine Documentation](https://docs.memmachine.ai/)
- [NeMo Agent Toolkit Documentation](https://docs.nvidia.com/nemo/agent-toolkit/latest/)

