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

# NVIDIA NeMo Agent Toolkit Vanna

Vanna-based Text-to-SQL integration for NeMo Agent toolkit.

## Overview

This package provides production-ready text-to-SQL capabilities using the Vanna framework with Databricks support.

## Features

- **AI-Powered SQL Generation**: Convert natural language to SQL using LLMs
- **Databricks Support**: Optimized for Databricks SQL warehouses
- **Vector-Based Similarity Search**: Milvus integration for few-shot learning
- **Streaming Support**: Real-time progress updates
- **Query Execution**: Optional database execution with formatted results
- **Highly Configurable**: Customizable prompts, examples, and connections

## Quick Start

Install the package:

```bash
pip install nvidia-nat-vanna
```

Create a workflow configuration:

```yaml
functions:
  text2sql:
    _type: text2sql
    llm_name: my_llm
    embedder_name: my_embedder
    milvus_retriever: my_retriever
    database_type: databricks
    connection_url: "${CONNECTION_URL}"
    execute_sql: false

  execute_db_query:
    _type: execute_db_query
    database_type: databricks
    connection_url: "${CONNECTION_URL}"
    max_rows: 100

llms:
  my_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    api_key: "${NVIDIA_API_KEY}"

embedders:
  my_embedder:
    _type: nim
    model_name: nvidia/llama-3.2-nv-embedqa-1b-v2
    api_key: "${NVIDIA_API_KEY}"

retrievers:
  my_retriever:
    _type: milvus_retriever
    uri: "${MILVUS_URI}"
    connection_args:
      user: "developer"
      password: "${MILVUS_PASSWORD}"
      db_name: "default"
    embedding_model: my_embedder
    content_field: text
    use_async_client: true

workflow:
  _type: rewoo_agent
  tool_names: [text2sql, execute_db_query]
  llm_name: my_llm
```

Run the workflow:

```bash
nat run --config config.yml --input "How many customers do we have?"
```

## Components

### `text2sql` Function

Generates SQL queries from natural language using:
- Few-shot learning with similar examples
- DDL (schema) information
- Custom documentation
- LLM-powered query generation

### `execute_db_query` Function

Executes SQL queries and returns formatted results:
- Databricks SQL execution
- Result limiting and pagination
- Structured output format
- SQLAlchemy Object Relational Mapper (ORM)-based connection

## Use Cases

- **Business Intelligence**: Enable non-technical users to query data
- **Data Exploration**: Rapid prototyping and analysis
- **Conversational Analytics**: Multi-turn Q&A about your data
- **SQL Assistance**: Help analysts write complex queries

## Documentation

Full documentation: <https://docs.nvidia.com/nemo/agent-toolkit/latest/>

## License

Part of NVIDIA NeMo Agent toolkit. See repository for license details.
