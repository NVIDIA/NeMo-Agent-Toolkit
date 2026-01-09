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

# Text-to-SQL with Vanna

The NVIDIA NeMo Agent toolkit provides text-to-SQL capabilities through the `text2sql` and `execute_db_query` functions, powered by the Vanna framework with Databricks support and vector-based few-shot learning.

## Features

- **Text-to-SQL Generation**: Convert natural language questions to SQL queries using AI
- **Databricks Support**: Optimized for Databricks SQL warehouses and compute clusters
- **Vector Store Integration**: Milvus-based similarity search for few-shot learning
- **Streaming Support**: Real-time progress updates during SQL generation
- **Database Execution**: Optional query execution with result formatting
- **Customizable**: Flexible configuration for prompts, examples, and database connections

## Installation

The text-to-SQL plugin is distributed as a separate package that can be installed alongside the NeMo Agent toolkit. If you have not yet installed the NeMo Agent toolkit, refer to the [Installation](../../get-started/installation.md) guide.

If you have performed a source code checkout, you can install this with the following command:

```bash
uv pip install -e '.[vanna]'
```

If you have installed the NeMo Agent toolkit from a package, you can install this with the following command:

```bash
uv pip install "nvidia-nat[vanna]"
```

## Quick Start

### Prerequisites

- NVIDIA API Key (refer to [Obtaining API Keys](../../get-started/quick-start.md#obtaining-api-keys))
- Milvus vector database (local or cloud)
- Databricks workspace with SQL warehouse or compute cluster access

### 1. Start Milvus

Install and start Milvus standalone with docker compose following [these steps](https://milvus.io/docs/v2.3.x/install_standalone-docker-compose.md).

### 2. Set Environment Variables

Create a `.env` file:

```bash
# NVIDIA API
NVIDIA_API_KEY=nvapi-xxx

# Database (Databricks)
CONNECTION_URL=databricks://token:<token>@<db_host>:443/default?http_path=<http_path>&catalog=main&schema=default

# Milvus
MILVUS_URI=http://localhost:19530
MILVUS_PASSWORD=your-password
```

### 3. Create Workflow Configuration

#### 3.1 Create training config `text2sql_training_config.yml`

```yaml
general:
  telemetry:
    logging:
      console:
        _type: console
        level: INFO

functions:
  text2sql:
    _type: text2sql
    llm_name: nim_llm
    embedder_name: nim_embedder
    milvus_retriever: milvus_retriever

    # Database config
    database_type: databricks
    connection_url: "${CONNECTION_URL}"

    # Vanna settings
    execute_sql: false
    train_on_startup: true
    auto_training: true # Auto-train Vanna (auto-extract DDL and generate training data from database) or manually train Vanna (uses training data from training_db_schema.py)
    n_results: 5
    milvus_search_limit: 1000

  execute_db_query:
    _type: execute_db_query
    database_type: databricks
    connection_url: "${CONNECTION_URL}"
    max_rows: 100

llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    api_key: "${NVIDIA_API_KEY}"
    base_url: https://integrate.api.nvidia.com/v1
    temperature: 0.0

embedders:
  nim_embedder:
    _type: nim
    model_name: nvidia/llama-3.2-nv-embedqa-1b-v2
    api_key: "${NVIDIA_API_KEY}"
    base_url: https://integrate.api.nvidia.com/v1

retrievers:
  milvus_retriever:
    _type: milvus_retriever
    uri: "${MILVUS_URI}"
    connection_args:
      user: "developer"
      password: "${MILVUS_PASSWORD}"
      db_name: "default"
    embedding_model: nim_embedder
    content_field: text
    use_async_client: true

workflow:
  _type: rewoo_agent
  tool_names: [text2sql, execute_db_query]
  llm_name: nim_llm
  tool_call_max_retries: 3
```

Update training materials in `training_db_schema.py`:
- `VANNA_TRAINING_DOCUMENTATION`: Add documentation about your tables and business logic
- `VANNA_TRAINING_DDL`: Provide DDL statements for your database schema
  - If `auto_training` is set to `true`, make sure `VANNA_ACTIVE_TABLES` is updated with the tables in your database. This ensures that automatic DDL extraction works properly.
- `VANNA_TRAINING_EXAMPLES`: Provide question-SQL example pairs for few-shot learning

#### 3.2 Create inference config `text2sql_config.yml`

Set `train_on_startup` to `false` for faster startup when using pre-trained data:
```yaml
functions:
  text2sql:
    train_on_startup: false
    auto_training: false
```
See `text2sql_training_config.yml` and `text2sql_config.yml` for reference.

### 4. Run the Workflow

The following examples show how to use the text-to-SQL workflow with the NeMo Agent toolkit CLI or programmatically.

```bash
# Using NeMo Agent toolkit CLI
# If auto_training is set to true, training takes approximately 7 minutes depending on endpoints and network conditions.
nat run --config_file packages/nvidia_nat_vanna/text2sql_training_config.yml --input "Retrieve the total number of customers."

# Once training is complete, use the inference configuration for faster generation.
nat run --config_file packages/nvidia_nat_vanna/text2sql_config.yml --input "What is the total profit?"
```

Or use the Python API:
```python
import asyncio
from nat.core import Workflow

async def main():
    workflow = Workflow.from_config("text2sql_config.yml")
    result = await workflow.run("Retrieve the total number of customers.")
    print(result)

asyncio.run(main())
```

Expected output:
```text
# Ingest DDL and synthesize query-SQL pairs for training
Training Vanna...

# ReWOO Agent Planning Phase
Plan 1: Generate SQL query from natural language
  Tool: text2sql
Plan 2: Execute the generated SQL query
  Tool: execute_db_query

# Execution Phase
Starting SQL generation...
Retrieved 1 similar SQL examples
SQL generated: SELECT COUNT(*) FROM customers

Executing SQL query...
Results: 42 customers found
```

## Configuration

### Text2SQL Function

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `llm_name` | `str` | [LLM](../../build-workflows/llms/index.md) reference for SQL generation | Required |
| `embedder_name` | `str` | [Embedder](../../build-workflows/embedders.md) reference for vector ops | Required |
| `milvus_retriever` | `str` | Milvus retriever reference (must use `use_async_client=true`) | Required |
| `database_type` | `str` | Database type (must be 'Databricks') | "Databricks" |
| `connection_url` | `str` | Database connection string (SQLAlchemy format) | Required |
| `execute_sql` | `bool` | Execute SQL or just return query | false |
| `allow_llm_to_see_data` | `bool` | Allow intermediate queries | false |
| `train_on_startup` | `bool` | Train Vanna on startup | false |
| `auto_training` | `bool` | Auto-extract DDL and generate training data | false |
| `initial_prompt` | `str` | Custom system prompt | null |
| `n_results` | `int` | Number of similar examples | 5 |
| `sql_collection` | `str` | Milvus collection name for SQL examples | `"vanna_sql"` |
| `ddl_collection` | `str` | Milvus collection name for DDL | `"vanna_ddl"` |
| `doc_collection` | `str` | Milvus collection name for documentation | `"vanna_documentation"` |
| `milvus_search_limit` | `int` | Maximum limit for vector search operations | 1000 |
| `reasoning_models` | `set[str]` | Models requiring think tag removal | See below |
| `chat_models` | `set[str]` | Models using standard response handling | See below |

**Default reasoning models**: `nvidia/llama-3.1-nemotron-ultra-253b-v1`, `nvidia/llama-3.3-nemotron-super-49b-v1.5`, `deepseek-ai/deepseek-v3.1`, `deepseek-ai/deepseek-r1`

**Default chat models**: `meta/llama-3.1-70b-instruct`

#### Understanding `train_on_startup` and `auto_training`

**`train_on_startup`**: Controls whether Vanna initializes and loads training data when the workflow starts.

- **`true`**: Automatically creates Milvus collections with names specified by `sql_collection`, `ddl_collection`, and `doc_collection` parameters (defaults: `"vanna_sql"`, `"vanna_ddl"`, `"vanna_documentation"`) and ingests training data during workflow initialization. This ensures the vector store is populated and ready for similarity search before the first query is processed. Use this setting when you want to ensure fresh training data is loaded each time the workflow starts.

- **`false`** (default): Skips automatic collection creation and training data ingestion. The workflow assumes Milvus collections already exist and contain previously trained data. Use this setting in production environments where training data is already loaded.

**`auto_training`**: Controls the source of training data (only used when `train_on_startup=true`).

- **`true`**: Automatically extracts DDL from the database using `VANNA_ACTIVE_TABLES` and generates question-SQL training pairs using the LLM. This is useful when you want to quickly bootstrap the system with your existing database schema.

- **`false`** (default): Uses manually defined training data from `training_db_schema.py` (`VANNA_TRAINING_DDL`, `VANNA_TRAINING_EXAMPLES`, `VANNA_TRAINING_DOCUMENTATION`). This gives you full control over the training data quality.

### Database Configuration

**Databricks:**
```yaml
database_type: databricks
connection_url: "databricks://token:${DB_TOKEN}@${DB_HOST}:443/default?http_path=${HTTP_PATH}&catalog=main&schema=default"
```

**Connection URL Format:**
```text
databricks://token:<token>@<db_host>:443/default?http_path=<http_path>&catalog=<catalog>&schema=<schema>
```

**Parameters:**
- `<token>`: Databricks personal access token or service principal token
- `<db_host>`: Your Databricks workspace URL, for example `your-workspace.cloud.databricks.com`
- `<http_path>`: Path to your SQL warehouse or compute cluster, for example `/sql/1.0/warehouses/abc123`
- `<catalog>`: Catalog name, for example `main`
- `<schema>`: Schema name, for example `default`

**Example:**
```bash
CONNECTION_URL="databricks://token:dapi-xxx@your-workspace.cloud.databricks.com:443/default?http_path=/sql/1.0/warehouses/abc123&catalog=main&schema=default"
```

**Note**: Only Databricks is currently supported. The connection uses SQLAlchemy with the `databricks-sql-connector` driver. Other databases can be customized as following:
```python
# PostgreSQL
engine = create_engine("postgresql+psycopg://user:password@localhost:5432/mydb")

# MS SQL Server
engine = create_engine(
    "mssql+pyodbc://user:password@server/db?driver=ODBC+Driver+18+for+SQL+Server"
)
# SQLite
engine = create_engine("sqlite:///local.db")
```

### Execute DB Query Function

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `database_type` | `str` | Database type (must be 'Databricks') | "Databricks" |
| `connection_url` | `str` | Database connection string (SQLAlchemy format) | Required |
| `max_rows` | `int` | Maximum rows to return | 100 |

### Milvus Configuration

The text2sql function connects to Milvus using environment variables and manages collections internally. For advanced use cases, you can configure Milvus connection settings:

```yaml
# Optional: Custom retriever for additional collections
retrievers:
  milvus_retriever:
    _type: milvus_retriever
    uri: "${MILVUS_URI}"  # Supports both http://localhost:19530 or https://host:443
    connection_args:
      user: "developer"
      password: "${MILVUS_PASSWORD}"
      db_name: "default"
    embedding_model: nim_embedder
    use_async_client: true
```

## Training Data

Training data is defined in `training_db_schema.py` and is used when `train_on_startup=true`.

### DDL (Data Definition Language)

Provide table schemas to help Vanna understand your database structure in `VANNA_TRAINING_DDL`:

```python
VANNA_TRAINING_DDL: list[str] = [
    "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(100), created_at TIMESTAMP)",
    "CREATE TABLE orders (id INT PRIMARY KEY, user_id INT, total DECIMAL(10,2))",
]
```

### Documentation

Add contextual information about your data in `VANNA_TRAINING_DOCUMENTATION`:

```python
VANNA_TRAINING_DOCUMENTATION: list[str] = [
    "The users table contains customer information. The created_at field shows when they signed up.",
    "Orders table tracks all purchases. The total field is in USD.",
]
```

### Examples (Few-Shot Learning)

Provide question-SQL pairs for better accuracy in `VANNA_TRAINING_EXAMPLES`:

```python
VANNA_TRAINING_EXAMPLES: list[dict[str, str]] = [
    {
        "question": "Who are our top 10 customers by revenue?",
        "sql": "SELECT u.name, SUM(o.total) as revenue FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.id ORDER BY revenue DESC LIMIT 10",
    },
    {
        "question": "How many new users signed up last month?",
        "sql": "SELECT COUNT(*) FROM users WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')",
    },
]
```

### Active Tables (for Auto-Training)

When `auto_training=true`, specify which tables to extract DDL from in `VANNA_ACTIVE_TABLES`:

```python
VANNA_ACTIVE_TABLES = ['catalog.schema.table_a', 'catalog.schema.table_b']
```

## Advanced Usage

### Multi-Step Query Planning

The ReWOO agent automatically plans a two-step workflow:
1. Generate SQL from natural language using `text2sql`
2. Execute the SQL using `execute_db_query`

You can customize the planning and solving prompts:

```yaml
workflow:
  _type: rewoo_agent
  tool_names: [text2sql, execute_db_query]
  llm_name: nim_llm
  tool_call_max_retries: 3
  additional_planner_instructions: |
    When generating SQL queries, prioritize performance and accuracy.
    Always plan to verify the SQL before execution.
  additional_solver_instructions: |
    Format the final results in a clear, user-friendly manner.
```

For alternative agent types, for example ReAct for multi-turn conversations:

```yaml
workflow:
  _type: react_agent
  tool_names: [text2sql, execute_db_query]
  llm_name: nim_llm
  max_history: 10
```

### Custom Prompts

Customize the system prompt for domain-specific SQL generation:

```yaml
text2sql:
  initial_prompt: |
    You are an expert in supply chain analytics using Databricks SQL.
    Generate queries that follow these conventions:
    - Use CTE (WITH clauses) for complex queries
    - Always include meaningful column aliases
    - Use QUALIFY for deduplication when appropriate
```

### Streaming Responses

Access streaming progress in your application:

```python
from nat.core import Workflow

workflow = Workflow.from_config("text2sql_config.yml")

async for update in workflow.stream("How many customers do we have?"):
    if update["type"] == "status":
        print(f"Status: {update['message']}")
    elif update["type"] == "result":
        print(f"Result: {update}")
```

## Production Considerations

### Security

- **Environment Variables**: Store credentials in environment variables, not in config files
- **Database Permissions**: Use read-only database users for query execution
- **Query Validation**: Review generated SQL before execution in production
- **Connection Pooling**: Configure connection limits for high-traffic scenarios

### Performance

- **Milvus Indexing**: Use appropriate index types for your vector dimensions
- **Result Limits**: Set `max_rows` to prevent large result sets
- **Caching**: Consider caching frequent queries
- **Connection Reuse**: Vanna maintains a singleton instance for efficiency

### Monitoring

Enable telemetry for observability:

```yaml
general:
  telemetry:
    tracing:
      phoenix:
        _type: phoenix
        endpoint: "http://localhost:6006"
    logging:
      console:
        _type: console
        level: INFO
```

Other features include:
- Full integration with the NeMo Agent toolkit intermediate step tracking system
- Better UI Display - Front-ends can now properly render intermediate steps
- Parent Tracking - Each function call has a `parent_id` to group related steps

## Troubleshooting

### Connection Issues

**Milvus connection failed:**
```text
Error: Failed to connect to Milvus
```
- Verify Milvus is running: `docker ps | grep milvus`
- Check host and port configuration
- Verify TLS settings match your Milvus deployment

**Database connection failed:**
```text
Error: Failed to connect to database
```
- Verify credentials and connection parameters
- Check network connectivity
- For Databricks, ensure HTTP path format is correct

### SQL Generation Issues

**Poor quality SQL:**
- Add more training examples similar to your use case (aim for 20+)
- Provide comprehensive DDL with column descriptions
- Add documentation about business logic
- Increase `n_results` to retrieve more examples

**SQL execution errors:**
- Enable `execute_sql: false` to review queries before execution
- Verify catalog and schema names

**No training data found:**
- Vanna needs examples to work. Set `train_on_startup: true` and add at least 3-5 training examples in `training_db_schema.py`
- Or use `auto_training: true` to automatically generate training data from your database

### Known Limitations

**LLM Limitations**:
- The `llama-3.1-70b-instruct` model does not always strictly follow instructions to output in the expected JSON format, which can cause parsing issues. A parsing fallback mechanism has been implemented to handle these cases.
- To ensure optimal performance and consistent JSON output formatting, we recommend using reasoning models in the configuration. These models demonstrate better instruction-following capabilities and reliably produce output in the expected format.

**Database Privileges**:
- This package provides text-to-SQL functionality without built-in guardrails. To prevent destructive operations, always configure the database connection with read-only privileges.

## Additional Resources

For more information:
- [Writing Custom Functions](../../extend/custom-components/custom-functions/functions.md) - Learn how to create your own functions
- [Workflow Configuration](../../build-workflows/workflow-configuration.md) - Complete configuration reference
- [Contributing Guidelines](../../resources/contributing/index.md) - How to contribute to the NeMo Agent toolkit
- [Support](../../resources/support.md) - Get help and support
