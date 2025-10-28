# nvidia-nat-vanna

Production-ready text-to-SQL integration for NeMo Agent Toolkit (NAT) using the Vanna framework with Databricks support and vector-based few-shot learning.

## Features

- **Text-to-SQL Generation**: Convert natural language questions to SQL queries using AI
- **Databricks Support**: Optimized for Databricks SQL warehouses and compute clusters
- **Vector Store Integration**: Milvus-based similarity search for few-shot learning
- **Streaming Support**: Real-time progress updates during SQL generation
- **Database Execution**: Optional query execution with result formatting
- **Customizable**: Flexible configuration for prompts, examples, and database connections

## Installation

```bash
uv venv --python 3.12
uv pip install -e packages/nvidia_nat_vanna
```

### Required Dependencies

The Databricks SQL connector is automatically installed with the package:

```bash
# Already included in nvidia-nat-vanna dependencies
# databricks-sql-connector~=4.0.5
```

## Quick Start

### Prerequisites

- Python 3.11+
- NVIDIA API Key from [NVIDIA API Catalog](https://build.nvidia.com)
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
DB_HOST=your-workspace.cloud.databricks.com
DB_PASSWORD=dapi-xxx
HTTP_PATH=/sql/1.0/warehouses/abc123

# Milvus
MILVUS_URI=http://localhost:19530
MILVUS_PASSWORD=your-password
```

### 3. Create Workflow Configuration

Create `text2sql_config.yml`:

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
    llm_name: my_llm
    embedder_name: my_embedder

    # Database config
    database_type: databricks
    db_host: "${DB_HOST}"
    db_password: "${DB_PASSWORD}"
    http_path: "${HTTP_PATH}"
    db_catalog: main
    db_schema: default

    # Settings
    execute_sql: false
    train_on_startup: true
    n_results: 5
    milvus_search_limit: 1000

    # Training data
    training_ddl:
      - "CREATE TABLE customers (id INT, name VARCHAR(100), email VARCHAR(100))"
      - "CREATE TABLE orders (id INT, customer_id INT, amount DECIMAL(10,2))"

    training_examples:
      - question: "How many customers do we have?"
        sql: "SELECT COUNT(*) FROM customers"
      - question: "What is the total revenue?"
        sql: "SELECT SUM(amount) FROM orders"

  execute_db_query:
    _type: execute_db_query
    database_type: databricks
    db_host: "${DB_HOST}"
    db_password: "${DB_PASSWORD}"
    http_path: "${HTTP_PATH}"
    db_catalog: main
    db_schema: default
    max_rows: 100

llms:
  my_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    api_key: "${NVIDIA_API_KEY}"
    base_url: https://integrate.api.nvidia.com/v1
    temperature: 0.0

embedders:
  my_embedder:
    _type: nim
    model_name: nvidia/llama-3.2-nv-embedqa-1b-v2
    api_key: "${NVIDIA_API_KEY}"
    base_url: https://integrate.api.nvidia.com/v1

workflow:
  _type: rewoo_agent
  tool_names: [text2sql, execute_db_query]
  llm_name: my_llm
  tool_call_max_retries: 3
```

### 4. Run the Workflow

```bash
# Using NAT CLI
uv run nat run --config_file packages/nvidia_nat_vanna/text2sql_config.yml --input "Retrieve the total number of customers."

# Or programmatically
```

Python code:
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
```
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
| `llm_name` | str | LLM reference for SQL generation | Required |
| `embedder_name` | str | Embedder reference for vector ops | Required |
| `database_type` | str | Database type (must be 'databricks') | "databricks" |
| `db_host` | str | Database host (Databricks server hostname) | null |
| `db_password` | str | Database password (Databricks access token) | null |
| `http_path` | str | HTTP path for database connection | null |
| `db_catalog` | str | Database catalog | null |
| `db_schema` | str | Database schema | null |
| `execute_sql` | bool | Execute SQL or just return query | false |
| `allow_llm_to_see_data` | bool | Allow intermediate queries | false |
| `train_on_startup` | bool | Train Vanna on startup | false |
| `training_ddl` | list[str] | DDL statements for training | null |
| `training_examples` | list[dict] | Question-SQL pairs | null |
| `training_documentation` | list[str] | Contextual information | null |
| `initial_prompt` | str | Custom system prompt | null |
| `n_results` | int | Number of similar examples | 5 |
| `sql_collection` | str | Milvus collection name for SQL examples | "vanna_sql" |
| `ddl_collection` | str | Milvus collection name for DDL | "vanna_ddl" |
| `doc_collection` | str | Milvus collection name for documentation | "vanna_documentation" |
| `milvus_search_limit` | int | Maximum limit for vector search operations | 1000 |
| `reasoning_models` | set[str] | Models requiring think tag removal | See below |
| `chat_models` | set[str] | Models using standard response handling | See below |

**Default reasoning models**: `nvidia/llama-3.1-nemotron-ultra-253b-v1`, `nvidia/llama-3.3-nemotron-super-49b-v1.5`, `deepseek-ai/deepseek-v3.1`, `deepseek-ai/deepseek-r1`

**Default chat models**: `meta/llama-3.1-70b-instruct`

#### Understanding `train_on_startup`

The `train_on_startup` parameter controls whether Vanna initializes and loads training data when the workflow starts:

- **`true`**: Automatically creates Milvus collections with names specified by `sql_collection`, `ddl_collection`, and `doc_collection` parameters (defaults: "vanna_sql", "vanna_ddl", "vanna_documentation") and ingests all training data (`training_ddl`, `training_examples`, `training_documentation`) during workflow initialization. This ensures the vector store is populated and ready for similarity search before the first query is processed. Use this setting when you want to ensure fresh training data is loaded each time the workflow starts.

- **`false`** (default): Skips automatic collection creation and training data ingestion. The workflow assumes Milvus collections already exist and contain previously trained data. Use this setting in production environments where training data is already loaded.

### Database Configuration

**Databricks:**
```yaml
database_type: databricks
db_host: "your-workspace.cloud.databricks.com"
db_password: "${DB_PASSWORD}"  # Databricks access token
http_path: "/sql/1.0/warehouses/abc123"
db_catalog: "main"
db_schema: "default"
```

**Note**: Only Databricks is supported. The configuration requires:
- `db_host`: Your Databricks workspace URL (server hostname)
- `db_password`: Databricks personal access token or service principal token
- `http_path`: Path to your SQL warehouse or compute cluster
- `db_catalog`: Optional catalog name (defaults to "main")
- `db_schema`: Optional schema name (defaults to "default")

### Execute DB Query Function

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `database_type` | str | Database type (must be 'databricks') | "databricks" |
| `db_host` | str | Database host (Databricks server hostname) | Required |
| `db_password` | str | Database password (Databricks access token) | Required |
| `http_path` | str | HTTP path for connection (Databricks) | Required |
| `max_rows` | int | Maximum rows to return | 100 |
| `db_catalog` | str | Database catalog | null |
| `db_schema` | str | Database schema | null |

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
    embedding_model: my_embedder
    use_async_client: true
```

## Training Data

### DDL (Data Definition Language)

Provide table schemas to help Vanna understand your database structure:

```yaml
training_ddl:
  - "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(100), created_at TIMESTAMP)"
  - "CREATE TABLE orders (id INT PRIMARY KEY, user_id INT, total DECIMAL(10,2))"
```

### Documentation

Add contextual information about your data:

```yaml
training_documentation:
  - "The users table contains customer information. The created_at field shows when they signed up."
  - "Orders table tracks all purchases. The total field is in USD."
```

### Examples (Few-Shot Learning)

Provide question-SQL pairs for better accuracy:

```yaml
training_examples:
  - question: "Who are our top 10 customers by revenue?"
    sql: "SELECT u.name, SUM(o.total) as revenue FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.id ORDER BY revenue DESC LIMIT 10"
  - question: "How many new users signed up last month?"
    sql: "SELECT COUNT(*) FROM users WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')"
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
  llm_name: my_llm
  tool_call_max_retries: 3
  additional_planner_instructions: |
    When generating SQL queries, prioritize performance and accuracy.
    Always plan to verify the SQL before execution.
  additional_solver_instructions: |
    Format the final results in a clear, user-friendly manner.
```

For alternative agent types (such as ReAct for multi-turn conversations):

```yaml
workflow:
  _type: react_agent
  tool_names: [text2sql, execute_db_query]
  llm_name: my_llm
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
- Full integration with NAT's intermediate step tracking system
- Better UI Display - Front-ends can now properly render intermediate steps
- Parent Tracking - Each function call has a parent_id to group related steps

## Troubleshooting

### Connection Issues

**Milvus connection failed:**
```
Error: Failed to connect to Milvus
```
- Verify Milvus is running: `docker ps | grep milvus`
- Check host/port configuration
- Verify TLS settings match your Milvus deployment

**Database connection failed:**
```
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
- Vanna needs examples to work. Add at least 3-5 training examples in your config

## Package Structure

```
nvidia_nat_vanna/
├── pyproject.toml                      # Package metadata and dependencies
├── README.md                           # This file
├── src/
│   └── nat/
│       └── plugins/
│           └── vanna/
│               ├── register.py         # NAT component registration
│               ├── text2sql.py         # Text-to-SQL function
│               ├── execute_db_query.py # Query execution function
│               ├── vanna_utils.py      # Vanna framework integration
│               ├── db_utils.py         # Database utilities
│               └── milvus_utils.py     # Milvus client utilities
```

## Contributing

Contributions are welcome! Please see the main NAT repository for contribution guidelines.

## License

This package is part of the NVIDIA NeMo Agent Toolkit and follows the same license.

## Support

For questions and support:
- GitHub Issues: https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues
- Documentation: https://nvidia.github.io/NeMo-Agent-Toolkit/
- Discord: Join the NAT community Discord
