# SQL Retriever

The SQL Retriever is a powerful component of the NeMo Agent Toolkit that enables natural language to SQL conversion and database querying. It uses [Vanna AI](https://vanna.ai/) with NVIDIA NIM services to convert user questions in plain English into SQL queries and execute them against your database.

## Features

- **Natural Language to SQL**: Convert plain English questions to SQL queries
- **Multiple Database Support**: 
  - SQLite (file-based databases)
  - PostgreSQL
  - Generic SQL databases via SQLAlchemy (MySQL, SQL Server, Oracle, etc.)
- **NVIDIA NIM Integration**: Leverage NVIDIA's language models and embeddings
- **Training Data Support**: Customize SQL generation with domain-specific training data
- **Vector Store**: Uses ChromaDB for efficient storage and retrieval of training examples

## Installation

### Basic Installation

The SQL Retriever is included in the base NeMo Agent Toolkit installation:

```bash
pip install nvidia-nat
```

### PostgreSQL Support

If you need PostgreSQL support, install the optional dependency:

```bash
pip install nvidia-nat[postgres]
```

### Generic SQL Database Support

For other SQL databases (MySQL, SQL Server, Oracle, etc.), you'll need to install the appropriate SQLAlchemy driver:

```bash
# MySQL
pip install pymysql

# SQL Server
pip install pyodbc

# Oracle
pip install cx_oracle
```

## Quick Start

### Configuration

Create a YAML configuration file (e.g., `config.yml`):

```yaml
llms:
  - name: nim_llm
    type: nim
    model_name: meta/llama-3.1-70b-instruct
    base_url: https://integrate.api.nvidia.com/v1

embedders:
  - name: nim_embeddings
    type: nim
    model_name: nvidia/llama-3.2-nv-embedqa-1b-v2
    base_url: https://integrate.api.nvidia.com/v1

retrievers:
  - name: sql_retriever
    type: sql_retriever
    llm_name: nim_llm
    embedding_name: nim_embeddings
    vector_store_path: ./vanna_vector_store
    db_connection_string: ./database.db
    db_type: sqlite
    training_data_path: ./training_data.yaml
    max_results: 100
```

### Training Data

Create a training data YAML file (e.g., `training_data.yaml`):

```yaml
ddl:
  - |
    CREATE TABLE products (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        price REAL,
        category TEXT
    );

documentation:
  - |
    The products table contains information about products.
    - id: Unique product identifier
    - name: Product name
    - price: Product price in USD
    - category: Product category

sql:
  - question: "What are the most expensive products?"
    sql: |
      SELECT name, price 
      FROM products 
      ORDER BY price DESC 
      LIMIT 10;
  
  - question: "How many products are in each category?"
    sql: |
      SELECT category, COUNT(*) as count
      FROM products
      GROUP BY category;
```

### Usage with NAT

```python
from nat.builder.builder import Builder

# Create builder from config
builder = Builder.from_config_file("config.yml")

# Get the SQL retriever
retriever = await builder.get_retriever("sql_retriever")

# Query the database
results = await retriever.search("What are the top 10 most expensive products?")

# Access the results
for doc in results.results:
    print(doc.page_content)  # JSON string of results
    print(doc.metadata)  # Metadata including SQL query
```

## Database Connection Strings

### SQLite

```yaml
db_type: sqlite
db_connection_string: /path/to/database.db
```

### PostgreSQL

```yaml
db_type: postgres
db_connection_string: postgresql://username:password@host:port/database
```

### MySQL

```yaml
db_type: sql
db_connection_string: mysql+pymysql://username:password@host:port/database
```

### SQL Server

```yaml
db_type: sql
db_connection_string: mssql+pyodbc://username:password@host:port/database?driver=ODBC+Driver+17+for+SQL+Server
```

### Oracle

```yaml
db_type: sql
db_connection_string: oracle+cx_oracle://username:password@host:port/?service_name=service
```

## Training Data Format

The training data YAML file supports three types of training data:

### 1. DDL Statements

Define your database schema:

```yaml
ddl:
  - CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(100));
  - CREATE TABLE orders (id INT PRIMARY KEY, user_id INT, total DECIMAL);
```

### 2. Documentation

Provide context about your data:

```yaml
documentation:
  - |
    The users table contains customer information.
    The orders table tracks customer purchases.
```

### 3. Question-SQL Pairs

Provide example questions and their corresponding SQL:

```yaml
sql:
  - question: "Who are our top customers?"
    sql: |
      SELECT u.name, SUM(o.total) as total_spent
      FROM users u
      JOIN orders o ON u.id = o.user_id
      GROUP BY u.id, u.name
      ORDER BY total_spent DESC;
```

## Advanced Usage

### Custom NVIDIA API Key

You can provide a custom API key instead of using the environment variable:

```yaml
retrievers:
  - name: sql_retriever
    type: sql_retriever
    nvidia_api_key: nvapi-your-key-here
    # ... other config
```

### Limiting Results

Control the maximum number of results returned:

```python
# In configuration
max_results: 50

# Or at query time
results = await retriever.search(
    "What are the products?",
    top_k=20
)
```

### Accessing Generated SQL

The generated SQL query is included in the metadata:

```python
results = await retriever.search("Show me all products")
sql_query = results.results[0].metadata["sql"]
print(f"Generated SQL: {sql_query}")
```

## Architecture

The SQL Retriever consists of three main components:

1. **VannaManager**: Manages Vanna instances with singleton pattern for efficiency
2. **Vanna Utilities**: Integrates NVIDIA NIM services with Vanna
3. **SQLRetriever**: Implements the NAT Retriever interface

```
┌─────────────────┐
│  SQLRetriever   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ VannaManager    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  NIMVanna       │
│  (Vanna + NIM)  │
└─────────────────┘
```

## Troubleshooting

### ChromaDB Issues

If you encounter ChromaDB-related errors, ensure your vector store directory has proper permissions:

```bash
mkdir -p ./vanna_vector_store
chmod 755 ./vanna_vector_store
```

### PostgreSQL Connection Issues

Make sure `psycopg2-binary` is installed:

```bash
pip install nvidia-nat[postgres]
```

### Training Data Not Loading

Check that your training data YAML file is valid and the path is correct. The retriever will log warnings if training data cannot be loaded.

### Empty Results

If you're getting empty results, check:
1. Your database connection string is correct
2. The database contains data
3. The generated SQL query is valid (check metadata)

## Environment Variables

- `NVIDIA_API_KEY`: Your NVIDIA API key for NIM services

```bash
export NVIDIA_API_KEY=nvapi-your-key-here
```

## Performance Tips

1. **Training Data**: Provide comprehensive training data for better SQL generation
2. **Vector Store**: The vector store is persistent - once trained, it doesn't need retraining
3. **Result Limits**: Use `max_results` to prevent overwhelming responses
4. **Database Indexes**: Ensure your database has proper indexes for query performance

## Examples

See the NeMo Agent Toolkit examples directory for complete working examples:

- Basic SQL Retriever example
- Multi-database example
- Custom training data example

## License

SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

