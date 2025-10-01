# SQL Retriever Testing Guide

This directory contains test scripts for the SQL Retriever with different configurations.

## Test Scripts

### `test_mysql_elastic.py` - MySQL + Elasticsearch

Tests the SQL Retriever using:
- **Database**: MySQL Server
- **Vector Store**: Elasticsearch
- **LLM**: NVIDIA NIM
- **Embeddings**: NVIDIA NIM Embeddings

## Prerequisites

### 1. Install Dependencies

```bash
# Core dependencies (should already be installed with NAT)
pip install nvidia-nat

# Additional dependencies for MySQL + Elasticsearch test
pip install pymysql elasticsearch sqlalchemy
```

### 2. Set Up Services

#### MySQL

```bash
# Using Docker
docker run --name test-mysql \
  -e MYSQL_ROOT_PASSWORD=testpassword \
  -e MYSQL_DATABASE=testdb \
  -p 3306:3306 \
  -d mysql:8.0

# Or install MySQL locally
# https://dev.mysql.com/downloads/mysql/
```

#### Elasticsearch

```bash
# Using Docker
docker run --name test-elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -d docker.elastic.co/elasticsearch/elasticsearch:8.11.0

# Verify Elasticsearch is running
curl http://localhost:9200
```

### 3. Set NVIDIA API Key

```bash
export NVIDIA_API_KEY="nvapi-your-key-here"
```

## Running Tests

### Basic Test (with defaults)

```bash
cd /path/to/NeMo-Agent-Toolkit/src/nat/retriever/sql_retriever

python test_mysql_elastic.py
```

**Default Configuration:**
- MySQL: `localhost:3306`, user: `root`, password: `""`, database: `test`
- Elasticsearch: `http://localhost:9200`, index: `vanna_sql_vectors`
- LLM: `meta/llama-3.1-70b-instruct`
- Embeddings: `nvidia/llama-3.2-nv-embedqa-1b-v2`
- Question: "What are the top 10 best-selling products by quantity?"

### Custom Question

```bash
python test_mysql_elastic.py -q "Show me the top 5 customers by total orders"
```

### Custom MySQL Connection

```bash
python test_mysql_elastic.py \
  --mysql-host localhost \
  --mysql-port 3306 \
  --mysql-user myuser \
  --mysql-password mypassword \
  --mysql-database mydb \
  -q "What are the monthly sales totals?"
```

### Custom Elasticsearch Configuration

```bash
python test_mysql_elastic.py \
  --es-url http://localhost:9200 \
  --es-index my_custom_index \
  -q "Show products by category"
```

### With Training Data

```bash
python test_mysql_elastic.py \
  --training-data test_training_data.yaml \
  -q "Which customers have the highest lifetime value?"
```

### Full Custom Configuration

```bash
python test_mysql_elastic.py \
  --mysql-host localhost \
  --mysql-port 3306 \
  --mysql-user root \
  --mysql-password testpassword \
  --mysql-database testdb \
  --es-url http://localhost:9200 \
  --es-index vanna_test_vectors \
  --llm-model meta/llama-3.1-70b-instruct \
  --embedding-model nvidia/llama-3.2-nv-embedqa-1b-v2 \
  --training-data test_training_data.yaml \
  -q "What are the top selling products this month?"
```

## Training Data

The `test_training_data.yaml` file contains sample training data for an e-commerce schema:
- **DDL**: Table definitions (customers, products, orders, order_items)
- **Documentation**: Schema overview and query patterns
- **SQL Examples**: Common question-SQL pairs

### Training Data Format

```yaml
ddl:
  - CREATE TABLE customers (...);
  - CREATE TABLE products (...);

documentation:
  - "Database contains e-commerce data..."
  - "Common query patterns..."

sql:
  - question: "What are the top products?"
    sql: "SELECT ... FROM products ..."
  - question: "Who are the best customers?"
    sql: "SELECT ... FROM customers ..."
```

## Command Line Options

### MySQL Configuration
- `--mysql-host`: MySQL host (default: localhost)
- `--mysql-port`: MySQL port (default: 3306)
- `--mysql-user`: MySQL username (default: root)
- `--mysql-password`: MySQL password (default: empty)
- `--mysql-database`: Database name (default: test)

### Elasticsearch Configuration
- `--es-url`: Elasticsearch URL (default: http://localhost:9200)
- `--es-index`: Index name (default: vanna_sql_vectors)
- `--es-username`: Username for basic auth (optional)
- `--es-password`: Password for basic auth (optional)
- `--es-api-key`: API key for authentication (optional)

### Model Configuration
- `--llm-model`: LLM model name (default: meta/llama-3.1-70b-instruct)
- `--embedding-model`: Embedding model (default: nvidia/llama-3.2-nv-embedqa-1b-v2)

### Other Options
- `-q, --question`: Natural language question to test
- `--training-data`: Path to YAML training data file

## Expected Output

```
================================================================================
SQL Retriever Test: MySQL + Elasticsearch
================================================================================
✓ NVIDIA_API_KEY is set
✓ Elasticsearch is accessible at http://localhost:9200
✓ MySQL is accessible at localhost:3306/testdb
--------------------------------------------------------------------------------
Configuration:
  LLM Model: meta/llama-3.1-70b-instruct
  Embedding Model: nvidia/llama-3.2-nv-embedqa-1b-v2
  MySQL: localhost:3306/testdb
  Elasticsearch: http://localhost:9200
  ES Index: vanna_sql_vectors
  Training Data: test_training_data.yaml
--------------------------------------------------------------------------------
Initializing NVIDIA embedding function...
Initializing ElasticNIMVanna...
✓ Connected to MySQL database
Index contains 25 documents -> skipping training
================================================================================
Testing SQL Generation
Question: What are the top 10 best-selling products by quantity?
--------------------------------------------------------------------------------
✓ SQL generated successfully:
--------------------------------------------------------------------------------
SELECT 
    p.name,
    p.category,
    SUM(oi.quantity) as total_quantity_sold
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY p.product_id, p.name, p.category
ORDER BY total_quantity_sold DESC
LIMIT 10;
--------------------------------------------------------------------------------
✓ Query executed successfully. Retrieved 10 rows
================================================================================
Results:
================================================================================
         name    category  total_quantity_sold
0   Product A  Electronics                  523
1   Product B    Clothing                  412
...
================================================================================

✓ Test completed successfully!
```

## Troubleshooting

### Cannot connect to Elasticsearch
```
✗ Cannot connect to Elasticsearch at http://localhost:9200
```
**Solution**: Ensure Elasticsearch is running and accessible
```bash
# Check if running
curl http://localhost:9200

# Start Elasticsearch
docker start test-elasticsearch
```

### Cannot connect to MySQL
```
✗ Error connecting to MySQL: (2003, "Can't connect to MySQL server")
```
**Solution**: Check MySQL is running and credentials are correct
```bash
# Test connection
mysql -h localhost -u root -p -e "SHOW DATABASES;"

# Start MySQL
docker start test-mysql
```

### Missing dependencies
```
✗ elasticsearch package not installed
```
**Solution**: Install required packages
```bash
pip install elasticsearch pymysql sqlalchemy
```

### NVIDIA_API_KEY not set
```
✗ NVIDIA_API_KEY environment variable is not set
```
**Solution**: Set your API key
```bash
export NVIDIA_API_KEY="nvapi-your-key-here"
```

## Cleaning Up

### Remove Elasticsearch Index
```bash
curl -X DELETE "http://localhost:9200/vanna_sql_vectors"
```

### Stop Docker Containers
```bash
docker stop test-mysql test-elasticsearch
docker rm test-mysql test-elasticsearch
```

## Next Steps

- Modify `test_training_data.yaml` to match your database schema
- Run the test with your actual MySQL database
- Experiment with different LLM models
- Try different question types and complexity levels

