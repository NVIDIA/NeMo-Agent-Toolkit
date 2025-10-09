# Text2SQL Standalone MCP Server Example

This is a standalone example of the Text2SQL functionality designed for MCP server deployment and load testing. It demonstrates how to serve a text-to-SQL tool via the Model Context Protocol (MCP) without requiring the full `talk-to-supply-chain-tools` package.

## Overview

This example provides:
- **Standalone Text2SQL function** that converts natural language questions to SQL queries
- **MCP server deployment** for easy integration with Claude Desktop and other MCP clients
- **Load testing capabilities** for memory profiling and performance testing
- **Minimal dependencies** for easier debugging and profiling

The standalone version uses:
- Vanna AI framework for text-to-SQL conversion
- NVIDIA NIM for LLM inference
- Milvus vector database for storing DDL, documentation, and few-shot examples
- Databricks for SQL execution (optional)

## Architecture

```
examples/text2sql/
├── src/text2sql/
│   ├── functions/           # Text2SQL implementation
│   │   ├── text2sql_standalone.py  # Main function registration
│   │   └── sql_utils.py            # Vanna integration & utilities
│   ├── utils/               # Utility modules
│   │   ├── constant.py             # Constants
│   │   ├── feature_flag.py         # Feature flags
│   │   ├── milvus_utils.py         # Milvus client utilities
│   │   ├── db_utils.py             # Database utilities
│   │   └── db_schema.py            # Database schema & examples
│   ├── resources/           # Resource files
│   │   └── followup_resources.py   # Follow-up question resources
│   ├── configs/             # Configuration files
│   │   └── config_text2sql_mcp.yml # MCP server config
│   └── register.py          # Component registration
├── pyproject.toml           # Project dependencies
├── env.example              # Environment variable template
└── README.md               # This file
```

## Prerequisites

- Python 3.11 or higher
- NVIDIA API key (for LLM and embeddings)
- Milvus instance (cloud or local)
- Databricks account (optional, only needed if executing SQL)

## Setup Instructions

### 1. Clone and Navigate to Example

```bash
cd examples/text2sql
```

### 2. Set Up Environment Variables

Copy the example environment file and fill in your credentials:

```bash
cp env.example .env
```

Edit `.env` with your credentials:

```bash
# Required: NVIDIA API Key
NVIDIA_API_KEY=your_nvidia_api_key_here

# Required for remote Milvus
MILVUS_HOST=your-milvus-host.zillizcloud.com
MILVUS_PORT=19530
MILVUS_USERNAME=your_milvus_username
MILVUS_PASSWORD=your_milvus_password

# Optional: Databricks (only if execute_sql=true)
DATABRICKS_SERVER_HOSTNAME=your_databricks_hostname.cloud.databricks.com
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/your_warehouse_id
DATABRICKS_ACCESS_TOKEN=your_databricks_access_token
```

### 3. Install Dependencies

Using uv (recommended):

```bash
# From the examples/text2sql directory
uv pip install -e .
```

Or using pip:

```bash
pip install -e .
```

### 4. Train Vanna (First Time Only)

On first run, you need to populate the Milvus vector database with training data (DDL, documentation, and examples). Edit `src/text2sql/configs/config_text2sql_mcp.yml`:

```yaml
functions:
  text2sql_standalone:
    train_on_startup: true  # Set to true for first run
    # ... other settings
```

Then run the workflow once to train:

```bash
nat-cli run --workflow-config src/text2sql/configs/config_text2sql_mcp.yml
```

After training completes, set `train_on_startup: false` for subsequent runs.

### 5. Run as MCP Server

To serve the text2sql function via MCP:

```bash
nat-cli serve mcp --workflow-config src/text2sql/configs/config_text2sql_mcp.yml
```

The MCP server will start and expose the `text2sql_standalone` tool.

## Configuration

The main configuration file is `src/text2sql/configs/config_text2sql_mcp.yml`. Key settings:

### Function Configuration

```yaml
functions:
  text2sql_standalone:
    _type: text2sql_standalone
    llm_name: nim_llm
    embedder_name: nim_embedder
    train_on_startup: false           # Set to true for first run
    execute_sql: false                # If true, executes SQL on Databricks
    authorize: false                  # If true, requires Bearer token
    enable_followup_questions: false  # Generate follow-up questions
    vanna_remote: true                # Use remote Milvus (true) or local (false)
    training_analysis_filter: ["pbr", "supply_gap"]  # Filter training examples
```

### LLM Configuration

```yaml
llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    temperature: 0.0
    max_tokens: 2048
    timeout: 60.0
```

### Embedder Configuration

```yaml
embedders:
  nim_embedder:
    _type: nvidia_ai_endpoints
    model_name: nvidia/nv-embedqa-e5-v5
    truncate: END
```

## Usage Examples

### Using with Claude Desktop

Add to your Claude Desktop MCP configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "text2sql": {
      "command": "nat-cli",
      "args": [
        "serve",
        "mcp",
        "--workflow-config",
        "/path/to/examples/text2sql/src/text2sql/configs/config_text2sql_mcp.yml"
      ],
      "env": {
        "NVIDIA_API_KEY": "your_key_here",
        "MILVUS_HOST": "your-host.zillizcloud.com",
        "MILVUS_PORT": "19530",
        "MILVUS_USERNAME": "your_username",
        "MILVUS_PASSWORD": "your_password"
      }
    }
  }
}
```

Then in Claude Desktop:
- Ask: "Convert this to SQL: Show me the top 10 components with highest shortages"
- Claude will use the `text2sql_standalone` tool to generate the SQL query

### Using Programmatically

```python
from nat.builder import Builder
from text2sql.functions.text2sql_standalone import Text2sqlStandaloneConfig

# Create config
config = Text2sqlStandaloneConfig(
    llm_name="nim_llm",
    embedder_name="nim_embedder",
    train_on_startup=False,
    execute_sql=False,
    vanna_remote=True,
    milvus_host="your-host.zillizcloud.com",
    milvus_port="19530",
    milvus_user="your_username",
    milvus_db_name="default"
)

# Initialize builder and run
builder = Builder()
async for result in text2sql_standalone(config, builder):
    # Use the function
    question = "Show me parts with shortages greater than 100"
    async for update in result.stream_fn(question):
        print(update)
```

### Using via CLI

```bash
# Run interactively
nat-cli run --workflow-config src/text2sql/configs/config_text2sql_mcp.yml

# Then ask questions:
# "Show me the top 10 components with highest shortages"
# "What are the lead times for NVPN 316-0899-000?"
```

## Load Testing

This standalone example includes comprehensive load testing tools for memory profiling and leak detection.

### Quick Start

Run the integrated test suite that automatically starts the server, monitors memory, and runs load tests:

```bash
cd examples/text2sql
python run_text2sql_memory_leak_test.py
```

This will simulate 40 concurrent users making realistic text2sql queries across 3 rounds and detect potential memory leaks.

### Manual Load Testing

For more control, run components separately:

1. **Start the MCP server:**

```bash
nat mcp serve --config_file configs/config_text2sql_mcp.yml
```

2. **Run load tests** (in another terminal):

```bash
python load_test_text2sql.py \
  --url http://localhost:9901/mcp \
  --users 40 \
  --calls 10 \
  --rounds 3
```

### Customization

Customize test parameters:

```bash
python run_text2sql_memory_leak_test.py \
  --users 50 \              # Number of concurrent users
  --calls 20 \              # Queries per user per round
  --rounds 5 \              # Number of test rounds
  --delay 15.0              # Delay between rounds (seconds)
```

### What Gets Tested

The load test uses realistic supply chain queries:
- Shortage analysis queries
- Lead time inquiries
- Inventory status checks
- Build request queries
- Material cost analysis
- CM site breakdowns
- Trend forecasts

See [LOAD_TESTING.md](LOAD_TESTING.md) for detailed documentation, troubleshooting, and performance analysis.

The simplified standalone version makes it easier to:
- Profile memory usage with real-time monitoring
- Detect memory leaks under concurrent load
- Measure performance and throughput
- Debug issues in isolation
- Simulate production workloads

## Features

### Text-to-SQL Conversion

- Converts natural language questions to SQL queries
- Supports complex supply chain queries
- Uses few-shot learning with domain-specific examples
- Handles multiple database tables (PBR, DEMAND_DLT, etc.)

### Optional SQL Execution

When `execute_sql: true`, the function will:
- Generate SQL from the question
- Execute it on Databricks
- Return results as JSON with row/column information
- Handle errors with retry logic

### Follow-up Questions

When `enable_followup_questions: true`, generates contextual follow-up questions based on:
- Query results
- Table structure
- Domain-specific use cases

### Analysis Type Filtering

The `training_analysis_filter` parameter allows filtering training examples:
- `["pbr"]` - Only prototype build request examples
- `["supply_gap"]` - Only supply gap analysis examples
- `["pbr", "supply_gap"]` - Both types

## Troubleshooting

### Milvus Connection Issues

If you see connection errors:
1. Verify your Milvus credentials in `.env`
2. Check that `MILVUS_PASSWORD` is set (it's required for cloud instances)
3. Ensure the host doesn't include `https://` prefix

### Training Failures

If training fails:
1. Check your NVIDIA API key is valid
2. Verify embedder model is accessible
3. Review logs for specific errors

### SQL Generation Issues

If SQL generation is poor:
1. Ensure training was completed successfully
2. Check that examples match your use case
3. Consider adjusting `training_analysis_filter`
4. Review the few-shot examples in `utils/db_schema.py`

### Memory Issues During Load Testing

If you encounter memory issues:
1. Monitor with: `ps aux | grep nat-cli`
2. Check MCP server logs for errors
3. Adjust batch sizes in your load test
4. Consider increasing timeout values

## Development

### Adding New Examples

To add new few-shot examples, edit `src/text2sql/utils/db_schema.py`:

```python
PBR_EXAMPLES = [
    {
        "Query": "Your natural language question",
        "SQL": "SELECT ... FROM ...",
        "metadata": {"analysis": "pbr"}
    },
    # ... more examples
]
```

### Modifying Database Schema

Update table schemas in `src/text2sql/utils/db_schema.py`:

```python
TABLES = [
    {
        "name": "table_name",
        "description": "Table description",
        "schema": [
            {
                "field": "column_name",
                "type": "data_type",
                "description": "Column description"
            },
            # ... more columns
        ]
    }
]
```

### Customizing Prompts

Modify system prompts in `src/text2sql/utils/db_schema.py`:
- `INSTRUCTION_PROMPT` - Main system prompt
- `CONCEPTS` - Domain-specific terminology
- `GUIDELINES` - SQL generation guidelines

## Differences from Full talk-to-supply-chain-tools

This standalone version:
- ✅ No authentication/authorization (simpler for testing)
- ✅ Minimal dependencies (easier to debug)
- ✅ Focused on SQL generation (not execution by default)
- ✅ Optimized for MCP deployment
- ❌ No multi-tool workflows (single tool only)
- ❌ No advanced features (chart generation, summarization, etc.)

## References

- [NeMo Agent Toolkit Documentation](../../docs/)
- [Vanna AI Framework](https://github.com/vanna-ai/vanna)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [NVIDIA NIM](https://developer.nvidia.com/nim)
- [Milvus Vector Database](https://milvus.io/)

## Support

For issues specific to this example:
1. Check the troubleshooting section above
2. Review the logs for detailed error messages
3. Verify your environment variables are set correctly
4. Ensure all dependencies are installed

For general NeMo Agent Toolkit issues, refer to the main repository documentation.
