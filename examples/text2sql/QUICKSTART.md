# Text2SQL Standalone - Quick Start Guide

Get started with the Text2SQL MCP server in 5 minutes!

## Quick Setup

### 1. Install Dependencies

```bash
cd examples/text2sql
uv pip install -e .
```

### 2. Configure Environment

Create a `.env` file:

```bash
cp env.example .env
```

Edit `.env` with your credentials:

```bash
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxx
MILVUS_HOST=your-milvus-instance.zillizcloud.com
MILVUS_PORT=19530
MILVUS_USERNAME=your_username
MILVUS_PASSWORD=your_password
```

### 3. Train on First Run

Edit `src/text2sql/configs/config_text2sql_mcp.yml`:

```yaml
functions:
  text2sql_standalone:
    train_on_startup: true  # ← Set to true
```

Run once to populate Milvus:

```bash
nat-cli run --workflow-config src/text2sql/configs/config_text2sql_mcp.yml
```

### 4. Set train_on_startup to False

After training completes, edit config again:

```yaml
functions:
  text2sql_standalone:
    train_on_startup: false  # ← Set to false
```

### 5. Start MCP Server

```bash
nat-cli serve mcp --workflow-config src/text2sql/configs/config_text2sql_mcp.yml
```

Done! Your MCP server is now running.

## Test It Out

### Option 1: Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "text2sql": {
      "command": "nat-cli",
      "args": [
        "serve",
        "mcp",
        "--workflow-config",
        "/full/path/to/examples/text2sql/src/text2sql/configs/config_text2sql_mcp.yml"
      ],
      "env": {
        "NVIDIA_API_KEY": "your_key",
        "MILVUS_HOST": "your-host.zillizcloud.com",
        "MILVUS_PORT": "19530",
        "MILVUS_USERNAME": "your_username",
        "MILVUS_PASSWORD": "your_password"
      }
    }
  }
}
```

Restart Claude Desktop and ask:
> "Convert to SQL: Show me the top 10 components with highest shortages"

### Option 2: CLI

```bash
nat-cli run --workflow-config src/text2sql/configs/config_text2sql_mcp.yml
```

Then ask questions like:
- "Show me all red items for build id PB-61738"
- "What are the lead times for NVPN 316-0899-000?"
- "Display components with shortages greater than 100"

## Example Questions

Try these sample questions:

### Supply Chain Analysis
- "Show me the top 20 NVPNs with highest shortages"
- "What components have lead time greater than 50 days?"
- "Display safety stock data for NVPN 316-0899-000"

### Build Requests
- "Show all consigned parts for build id PB-61738 that are green"
- "List components without lead time for project E2425"
- "Which items at FXHC have shortages more than 8000?"

### Inventory
- "Show supply and demand trend for NVPN 315-1157-000"
- "What is the latest material cost by CM for NVPN 316-0899-000?"
- "Display items with nettable inventory above 1000 units"

## Configuration Options

### SQL Execution

To execute SQL on Databricks (requires Databricks credentials):

```yaml
functions:
  text2sql_standalone:
    execute_sql: true  # Will execute and return results
    allow_llm_to_see_data: false
```

### Follow-up Questions

To generate follow-up questions:

```yaml
functions:
  text2sql_standalone:
    enable_followup_questions: true
```

### Local Milvus

To use local Milvus instead of cloud:

```yaml
functions:
  text2sql_standalone:
    vanna_remote: false
    # Don't need milvus_host, milvus_port, etc.
```

Set in `.env`:
```bash
SUPPLY_CHAIN_VDB_PATH=./milvus_vanna.db
```

## Troubleshooting

### "Connection refused" error
- Check Milvus credentials in `.env`
- Verify network connectivity to Milvus instance

### "No training data found"
- Run with `train_on_startup: true` first
- Check logs for training completion

### Poor SQL quality
- Ensure training completed successfully
- Review few-shot examples in `src/text2sql/utils/db_schema.py`
- Consider adjusting `training_analysis_filter`

### "NVIDIA_API_KEY not set"
- Check `.env` file exists and has correct key
- Verify key is valid at https://build.nvidia.com

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore configuration options in `src/text2sql/configs/config_text2sql_mcp.yml`
- Add custom examples in `src/text2sql/utils/db_schema.py`
- Set up load testing for performance evaluation

## Need Help?

- Check the main [README.md](README.md) for detailed troubleshooting
- Review logs for specific error messages
- Verify all environment variables are set correctly
