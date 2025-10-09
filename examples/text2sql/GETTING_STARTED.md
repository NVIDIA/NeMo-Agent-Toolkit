# Getting Started with Text2SQL Example

## Quick Setup (Local Milvus - Easiest)

### Step 1: Install the Example

```bash
cd examples/text2sql
uv pip install -e .
```

### Step 2: Start Local Milvus (Docker)

```bash
# Pull and run Milvus Lite
docker run -d --name milvus \
  -p 19530:19530 \
  -p 9091:9091 \
  -v $(pwd)/milvus_data:/var/lib/milvus \
  milvusdb/milvus:latest
```

Verify it's running:
```bash
docker ps | grep milvus
```

### Step 3: Create Environment File

Create `.env` file:

```bash
cat > .env << 'EOF'
# NVIDIA API Key (Required)
NVIDIA_API_KEY=nvapi-your-key-here

# Local Milvus (using Docker container)
# No additional config needed for local!
EOF
```

**Get your NVIDIA API key:**
1. Go to https://build.nvidia.com
2. Sign in
3. Click on your profile → "Get API Key"
4. Copy the key and paste it in `.env`

### Step 4: Configure for Local Milvus

Edit `src/text2sql/configs/config_text2sql_mcp.yml`:

```yaml
functions:
  text2sql_standalone:
    # ... other settings ...
    vanna_remote: false  # ← Change to false for local Milvus
    train_on_startup: true  # ← Set to true for first run
    execute_sql: false  # ← Keep false (no Databricks needed)
```

### Step 5: Train Vanna (First Time Only)

This populates the vector database with examples:

```bash
nat-cli run --workflow-config src/text2sql/configs/config_text2sql_mcp.yml
```

You should see:
```
Training vanna
Added to sql collection: ...
Added to sql collection: ...
...
```

**After training completes**, edit the config again:

```yaml
functions:
  text2sql_standalone:
    train_on_startup: false  # ← Change back to false
```

### Step 6: Test It!

#### Interactive CLI Mode:

```bash
nat-cli run --workflow-config src/text2sql/configs/config_text2sql_mcp.yml
```

Then ask questions like:
```
> Show me all red items for build id PB-61738
> What are the top 10 components with highest shortages?
> Display components with lead time greater than 50 days
```

#### MCP Server Mode:

```bash
nat-cli serve mcp --workflow-config src/text2sql/configs/config_text2sql_mcp.yml
```

The server will start and you can connect Claude Desktop or other MCP clients to it.

## Example Questions to Try

### Supply Chain Queries:
```
1. "Show me the top 20 NVPNs with highest shortages"
2. "What components have lead time greater than 50 days?"
3. "Display the latest material cost by CM for NVPN 316-0899-000"
4. "Show all consigned parts for build id PB-61738 that are green"
5. "List components without lead time for project E2425"
```

### Build Request Queries:
```
1. "Show me all the components without lead time for build id PB-60506"
2. "Give me a list of components with insufficient quantity for PB-55330"
3. "Show items with shortages more than 50 units for SKU 699-13925-1099-TS3"
```

### Inventory Queries:
```
1. "Show latest demand in next 26 weeks for NVPN 681-24287-0012.A by CM site"
2. "Display components with nettable inventory above 1000 units"
3. "Show supply and demand trend for NVPN 315-1157-000"
```

## Expected Output

When you ask a question, you'll get back SQL:

```json
{
  "sql": "SELECT cm_site_name, nvpn, shortage, total_shortage, ...",
  "explanation": "Generated SQL query for: Show me the top 10...",
  "confidence": "high",
  "method": "vanna"
}
```

If `execute_sql: true` (requires Databricks), you'd also get results:

```json
{
  "sql": "SELECT ...",
  "results": {
    "row_count": 10,
    "columns": ["cm_site_name", "nvpn", "shortage"],
    "data": [...]
  }
}
```

## Using Cloud Milvus (Alternative)

If you prefer cloud Milvus:

1. Sign up at https://cloud.zilliz.com (free tier available)
2. Create a cluster
3. Get credentials (host, port, username, password)
4. Update `.env`:

```bash
NVIDIA_API_KEY=nvapi-your-key-here
MILVUS_HOST=your-cluster.aws-us-west-2.vectordb.zillizcloud.com
MILVUS_PORT=19530
MILVUS_USERNAME=your_username
MILVUS_PASSWORD=your_password
```

5. In config, set `vanna_remote: true`

## Troubleshooting

### "Connection refused" to Milvus

```bash
# Check if Milvus container is running
docker ps | grep milvus

# If not running, start it
docker start milvus

# Check logs
docker logs milvus
```

### "NVIDIA_API_KEY not set"

Make sure your `.env` file is in the `examples/text2sql` directory and the key is correct.

### Poor SQL Quality

The first time you run, make sure training completed successfully. Check that you see messages like:
```
Added to sql collection: ...
```

If training failed, you might need to:
1. Check your NVIDIA API key is valid
2. Ensure Milvus is accessible
3. Try running training again

### Memory Issues

If you encounter memory issues during training or usage:
```bash
# Check available memory
docker stats milvus

# Restart Milvus if needed
docker restart milvus
```

## Next Steps

Once you have it running:

1. **Try different questions** - Explore the example queries above
2. **Add custom examples** - Edit `src/text2sql/utils/db_schema.py` to add your own SQL examples
3. **Integrate with Claude Desktop** - See main README.md for configuration
4. **Enable SQL execution** - Set up Databricks credentials to actually run queries
5. **Load testing** - Use for performance testing and memory profiling

## Cleanup

When done:

```bash
# Stop Milvus
docker stop milvus

# Remove container (keeps data)
docker rm milvus

# Remove data volume (if you want to start fresh)
rm -rf milvus_data/
```

## Architecture

```
┌─────────────┐
│   Your CLI  │
│  or Claude  │
└──────┬──────┘
       │
       v
┌──────────────────┐
│  Text2SQL Tool   │
│   (NAT CLI/MCP)  │
└──────┬───────────┘
       │
       v
┌──────────────────┐      ┌──────────────┐
│   Vanna.ai       │─────>│  NVIDIA NIM  │
│  (SQL Generator) │      │     LLM      │
└──────┬───────────┘      └──────────────┘
       │
       v
┌──────────────────┐
│     Milvus       │
│ (Vector Store)   │
│ - DDL            │
│ - Examples       │
│ - Documentation  │
└──────────────────┘
```

## Resources

- Main README: `README.md` - Comprehensive documentation
- Quick Start: `QUICKSTART.md` - 5-minute guide
- Configuration: `src/text2sql/configs/config_text2sql_mcp.yml`
- Examples: `src/text2sql/utils/db_schema.py` - 80+ SQL examples
