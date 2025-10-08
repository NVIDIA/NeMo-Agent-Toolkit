# Text-to-SQL MCP Server Load Testing

This directory contains tools for load testing the text2sql function as an MCP server to identify potential memory leaks.

## Overview

The `text2sql_standalone` function is a simplified version of the production `text2sql` function designed for:
- Independent MCP server deployment
- Load testing and memory leak detection
- Minimal dependencies for easier profiling
- No authentication requirements (for testing)
- No SQL execution (only generation)

## Prerequisites

1. **Install dependencies:**
   ```bash
   # Install NeMo Agent Toolkit with MCP support
   pip install -e ".[mcp]"

   # Install the talk-to-supply-chain-tools package
   pip install -e talk-to-supply-chain-tools/
   ```

2. **Set required environment variables:**
   ```bash
   # NVIDIA NIM API Key (required for LLM and embedder)
   export NVIDIA_API_KEY="your-api-key-here"

   # Milvus Cloud configuration (if using remote Milvus)
   export MILVUS_HOST="your-milvus-host.com"
   export MILVUS_PORT="19530"
   export MILVUS_USERNAME="your-username"
   export MILVUS_PASSWORD="your-password"
   ```

## Quick Start

### Option 1: Automated Test (Recommended)

Run the complete test with one command:

```bash
./examples/text2sql/run_text2sql_load_test.sh
```

This script will:
1. Start the MCP server in the background
2. Wait for server to be ready
3. Run the load test with memory monitoring
4. Generate a report
5. Clean up the server

### Option 2: Manual Testing

#### Step 1: Start the MCP Server

```bash
# Start the MCP server on default port 9901
nat mcp serve \
  --config_file examples/text2sql/config_text2sql_mcp.yml \
  --host localhost \
  --port 9901 \
  --log_level INFO
```

The server will:
- Load the `text2sql_standalone` function
- Connect to Milvus vector store
- Expose the function as an MCP tool at `http://localhost:9901/mcp`

#### Step 2: Verify Server is Running

In another terminal:

```bash
# Check health endpoint
curl http://localhost:9901/health

# List available tools
curl http://localhost:9901/debug/tools/list
```

You should see `text2sql_standalone` in the tools list.

#### Step 3: Run Load Test

```bash
# Run a comprehensive load test
python examples/text2sql/text2sql_load_test.py \
  --url http://localhost:9901/mcp \
  --users 20 \
  --calls 10 \
  --rounds 3 \
  --delay 5.0 \
  --verbose
```

Parameters:
- `--users`: Number of concurrent users (default: 20)
- `--calls`: Number of calls per user (default: 10)
- `--rounds`: Number of test rounds (default: 3)
- `--delay`: Delay between rounds in seconds (default: 5.0)
- `--verbose`: Enable debug logging

#### Step 4: Monitor Memory Usage

While the load test is running, monitor memory in another terminal:

```bash
# Monitor memory usage of the MCP server process
python debug_tools/monitor_memory.py --process-name "nat mcp serve"
```

Or use system tools:
```bash
# macOS
ps aux | grep "nat mcp serve"

# Linux
ps aux | grep "nat mcp serve"
top -p $(pgrep -f "nat mcp serve")
```

## Load Test Features

The `text2sql_load_test.py` script simulates realistic usage patterns:

- **Multiple concurrent users**: Simulates 20+ concurrent clients
- **Natural language queries**: Uses realistic supply chain questions
- **Analysis type filtering**: Tests with different analysis types (pbr, supply_gap)
- **Proper MCP protocol**: Uses `nat mcp client` CLI for correct protocol handling
- **Multiple rounds**: Runs multiple rounds to observe memory trends
- **Detailed metrics**: Reports success rate, throughput, and timing

### Sample Questions

The load test uses questions like:
- "Show me top 10 suppliers by revenue"
- "What are the parts with low inventory?"
- "Find suppliers with delivery delays"
- "Show part demand for Q1 2024"
- "List critical shortage items"

## Configuration

### MCP Server Configuration (`config_text2sql_mcp.yml`)

```yaml
functions:
  text2sql_standalone:
    _type: text2sql_standalone
    llm_name: nim_llm
    embedder_name: nim_embedder
    train_on_startup: false       # Set to true for first run
    execute_sql: false            # Only generate SQL, don't execute
    authorize: false              # No auth for load testing
    vanna_remote: true            # Use remote Milvus
    training_analysis_filter:     # Filter few-shot examples
      - pbr
      - supply_gap
```

### LLM Configuration

The config uses NVIDIA NIM for:
- **LLM**: `meta/llama-3.1-70b-instruct` for SQL generation
- **Embedder**: `nvidia/nv-embedqa-e5-v5` for vector search

## Memory Leak Detection

### What to Look For

1. **Steady memory growth** across rounds with same load
2. **Memory not returning to baseline** after load decreases
3. **Increasing response times** over multiple rounds
4. **Session object accumulation** in logs

### Analyzing Results

After running tests, check:

```bash
# View the load test output
tail -f text2sql_load_test.log

# Check memory profile
ls -lh memory_profile_*.txt
```

Look for patterns like:
```
Round 1: 250 MB avg memory
Round 2: 350 MB avg memory  # ⚠️ Growing!
Round 3: 450 MB avg memory  # ⚠️ Still growing!
```

## Troubleshooting

### Server Won't Start

**Error: Module not found 'talk_to_supply_chain'**
```bash
# Install the package
pip install -e talk-to-supply-chain-tools/
```

**Error: Connection to Milvus failed**
```bash
# Check environment variables
echo $MILVUS_HOST
echo $MILVUS_PORT

# Or use local Milvus (set vanna_remote: false in config)
```

### Load Test Issues

**Error: No tools available**
- Verify server is running: `curl http://localhost:9901/health`
- Check tools list: `curl http://localhost:9901/debug/tools/list`

**Error: All calls failing**
- Check server logs for errors
- Verify NVIDIA_API_KEY is set
- Ensure Milvus connection is working

### Memory Profiling

For detailed memory profiling:

```bash
# Use memory_profiler
pip install memory-profiler

# Profile the server
python -m memory_profiler examples/text2sql/text2sql_standalone.py
```

## Files in This Directory

- `text2sql_standalone.py` - Standalone text2sql function (no auth, no execution)
- `text2sql_function.py` - Production function with auth and execution
- `sql_utils.py` - Copy of Vanna utilities from talk_to_supply_chain
- `config_text2sql_mcp.yml` - MCP server configuration
- `text2sql_load_test.py` - Load testing script
- `run_text2sql_load_test.sh` - Automated test runner
- `README.md` - This file

## Advanced Usage

### Custom Load Patterns

Edit `text2sql_load_test.py` to customize:

```python
# Change the questions
SAMPLE_QUESTIONS = [
    "Your custom question 1",
    "Your custom question 2",
    # ...
]

# Change analysis types
ANALYSIS_TYPES = ["custom_type1", "custom_type2"]
```

### Long-Running Stress Test

```bash
# Run overnight test
python examples/text2sql/text2sql_load_test.py \
  --users 50 \
  --calls 100 \
  --rounds 20 \
  --delay 10.0 \
  > overnight_test.log 2>&1 &
```

### Compare with Production Function

To test the production function (with auth):

1. Update `config_text2sql_mcp.yml` to use `text2sql` instead of `text2sql_standalone`
2. Set `authorize: true` and `execute_sql: true`
3. Provide auth token in load test
4. Compare memory usage patterns

## Next Steps

1. Run baseline test to establish normal memory usage
2. Run extended test (multiple rounds) to detect leaks
3. Compare memory profiles between versions
4. Use profiling tools to identify leak sources
5. Fix identified issues and re-test

## Related Documentation

- [MCP Server Documentation](../../docs/source/workflows/mcp/mcp-server.md)
- [MCP Client Documentation](../../docs/source/workflows/mcp/mcp-client.md)
- [Memory Leak Analysis](../../MEMORY_LEAK_ANALYSIS.md)
- [Debug Tools](../../debug_tools/README_MEMORY_LEAK_TEST.md)
