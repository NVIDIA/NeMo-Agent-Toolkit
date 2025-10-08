# Text2SQL MCP Server Load Testing - Quick Start

This guide will help you quickly set up and run load tests for the text2sql function.

## 1. Install Dependencies

```bash
# Install NeMo Agent Toolkit with MCP support
pip install -e ".[mcp]"

# Install talk-to-supply-chain-tools (if not already installed)
# Adjust path as needed based on your directory structure
pip install -e talk-to-supply-chain-tools/

# Install additional dependencies
pip install psutil aiohttp
```

## 2. Set Environment Variables

Create a `.env` file or export variables:

```bash
# Required: NVIDIA API Key for LLM and embeddings
export NVIDIA_API_KEY="nvapi-xxxxxxxxxxxx"

# Required if using remote Milvus (vanna_remote: true in config)
export MILVUS_HOST="your-milvus-host.com"
export MILVUS_PORT="19530"
export MILVUS_USERNAME="your-username"
export MILVUS_PASSWORD="your-password"
```

## 3. Run the Test

### Option A: Automated (Recommended)

Run everything with one command:

```bash
./examples/text2sql/run_text2sql_load_test.sh
```

This will:
- Start the MCP server
- Run the load test
- Monitor memory usage
- Generate a report
- Clean up automatically

### Option B: Manual

#### Terminal 1: Start Server

```bash
nat mcp serve \
  --config_file examples/text2sql/config_text2sql_mcp.yml \
  --port 9901
```

Wait for: `Server is ready`

#### Terminal 2: Run Load Test

```bash
python examples/text2sql/text2sql_load_test.py \
  --users 20 \
  --calls 10 \
  --rounds 3
```

#### Terminal 3: Monitor Memory (Optional)

```bash
# Find the server process ID
ps aux | grep "nat mcp serve"

# Monitor it
python examples/text2sql/monitor_server_memory.py --pid <PID>
```

## 4. Interpret Results

### Look for Memory Leaks

**Good** (no leak):
```
Round 1: avg response time: 2.3s
Round 2: avg response time: 2.4s
Round 3: avg response time: 2.3s
✓ Performance stable across rounds
```

**Bad** (potential leak):
```
Round 1: avg response time: 2.3s
Round 2: avg response time: 3.8s
Round 3: avg response time: 5.2s
⚠️  Performance degradation detected
```

### Memory Usage

Check the memory log:
```bash
tail examples/text2sql/logs/memory_*.log
```

**Warning signs:**
- Memory increasing >20% between rounds
- Memory not returning to baseline
- Continuously growing memory usage

## 5. Customize the Test

### Change Load Parameters

```bash
# More aggressive load
NUM_USERS=50 CALLS_PER_USER=20 NUM_ROUNDS=5 \
  ./examples/text2sql/run_text2sql_load_test.sh

# Lighter load
NUM_USERS=5 CALLS_PER_USER=5 NUM_ROUNDS=2 \
  ./examples/text2sql/run_text2sql_load_test.sh
```

### Edit Test Configuration

Edit `examples/text2sql/config_text2sql_mcp.yml`:

```yaml
functions:
  text2sql_standalone:
    train_on_startup: true    # Train on startup (first run only)
    execute_sql: true         # Actually execute SQL (needs DB)
    authorize: true           # Test with authentication
```

### Edit Sample Questions

Edit `examples/text2sql/text2sql_load_test.py`:

```python
SAMPLE_QUESTIONS = [
    "Your custom question 1",
    "Your custom question 2",
    # ... add more
]
```

## 6. Troubleshooting

### Issue: "text2sql_standalone not found"

**Solution:** Check the config file has the correct function type:
```yaml
functions:
  text2sql_standalone:  # Must match the function name
    _type: text2sql_standalone  # Must match the registered type
```

### Issue: "Module 'talk_to_supply_chain' not found"

**Solution:** Install the package:
```bash
pip install -e talk-to-supply-chain-tools/
```

Or update imports in `text2sql_standalone.py` to use local files.

### Issue: "Milvus connection failed"

**Solution 1:** Use local Milvus:
```yaml
vanna_remote: false
```

**Solution 2:** Verify remote Milvus credentials:
```bash
echo $MILVUS_HOST
echo $MILVUS_PORT
```

### Issue: Server starts but no requests complete

**Solution:** Check server logs:
```bash
tail -f examples/text2sql/logs/server_*.log
```

Look for errors related to:
- NVIDIA API key
- Milvus connection
- Missing dependencies

## 7. Next Steps

### Profile Memory in Detail

Use Python memory profiler:
```bash
pip install memory-profiler

# Profile the server
python -m memory_profiler -m nat.cli.cli mcp serve \
  --config_file examples/text2sql/config_text2sql_mcp.yml
```

### Compare Production vs Standalone

1. Test standalone (current setup)
2. Test production function:
   - Update config to use `text2sql` function
   - Add auth token
   - Enable SQL execution
   - Compare memory profiles

### Long-Running Test

Run overnight to catch slow leaks:
```bash
nohup ./examples/text2sql/run_text2sql_load_test.sh \
  NUM_USERS=30 NUM_ROUNDS=50 > overnight.log 2>&1 &
```

## Support

For issues or questions:
- Check the main [README.md](./README.md)
- Review [MEMORY_LEAK_ANALYSIS.md](../../MEMORY_LEAK_ANALYSIS.md)
- See [debug_tools/](../../debug_tools/) for more testing utilities
