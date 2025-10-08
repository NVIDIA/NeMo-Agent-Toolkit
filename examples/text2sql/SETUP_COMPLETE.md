# Text2SQL MCP Server Load Testing - Setup Complete ✅

This document summarizes what has been created for testing the text2sql function for memory leaks.

## Files Created

### Configuration & Code
- ✅ `config_text2sql_mcp.yml` - MCP server configuration (already existed)
- ✅ `text2sql_standalone.py` - Standalone function without auth (already existed)
- ✅ `text2sql_function.py` - Production function with auth (already existed)
- ✅ `sql_utils.py` - Vanna utilities (copy from talk_to_supply_chain)
- ✅ `__init__.py` - Python package initialization

### Load Testing Tools
- ✅ `text2sql_load_test.py` - Load testing script with realistic queries
- ✅ `monitor_server_memory.py` - Memory monitoring script
- ✅ `run_text2sql_load_test.sh` - Automated test runner (all-in-one)

### Documentation
- ✅ `README.md` - Comprehensive guide with all details
- ✅ `QUICKSTART.md` - Quick start guide for fast setup
- ✅ `.gitignore` - Ignore logs and temporary files

## What You Can Do Now

### 1. Quick Test (5 minutes)

```bash
# Make sure you have the environment variables set
export NVIDIA_API_KEY="your-api-key"
export MILVUS_HOST="your-milvus-host"
export MILVUS_PORT="19530"
export MILVUS_USERNAME="your-username"
export MILVUS_PASSWORD="your-password"

# Run the automated test
./examples/text2sql/run_text2sql_load_test.sh
```

This will:
1. Start MCP server on port 9901
2. Monitor memory usage
3. Run 3 rounds of load testing (20 users × 10 calls each)
4. Generate report showing:
   - Success rate
   - Response times
   - Memory usage patterns
   - Performance degradation (if any)
5. Clean up automatically

### 2. Manual Testing (for debugging)

**Terminal 1 - Start Server:**
```bash
nat mcp serve \
  --config_file examples/text2sql/config_text2sql_mcp.yml \
  --port 9901 \
  --log_level INFO
```

**Terminal 2 - Run Load Test:**
```bash
python examples/text2sql/text2sql_load_test.py \
  --url http://localhost:9901/mcp \
  --users 20 \
  --calls 10 \
  --rounds 3 \
  --verbose
```

**Terminal 3 - Monitor Memory:**
```bash
# Get server PID
SERVER_PID=$(pgrep -f "nat mcp serve")

# Monitor it
python examples/text2sql/monitor_server_memory.py --pid $SERVER_PID
```

### 3. Stress Test (for finding slow leaks)

```bash
# Run with higher load over longer period
NUM_USERS=50 \
CALLS_PER_USER=20 \
NUM_ROUNDS=10 \
DELAY_BETWEEN_ROUNDS=10.0 \
./examples/text2sql/run_text2sql_load_test.sh
```

## Understanding the Test

### Test Flow

1. **Server Start**: MCP server loads `text2sql_standalone` function
2. **Connection**: Vanna connects to Milvus vector store for few-shot examples
3. **Load Test**: Multiple concurrent users send natural language questions
4. **SQL Generation**: Vanna generates SQL using LLM + RAG (no execution)
5. **Monitoring**: Memory and performance metrics collected
6. **Analysis**: Trends analyzed across multiple rounds

### What We're Testing

The `text2sql_standalone` function:
- Takes natural language questions
- Retrieves relevant few-shot examples from Milvus
- Uses NVIDIA NIM LLM to generate SQL
- Returns SQL query (no execution in standalone version)

**Key differences from production:**
- ❌ No authentication
- ❌ No SQL execution
- ❌ No database connection needed
- ✅ Same Vanna logic
- ✅ Same LLM/embedder
- ✅ Same Milvus integration

This isolates the core text2sql logic for testing.

### Sample Test Questions

The load test uses realistic supply chain questions:
- "Show me the top 10 suppliers by total revenue"
- "List all parts with inventory below 50 units"
- "What are the most expensive parts in inventory?"
- "Find orders with delivery delays"
- "Show seasonal demand patterns"

See `text2sql_load_test.py` for the full list.

### Memory Leak Indicators

**Warning Signs:**
1. Memory increases >20% across rounds
2. Response times increase significantly
3. Memory doesn't return to baseline
4. Continuous growth pattern

**What to Look For in Logs:**
```
# Good (no leak)
Round 1: 250 MB avg memory, 2.3s avg response
Round 2: 255 MB avg memory, 2.4s avg response
Round 3: 252 MB avg memory, 2.3s avg response

# Bad (potential leak)
Round 1: 250 MB avg memory, 2.3s avg response
Round 2: 350 MB avg memory, 3.1s avg response
Round 3: 450 MB avg memory, 4.2s avg response
```

## Test Results Location

After running tests, check these directories:

```
examples/text2sql/logs/
├── server_YYYYMMDD_HHMMSS.log      # Server output
├── loadtest_YYYYMMDD_HHMMSS.log    # Load test results
└── memory_YYYYMMDD_HHMMSS.log      # Memory monitoring data
```

## Configuration Options

Edit `config_text2sql_mcp.yml` to change:

```yaml
functions:
  text2sql_standalone:
    # Training
    train_on_startup: false         # Set true for first run only
    training_analysis_filter:       # Filter few-shot examples
      - pbr
      - supply_gap

    # Milvus
    vanna_remote: true              # Use remote Milvus (vs local)
    milvus_host: ${MILVUS_HOST}
    milvus_port: ${MILVUS_PORT}

    # Functionality
    execute_sql: false              # Don't execute (for testing)
    authorize: false                # No auth (for testing)
    allow_llm_to_see_data: false    # Don't show data to LLM
    enable_followup_questions: false  # No followup generation

# LLM settings
llms:
  nim_llm:
    model_name: meta/llama-3.1-70b-instruct
    temperature: 0.0                # Deterministic for testing
    max_tokens: 2048
    timeout: 60.0

# Embedder settings
embedders:
  nim_embedder:
    model_name: nvidia/nv-embedqa-e5-v5
    truncate: END
```

## Troubleshooting Common Issues

### 1. "Module 'talk_to_supply_chain' not found"

**Solution:**
```bash
pip install -e talk-to-supply-chain-tools/
```

Or update imports in `text2sql_standalone.py` to use local `sql_utils.py`.

### 2. "Milvus connection failed"

**Option A - Use local Milvus:**
```yaml
# In config_text2sql_mcp.yml
vanna_remote: false
```

**Option B - Check remote credentials:**
```bash
echo $MILVUS_HOST
echo $MILVUS_PORT
echo $MILVUS_USERNAME
```

### 3. "NVIDIA API key invalid"

```bash
# Verify key is set
echo $NVIDIA_API_KEY

# Should start with "nvapi-"
```

### 4. Server starts but requests timeout

Check server logs:
```bash
tail -f examples/text2sql/logs/server_*.log
```

Common causes:
- Milvus connection slow/failing
- LLM API rate limiting
- Missing dependencies

### 5. No memory increase detected

This is actually good! It means no obvious leak. To be thorough:
- Increase load: more users, more calls
- Run longer: more rounds
- Check for subtle patterns in detailed memory logs

## Next Steps

### After Running Tests

1. **Review logs** in `examples/text2sql/logs/`
2. **Check for patterns** across multiple rounds
3. **Compare with production** (if accessible)
4. **Use profilers** for deeper analysis:
   ```bash
   pip install memory-profiler
   python -m memory_profiler text2sql_standalone.py
   ```

### If Memory Leak Found

1. **Identify the source:**
   - Vanna instance caching?
   - Milvus connections?
   - LLM client pooling?
   - Session management?

2. **Review related code:**
   - `sql_utils.py` - Vanna lifecycle
   - `text2sql_standalone.py` - Function registration
   - Builder/context management in NAT

3. **Test fixes:**
   - Modify code
   - Re-run load test
   - Compare memory profiles

### If No Leak Found

1. **Test production function:**
   - Use `text2sql_function.py` instead
   - Enable auth and SQL execution
   - Compare memory patterns

2. **Test other components:**
   - Auth module
   - Database connections
   - Other functions in the workflow

3. **Increase test intensity:**
   - More concurrent users (50+)
   - Longer duration (100+ rounds)
   - Vary query patterns

## Additional Resources

- **Main README**: [README.md](./README.md) - Full documentation
- **Quick Start**: [QUICKSTART.md](./QUICKSTART.md) - Fast setup guide
- **Memory Analysis**: [../../MEMORY_LEAK_ANALYSIS.md](../../MEMORY_LEAK_ANALYSIS.md) - Overall memory leak investigation
- **Debug Tools**: [../../debug_tools/](../../debug_tools/) - Additional testing utilities
- **MCP Docs**: [../../docs/source/workflows/mcp/](../../docs/source/workflows/mcp/) - MCP server/client documentation

## Contact & Support

For questions or issues:
1. Check documentation above
2. Review existing load test results
3. Examine server logs for errors
4. Consult with the team

---

**Setup completed on**: 2025-10-08

**Created for**: Testing text2sql function for memory leaks in production environment

**Status**: ✅ Ready to run
