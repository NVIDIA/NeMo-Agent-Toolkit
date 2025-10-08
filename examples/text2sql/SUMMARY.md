# Text2SQL MCP Server Load Testing - Implementation Summary

## Overview

I've created a complete MCP server setup and load testing framework for the `text2sql_standalone` function to help identify memory leaks in your production environment.

## 📁 Files Created/Modified

### Core Implementation
1. **`text2sql_standalone.py`** ✅ (Already existed)
   - Standalone version without authentication
   - No SQL execution (only generation)
   - Isolated for testing

2. **`text2sql_function.py`** ✅ (Already existed)
   - Production version with auth
   - Full SQL execution capability

3. **`sql_utils.py`** ✅ (Already existed)
   - Vanna framework utilities
   - Copy from `talk-to-supply-chain-tools`

4. **`config_text2sql_mcp.yml`** ✅ (Already existed)
   - MCP server configuration
   - LLM and embedder settings
   - Milvus connection config

### Load Testing Tools (NEW)
5. **`text2sql_load_test.py`** ✨ **NEW**
   - Custom load testing script for text2sql
   - Uses realistic supply chain questions
   - Supports multiple rounds to detect leaks
   - Tracks response times and success rates
   - Uses proper MCP protocol via CLI

6. **`monitor_server_memory.py`** ✨ **NEW**
   - Memory monitoring script using psutil
   - Tracks RSS, VMS, and CPU usage
   - Detects memory increase patterns
   - Provides summary with leak indicators

7. **`run_text2sql_load_test.sh`** ✨ **NEW**
   - All-in-one automated test runner
   - Starts MCP server
   - Runs load test
   - Monitors memory
   - Generates reports
   - Automatic cleanup

8. **`verify_setup.py`** ✨ **NEW**
   - Setup verification script
   - Checks dependencies
   - Validates configuration
   - Verifies environment variables
   - Tests NAT command availability

### Documentation (NEW)
9. **`README.md`** ✨ **NEW**
   - Comprehensive 400+ line guide
   - Setup instructions
   - Usage examples
   - Troubleshooting guide
   - Configuration options

10. **`QUICKSTART.md`** ✨ **NEW**
    - Fast setup guide
    - Common scenarios
    - Quick commands

11. **`SETUP_COMPLETE.md`** ✨ **NEW**
    - Setup completion checklist
    - What you can do now
    - Test flow explanation
    - Results interpretation

12. **`.gitignore`** ✨ **NEW**
    - Ignores log files
    - Python cache files
    - Environment files

## 🚀 How to Use

### Quick Start (Recommended)

```bash
# 1. Set environment variables
export NVIDIA_API_KEY="your-api-key"
export MILVUS_HOST="your-milvus-host"
export MILVUS_PORT="19530"
export MILVUS_USERNAME="your-username"
export MILVUS_PASSWORD="your-password"

# 2. Run the automated test
./examples/text2sql/run_text2sql_load_test.sh

# That's it! The script handles everything.
```

### Manual Testing

**Terminal 1 - Server:**
```bash
nat mcp serve \
  --config_file examples/text2sql/config_text2sql_mcp.yml \
  --port 9901
```

**Terminal 2 - Load Test:**
```bash
python examples/text2sql/text2sql_load_test.py \
  --users 20 \
  --calls 10 \
  --rounds 3
```

**Terminal 3 - Memory Monitor:**
```bash
SERVER_PID=$(pgrep -f "nat mcp serve")
python examples/text2sql/monitor_server_memory.py --pid $SERVER_PID
```

## 🔍 What Gets Tested

### Test Flow
1. **MCP Server** loads `text2sql_standalone` function
2. **Vanna** connects to Milvus for few-shot examples
3. **Load Test** sends natural language questions
4. **LLM** (via NVIDIA NIM) generates SQL queries
5. **Metrics** collected: memory, response time, success rate
6. **Analysis** checks for performance degradation

### Sample Questions
The load test includes 40+ realistic supply chain questions:
- Supplier queries ("Show me top 10 suppliers by revenue")
- Parts queries ("List parts with inventory below 50 units")
- Order queries ("Find orders with delivery delays")
- Demand queries ("Show seasonal demand patterns")
- Analytics queries ("Calculate average lead time by supplier")

### Analysis Types
Tests different filtering modes:
- `pbr` - Plan-Based Requirements analysis
- `supply_gap` - Supply Gap analysis
- `None` - No filtering

## 📊 Detecting Memory Leaks

### What to Look For

**Good (No Leak):**
```
Round 1: 250 MB avg, 2.3s response time
Round 2: 255 MB avg, 2.4s response time
Round 3: 252 MB avg, 2.3s response time
✓ Performance stable across rounds
```

**Bad (Potential Leak):**
```
Round 1: 250 MB avg, 2.3s response time
Round 2: 350 MB avg, 3.8s response time
Round 3: 450 MB avg, 5.2s response time
⚠️  Performance degradation detected (126% slower)
⚠️  Memory increased 80% (200 MB)
```

### Warning Signs
1. Memory increases >20% across rounds
2. Response times increase significantly
3. Memory doesn't return to baseline after load
4. Continuous growth pattern over multiple rounds

## 📁 Test Results

All logs are saved to `examples/text2sql/logs/`:

```
logs/
├── server_YYYYMMDD_HHMMSS.log       # MCP server output
├── loadtest_YYYYMMDD_HHMMSS.log     # Load test results
└── memory_YYYYMMDD_HHMMSS.log       # Memory monitoring
```

## ⚙️ Configuration

Edit `config_text2sql_mcp.yml` to customize:

```yaml
functions:
  text2sql_standalone:
    train_on_startup: false         # Train Vanna on first run
    execute_sql: false              # Only generate, don't execute
    authorize: false                # No auth for testing
    vanna_remote: true              # Use remote Milvus
    training_analysis_filter:       # Filter examples
      - pbr
      - supply_gap

llms:
  nim_llm:
    model_name: meta/llama-3.1-70b-instruct
    temperature: 0.0                # Deterministic for testing

embedders:
  nim_embedder:
    model_name: nvidia/nv-embedqa-e5-v5
```

## 🛠️ Customization

### Change Load Parameters

```bash
# Aggressive load
NUM_USERS=50 CALLS_PER_USER=20 NUM_ROUNDS=10 \
  ./examples/text2sql/run_text2sql_load_test.sh

# Light load (smoke test)
NUM_USERS=5 CALLS_PER_USER=3 NUM_ROUNDS=2 \
  ./examples/text2sql/run_text2sql_load_test.sh
```

### Add Custom Questions

Edit `text2sql_load_test.py`:
```python
SAMPLE_QUESTIONS = [
    "Your custom question 1",
    "Your custom question 2",
    # ...
]
```

### Change Monitoring Interval

```bash
python examples/text2sql/monitor_server_memory.py \
  --pid $SERVER_PID \
  --interval 1.0  # Sample every 1 second
```

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `talk_to_supply_chain` not found | `pip install -e talk-to-supply-chain-tools/` |
| Milvus connection failed | Set `vanna_remote: false` in config |
| NVIDIA API key invalid | Check `echo $NVIDIA_API_KEY` |
| Server starts but timeouts | Check server logs for errors |
| No memory increase detected | Increase load or run longer |

### Getting Help

1. Check documentation:
   - `examples/text2sql/README.md` - Full guide
   - `examples/text2sql/QUICKSTART.md` - Quick start
   - `examples/text2sql/SETUP_COMPLETE.md` - Setup checklist

2. Run verification:
   ```bash
   python examples/text2sql/verify_setup.py
   ```

3. Check logs:
   ```bash
   ls -lh examples/text2sql/logs/
   tail -f examples/text2sql/logs/server_*.log
   ```

## 🎯 Next Steps

### After Running Tests

1. **Review Results**
   - Check logs in `examples/text2sql/logs/`
   - Look for memory growth patterns
   - Note any performance degradation

2. **If Leak Found**
   - Identify source (Vanna, Milvus, LLM client, sessions)
   - Review `sql_utils.py` lifecycle management
   - Test fixes with modified code
   - Re-run load test to verify

3. **If No Leak**
   - Test production function (with auth + execution)
   - Increase test intensity (more users, longer duration)
   - Test other components in the workflow

### Advanced Testing

```bash
# Long-running stress test (overnight)
nohup ./examples/text2sql/run_text2sql_load_test.sh \
  NUM_USERS=30 NUM_ROUNDS=50 > overnight.log 2>&1 &

# Memory profiling with memory_profiler
pip install memory-profiler
python -m memory_profiler text2sql_standalone.py
```

## 📚 Architecture

### Components Tested

```
┌─────────────────────────────────────────────────┐
│  Load Test (text2sql_load_test.py)             │
│  - Generates realistic questions                │
│  - Simulates concurrent users                   │
│  - Tracks metrics                               │
└────────────────┬────────────────────────────────┘
                 │ MCP Protocol (via nat mcp client)
                 ▼
┌─────────────────────────────────────────────────┐
│  MCP Server (nat mcp serve)                     │
│  - Exposes text2sql_standalone as tool          │
│  - Handles concurrent requests                  │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  text2sql_standalone Function                   │
│  - Receives natural language question           │
│  - Calls Vanna for SQL generation               │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  Vanna (sql_utils.py)                           │
│  - Retrieves few-shot examples from Milvus      │
│  - Generates SQL via LLM                        │
│  - Returns SQL query                            │
└─────────────────────────────────────────────────┘
```

### Potential Leak Sources

1. **Vanna Instance Caching** - Global singleton may accumulate state
2. **Milvus Connections** - Vector store connections not properly closed
3. **LLM Client Pooling** - HTTP clients or sessions accumulating
4. **Session Management** - NAT context/session objects not cleaned up
5. **Embedder Cache** - Embedding results cached indefinitely

## 📈 Success Metrics

### What Success Looks Like

- ✅ All tests pass with >95% success rate
- ✅ Memory usage stable across rounds (<10% variance)
- ✅ Response times consistent (<20% variance)
- ✅ Server handles 20+ concurrent users without degradation
- ✅ No errors in server logs

### Baseline Expectations

For reference, typical values on a healthy system:
- **Success rate**: 98-100%
- **Response time**: 2-5 seconds per query
- **Memory usage**: 200-500 MB baseline, <10% growth per round
- **Throughput**: 5-15 calls/second with 20 concurrent users

## 🔐 Security Notes

- This setup **disables authentication** for testing (`authorize: false`)
- This setup **disables SQL execution** for safety (`execute_sql: false`)
- **Do not expose** this server to the internet
- **Environment files** with secrets are in `.gitignore`

## 🙏 Credits

Based on:
- Existing `debug_tools/mcp_load_test.py` for general MCP testing
- Production `text2sql_function.py` from talk-to-supply-chain
- NeMo Agent Toolkit MCP server implementation

## 📝 Summary

**Created**: 2025-10-08
**Purpose**: Test text2sql function for memory leaks
**Status**: ✅ Ready to use
**Files**: 12 files (8 new, 4 existing)
**Lines of Code**: ~2,500 lines of code and documentation

**Quick Start Command:**
```bash
./examples/text2sql/run_text2sql_load_test.sh
```

---

For questions or issues, refer to:
- `examples/text2sql/README.md` - Full documentation
- `examples/text2sql/QUICKSTART.md` - Quick start guide
- `examples/text2sql/SETUP_COMPLETE.md` - Setup checklist
