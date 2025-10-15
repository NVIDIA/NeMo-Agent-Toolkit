# Text2SQL MCP Server Load Testing and Memory Leak Detection

Load testing and memory leak detection tools for the Text2SQL MCP server. The test suite simulates multiple concurrent users making text2sql queries while monitoring memory usage to identify memory leaks, performance degradation, and concurrency issues.

## Prerequisites

### Required Dependencies

```bash
pip install psutil requests aiohttp
```

### Environment Setup

1. **Install Text2SQL:**
```bash
uv pip install -e ./examples/text2sql
```

2. **Set up Milvus** (for real Vanna tests only):
```bash
# Local Docker
docker run -d --name milvus \
  -p 19530:19530 \
  -p 9091:9091 \
  -v $(pwd)/milvus_data:/var/lib/milvus \
  milvusdb/milvus:latest
```

3. **Configure environment variables** (create `.env` file):
```bash
# NVIDIA API Key (Required for real Vanna)
NVIDIA_API_KEY=nvapi-your-key-here

# Milvus Configuration (optional, for cloud Milvus)
# MILVUS_HOST=your-cluster.aws-us-west-2.vectordb.zillizcloud.com
# MILVUS_PORT=19530
# MILVUS_USERNAME=your_username
# MILVUS_PASSWORD=your_password
```

4. **Train Vanna** (first time only, for real Vanna tests):
Edit `src/text2sql/configs/config_text2sql_mcp.yml` and set `train_on_startup: true`, then run:
```bash
nat run --config_file ./examples/text2sql/src/text2sql/configs/config_text2sql_mcp.yml --input "test"
```
After training completes, set `train_on_startup: false` in the config.

## Running Load Tests

### Mock Test (Isolate Vanna/Milvus)

To determine if memory leaks are in the Vanna/Milvus layer, run with mock mode:

```bash
python ./examples/text2sql/run_text2sql_memory_leak_test.py \
  --config_file ./examples/text2sql/src/text2sql/configs/config_text2sql_mock.yml
```

Mock mode bypasses all Milvus connections and returns instant mock SQL responses. If the memory leak disappears with mock mode, the leak is in the Vanna/Milvus layer. If it persists, the leak is elsewhere in the system.

### Comparing Real vs Mock

```bash
# Test with real Vanna
python ./examples/text2sql/run_text2sql_memory_leak_test.py \
  --config_file ./examples/text2sql/src/text2sql/configs/config_text2sql_mcp.yml \
  --output_dir test_results/real_vanna

# Test with mock Vanna
python ./examples/text2sql/run_text2sql_memory_leak_test.py \
  --config_file ./examples/text2sql/src/text2sql/configs/config_text2sql_mock.yml \
  --output_dir test_results/mock_vanna

# Compare results
python ./examples/text2sql/analyze_memory_leak.py \
  test_results/real_vanna/text2sql_memory_*.csv \
  test_results/mock_vanna/text2sql_memory_*.csv \
  --compare
```

### Customizing Test Parameters

```bash
python ./examples/text2sql/run_text2sql_memory_leak_test.py --users 50 --calls 20 --rounds 5 --delay 15.0 --output_dir my_results --config_file ./examples/text2sql/src/text2sql/configs/config_text2sql_mcp.yml
```

## Analyzing Results

### Analyze Memory Data

```bash
# Analyze a single test
python ./examples/text2sql/analyze_memory_leak.py test_results/text2sql_memory_*.csv --recommendations

# Compare multiple tests
python ./examples/text2sql/analyze_memory_leak.py \
  test_results/real_vanna/text2sql_memory_*.csv \
  test_results/mock_vanna/text2sql_memory_*.csv \
  --compare
```

### Memory Analysis

The test monitors memory usage and flags potential issues:

- **Normal (<20% growth)**: ✓ Expected memory increase from caching and normal operations
- **Significant (20-50% growth)**: ⚠️ May indicate inefficient resource management
- **Leak (>50% growth)**: ⚠️ Strong indication of a memory leak

### Interpreting Mock Test Results

| Scenario | Real Vanna | Mock Vanna | Conclusion |
|----------|-----------|------------|------------|
| Leak in Vanna/Milvus | 71% growth 🔴 | <10% growth ✅ | Leak is in Milvus connections |
| Leak elsewhere | 71% growth 🔴 | 65% growth 🔴 | Leak is NOT in Vanna/Milvus |
| Multiple leaks | 71% growth 🔴 | 30% growth ⚠️ | Multiple leak sources |

### Test Output

After the test completes, you'll see:

```
======================================================================
TEXT2SQL MEMORY LEAK TEST RESULTS
======================================================================

Memory Analysis:
  Initial memory:    245.32 MB
  Final memory:      289.45 MB
  Peak memory:       312.18 MB
  Memory growth:     44.13 MB
  Growth percentage: 17.99%

✓  Memory growth appears normal (<20%)

Output files:
  Memory data:       test_results/text2sql_memory_20251009_143022.csv
  Load test log:     test_results/text2sql_load_test_20251009_143022.log
  Server log:        test_results/text2sql_server_20251009_143022.log
======================================================================
```

### Output Files

The test generates three files in the output directory:

1. **Memory CSV** (`text2sql_memory_*.csv`): Timestamp, Process ID, RSS in MB, VMS in MB, Total RSS (including children), CPU percentage, Number of child processes

2. **Load Test Log** (`text2sql_load_test_*.log`): User activity, Query execution results, Success/failure rates, Performance metrics

3. **Server Log** (`text2sql_server_*.log`): Server startup messages, Request handling logs, Error messages

## Test Queries

The load test uses realistic supply chain queries:

- **Shortage queries**: "Show me the top 10 components with highest shortages"
- **Lead time queries**: "What components have lead time greater than 50 days?"
- **Inventory queries**: "Display components with nettable inventory above 1000 units"
- **Build request queries**: "Show all components without lead time for build id PB-60506"
- **Material cost queries**: "Display the latest material cost by CM for NVPN 316-0899-000"
- **CM and site queries**: "Show shortage breakdown by CM site"
- **Trend analysis**: "Show demand forecast for next quarter"

These queries are randomly selected during the load test to simulate realistic usage patterns.

## What Gets Tested

### Real Vanna Mode

Tests with actual Milvus connections:
- Milvus sync and async clients
- NVIDIA NIM LLM inference
- NVIDIA NIM embedding generation
- Vector similarity searches
- Training data storage

### Mock Vanna Mode

Bypasses Vanna/Milvus layer entirely:
- No Milvus connections
- No LLM API calls
- No embedding generation
- Returns instant mock SQL responses
- Minimal memory footprint

Both modes test the same MCP server infrastructure, HTTP handling, and session management.

## Troubleshooting

### Server Fails to Start

**Solutions:**
- Check server log for error messages
- Ensure Milvus is running and accessible (for real Vanna)
- Verify NVIDIA API key is valid (for real Vanna)
- Ensure training was completed successfully (for real Vanna)
- Check port 9901 is not already in use: `lsof -i :9901`
- Check Milvus status: `docker ps | grep milvus`

### Memory Monitor Fails

**Solutions:**
- Ensure psutil is installed: `pip install psutil`
- Check that server started successfully
- Verify process name in monitor script

### Load Test Timeouts

**Solutions:**
- Increase timeout in `load_test_text2sql.py` (default 60s)
- Reduce number of concurrent users
- Increase delay between calls
- Check server logs for bottlenecks

### High Failure Rate

**Solutions:**
- Verify server is healthy: `curl http://localhost:9901/health`
- Check server logs for errors
- Ensure training data is properly loaded (for real Vanna)
- Reduce load (fewer users or calls)

## Performance Baselines

Expected performance on a typical development machine:

- **Concurrent users**: 40
- **Queries per second**: 2-5 (depending on query complexity)
- **Memory per query**: ~1-5 MB (temporary, should be released)
- **Initial memory**: 200-300 MB
- **Memory growth**: <20% over three rounds
- **Average query time**: 5-15 seconds (real Vanna), <10ms (mock Vanna)

## Files

### Scripts
- `run_text2sql_memory_leak_test.py` - Main test runner (starts server, monitors memory, runs load tests)
- `load_test_text2sql.py` - Load test client with realistic queries
- `analyze_memory_leak.py` - Memory analysis and leak detection

### Configuration
- `examples/text2sql/src/text2sql/configs/config_text2sql_mcp.yml` - Real Vanna with Milvus
- `examples/text2sql/src/text2sql/configs/config_text2sql_mock.yml` - Mock Vanna without Milvus
- `env.example` - Environment variables template

### Source Code
- `examples/text2sql/src/text2sql/functions/text2sql_standalone.py` - Main function with mock/real mode support
- `examples/text2sql/src/text2sql/functions/sql_utils.py` - Real Vanna integration with cleanup
- `examples/text2sql/src/text2sql/functions/mock_vanna.py` - Mock Vanna for testing
- `examples/text2sql/src/text2sql/utils/` - Database and Milvus utilities
- `examples/text2sql/src/text2sql/resources/` - Follow-up question resources
