# Text2SQL MCP Server Load Testing and Memory Leak Detection

This guide explains how to run load tests on the Text2SQL MCP server to detect potential memory leaks and performance issues.

## Overview

The load testing suite simulates multiple concurrent users making text2sql queries to an MCP server while monitoring memory usage. It helps identify:

- Memory leaks in the server implementation
- Performance degradation under load
- Concurrency issues
- Resource consumption patterns

## Prerequisites

### Required Dependencies

```bash
pip install psutil requests aiohttp
```

### Environment Setup

1. **Ensure Text2SQL is installed:**
```bash
cd examples/text2sql
uv pip install -e .
```

2. **Set up Milvus** (if not already running):
```bash
# Option 1: Local Docker
docker run -d --name milvus \
  -p 19530:19530 \
  -p 9091:9091 \
  -v $(pwd)/milvus_data:/var/lib/milvus \
  milvusdb/milvus:latest

# Option 2: Use cloud Milvus (set environment variables)
```

3. **Configure environment variables:**
Create a `.env` file in the `examples/text2sql` directory:

```bash
# NVIDIA API Key (Required)
NVIDIA_API_KEY=nvapi-your-key-here

# Milvus Configuration (if using cloud)
# MILVUS_HOST=your-cluster.aws-us-west-2.vectordb.zillizcloud.com
# MILVUS_PORT=19530
# MILVUS_USERNAME=your_username
# MILVUS_PASSWORD=your_password
```

4. **Train Vanna** (first time only):
Edit `configs/config_text2sql_mcp.yml` and set:
```yaml
functions:
  text2sql_standalone:
    train_on_startup: true  # Set to true for first run
```

Run training:
```bash
nat run --config_file configs/config_text2sql_mcp.yml --input "test"
```

Then set `train_on_startup: false` in the config.

## Running Load Tests

### Quick Start

Run the complete integrated test suite:

```bash
cd examples/text2sql
python run_text2sql_memory_leak_test.py
```

This will:
1. Start the Text2SQL MCP server
2. Start memory monitoring
3. Run 3 rounds of load tests with 40 concurrent users
4. Analyze memory usage and detect potential leaks
5. Save results to `test_results/` directory

### Test Results

After the test completes, you'll see output like:

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

### Customizing Test Parameters

```bash
python run_text2sql_memory_leak_test.py \
  --config_file configs/config_text2sql_mcp.yml \
  --host localhost \
  --port 9901 \
  --users 50 \                    # Number of concurrent users
  --calls 20 \                     # Calls per user per round
  --rounds 5 \                     # Number of test rounds
  --delay 15.0 \                   # Delay between rounds (seconds)
  --output_dir my_test_results
```

### Running Only the Load Test

If you already have a running MCP server:

```bash
# Start the server in one terminal
nat mcp serve --config_file configs/config_text2sql_mcp.yml

# Run the load test in another terminal
python load_test_text2sql.py \
  --url http://localhost:9901/mcp \
  --users 40 \
  --calls 10 \
  --rounds 3 \
  --delay 5.0
```

## Understanding Test Results

### Memory Analysis

The test monitors memory usage and flags potential issues:

- **Normal (<20% growth)**: ✓ Expected memory increase from caching and normal operations
- **Significant (20-50% growth)**: ⚠️ May indicate inefficient resource management
- **Leak (>50% growth)**: ⚠️ Strong indication of a memory leak

### Output Files

The test generates three files in the output directory:

1. **Memory CSV** (`text2sql_memory_*.csv`):
   - Timestamp
   - Process ID
   - RSS (Resident Set Size) in MB
   - VMS (Virtual Memory Size) in MB
   - Total RSS (including children)
   - CPU percentage
   - Number of child processes

2. **Load Test Log** (`text2sql_load_test_*.log`):
   - User activity
   - Query execution results
   - Success/failure rates
   - Performance metrics

3. **Server Log** (`text2sql_server_*.log`):
   - Server startup messages
   - Request handling logs
   - Error messages (if any)

### Analyzing Memory CSV

You can analyze the memory data using Python:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('test_results/text2sql_memory_20251009_143022.csv')

# Plot memory usage over time
df['timestamp'] = pd.to_datetime(df['timestamp'])
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['rss_total_mb'])
plt.xlabel('Time')
plt.ylabel('Memory (MB)')
plt.title('Text2SQL MCP Server Memory Usage')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('memory_usage.png')
```

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

## Troubleshooting

### Server Fails to Start

**Issue**: Server process terminates unexpectedly

**Solutions**:
- Check server log for error messages
- Ensure Milvus is running and accessible
- Verify NVIDIA API key is valid
- Ensure training was completed successfully
- Check port 9901 is not already in use

```bash
# Check if port is in use
lsof -i :9901

# Check Milvus status
docker ps | grep milvus
```

### Memory Monitor Fails

**Issue**: Cannot find process to monitor

**Solutions**:
- Ensure psutil is installed: `pip install psutil`
- Check that server started successfully
- Verify process name in monitor script

### Load Test Timeouts

**Issue**: Tool calls timing out

**Solutions**:
- Increase timeout in `load_test_text2sql.py` (default 60s)
- Reduce number of concurrent users
- Increase delay between calls
- Check server logs for bottlenecks

### High Failure Rate

**Issue**: Many failed calls in load test

**Solutions**:
- Verify server is healthy: `curl http://localhost:9901/health`
- Check server logs for errors
- Ensure training data is properly loaded
- Reduce load (fewer users or calls)

## Advanced Usage

### Running Manual Tests

Start the server and memory monitor separately:

```bash
# Terminal 1: Start server
nat mcp serve --config_file configs/config_text2sql_mcp.yml

# Terminal 2: Start memory monitor
python ../../debug_tools/monitor_memory.py \
  --name uvicorn \
  --interval 1.0 \
  --output memory_data.csv

# Terminal 3: Run load test
python load_test_text2sql.py --users 40 --calls 10
```

### Continuous Monitoring

For long-running tests:

```bash
python run_text2sql_memory_leak_test.py \
  --users 20 \
  --calls 50 \
  --rounds 10 \
  --delay 30.0
```

### Integration with CI/CD

Add memory leak detection to your continuous integration:

```bash
# Run test and check exit code
python run_text2sql_memory_leak_test.py --users 10 --calls 5 --rounds 2

# Parse results and fail if memory growth >50%
python analyze_memory_results.py test_results/text2sql_memory_*.csv --threshold 0.5
```

## Performance Baselines

Expected performance on a typical development machine:

- **Concurrent users**: 40
- **Queries per second**: 2-5 (depending on query complexity)
- **Memory per query**: ~1-5 MB (temporary, should be released)
- **Initial memory**: 200-300 MB
- **Memory growth**: <20% over 3 rounds
- **Average query time**: 5-15 seconds

## Known Issues

- First queries after server start may be slower (cold start)
- Vanna training increases initial memory usage
- LLM API rate limits may cause some failures
- Milvus connection issues can cause timeouts

## References

- [Text2SQL Getting Started](GETTING_STARTED.md)
- [Text2SQL README](README.md)
- [NAT Memory Leak Analysis](../../MEMORY_LEAK_ANALYSIS.md)
- [Debug Tools](../../debug_tools/)
