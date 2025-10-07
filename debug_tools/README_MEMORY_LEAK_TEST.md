# MCP Server Memory Leak Test Suite

This test suite helps reproduce and debug memory leak issues in the NeMo Agent toolkit MCP server.

## Overview

The test suite consists of three main components:

1. **`mcp_load_test.py`**: Simulates multiple concurrent users making tool calls to the MCP server
2. **`monitor_memory.py`**: Monitors and logs memory usage of the MCP server process
3. **`run_memory_leak_test.py`**: Orchestrates the entire test (starts server, monitors memory, runs load tests)

## Prerequisites

Install required dependencies:

```bash
# Install psutil for memory monitoring
pip install psutil

# Install requests for health checks
pip install requests

# Install aiohttp for async HTTP requests
pip install aiohttp

# Or install all at once
pip install psutil requests aiohttp
```

Ensure you have the MCP server dependencies installed:

```bash
uv pip install "nvidia-nat[mcp]"
```

## Quick Start - Integrated Test

The easiest way to run the complete test is using the integrated test runner:

```bash
python debug_tools/run_memory_leak_test.py \
  --config_file examples/getting_started/simple_calculator/configs/config.yml \
  --users 40 \
  --calls 10 \
  --rounds 3
```

This will:
1. Start the MCP server
2. Start memory monitoring
3. Run 3 rounds of load tests with 40 concurrent users making 10 calls each
4. Generate a report with memory analysis
5. Save all logs and data to `debug_tools/memory_leak_test_results/`

### Command Options

- `--config_file`: Path to NAT workflow config file (required)
- `--host`: MCP server host (default: localhost)
- `--port`: MCP server port (default: 9901)
- `--users`: Number of concurrent users to simulate (default: 40)
- `--calls`: Number of calls per user (default: 10)
- `--rounds`: Number of load test rounds (default: 3)
- `--delay`: Delay between rounds in seconds (default: 10.0)
- `--output_dir`: Output directory for results

## Manual Testing - Step by Step

If you prefer to run each component separately for more control:

### Step 1: Start the MCP Server

```bash
nat mcp serve --config_file examples/getting_started/simple_calculator/configs/config.yml
```

Leave this running in a separate terminal.

### Step 2: Start Memory Monitoring

In another terminal, start monitoring the server process:

```bash
# If you know the PID
python debug_tools/monitor_memory.py --pid <PID>

# Or let it auto-detect the process
python debug_tools/monitor_memory.py --name uvicorn
```

The monitor will log memory usage to a CSV file (auto-generated with timestamp).

### Step 3: Run Load Tests

In a third terminal, run the load testing:

```bash
# Single round with 40 users
python debug_tools/mcp_load_test.py --users 40 --calls 10

# Multiple rounds
python debug_tools/mcp_load_test.py --users 40 --calls 10 --rounds 3 --delay 10
```

### Step 4: Stop Monitoring

Press `Ctrl+C` in the memory monitoring terminal to stop and see the summary.

## Understanding the Results

### Memory Monitor Output

The memory monitor tracks several metrics:

- **RSS (Resident Set Size)**: Physical memory used by the process
- **RSS Total**: Physical memory including child processes
- **VMS (Virtual Memory Size)**: Total virtual memory allocated
- **Percent**: Memory usage as percentage of total system memory

Example summary:

```
MEMORY MONITORING SUMMARY
======================================================================
Process PID:           12345
Total samples:         300
Duration:              300.00 seconds

Process Memory (RSS):
  Initial:             250.00 MB
  Final:               850.00 MB
  Min:                 245.00 MB
  Max:                 860.00 MB
  Avg:                 550.00 MB
  Growth:              600.00 MB    <- Key indicator

Total Memory (Process + Children):
  Initial:             255.00 MB
  Final:               870.00 MB
  Growth:              615.00 MB    <- Includes all subprocesses
======================================================================
```

**Red Flags for Memory Leaks:**
- Continuous growth without plateau
- Growth > 50% of initial memory
- Memory not returning to baseline after load stops
- Linear growth correlated with number of requests

### Load Test Output

The load tester provides statistics:

```
LOAD TEST SUMMARY
======================================================================
Total users:        40
Calls per user:     10
Total calls:        400
Successful calls:   398
Failed calls:       2
Success rate:       99.50%
Duration:           45.23 seconds
Calls per second:   8.84
======================================================================
```

### Integrated Test Analysis

The integrated test runner provides automatic analysis:

```
TEST RESULTS ANALYSIS
======================================================================
Memory Analysis:
  Initial memory:    250.00 MB
  Final memory:      850.00 MB
  Peak memory:       870.00 MB
  Memory growth:     600.00 MB
  Growth percentage: 240.00%

⚠️  POTENTIAL MEMORY LEAK DETECTED!
   Memory increased by >50% during test

Output files:
  Memory data:       debug_tools/memory_leak_test_results/memory_20251007_123456.csv
  Load test log:     debug_tools/memory_leak_test_results/load_test_20251007_123456.log
  Server log:        debug_tools/memory_leak_test_results/server_20251007_123456.log
======================================================================
```

## Analyzing Memory Data

The memory CSV file can be imported into spreadsheet software or analyzed with Python:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('debug_tools/memory_leak_test_results/memory_20251007_123456.csv')

# Plot memory usage over time
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['rss_total_mb'], label='Total RSS (MB)')
plt.xlabel('Sample Number')
plt.ylabel('Memory (MB)')
plt.title('MCP Server Memory Usage Over Time')
plt.legend()
plt.grid(True)
plt.savefig('memory_usage.png')
plt.show()
```

## Troubleshooting

### MCP Server Won't Start

Check if the port is already in use:

```bash
lsof -i :9901
```

Use a different port:

```bash
python debug_tools/run_memory_leak_test.py --config_file <config> --port 9902
```

### Memory Monitor Can't Find Process

List running processes to find the correct one:

```bash
ps aux | grep uvicorn
```

Use the specific PID:

```bash
python debug_tools/monitor_memory.py --pid <PID>
```

### Load Test Connection Errors

Verify the MCP server is running and healthy:

```bash
curl http://localhost:9901/health
```

Check available tools:

```bash
curl http://localhost:9901/debug/tools/list
```

### Permission Errors

On some systems, you may need elevated permissions to monitor processes:

```bash
sudo python debug_tools/monitor_memory.py --pid <PID>
```

## Advanced Usage

### Custom Load Patterns

Modify `mcp_load_test.py` to simulate different usage patterns:

```python
# Add variable think time between requests
await asyncio.sleep(random.uniform(0.1, 2.0))

# Simulate bursts of activity
for burst in range(5):
    # Make multiple rapid calls
    await call_multiple_tools()
    # Rest period
    await asyncio.sleep(30)
```

### Memory Profiling

For deeper memory analysis, use Python memory profilers:

```bash
# Using memory_profiler
pip install memory_profiler

# Profile the MCP server startup
python -m memory_profiler -m nat.cli mcp serve --config_file <config>
```

### Continuous Monitoring

Run the test for extended periods to observe long-term trends:

```bash
# Run 10 rounds with longer delays
python debug_tools/run_memory_leak_test.py \
  --config_file <config> \
  --rounds 10 \
  --delay 60 \
  --users 20
```

## Expected Behavior vs Memory Leak

### Normal Behavior
- Initial spike during server initialization
- Gradual increase during load
- Plateau or slow growth during sustained load
- Memory partially or fully released after load stops
- Stable baseline between rounds

### Memory Leak Indicators
- Continuous linear growth
- Memory never returns to baseline
- Growth proportional to number of requests
- No plateau even with constant load
- Each test round increases baseline

## Next Steps for Debugging

Once you've reproduced the memory leak:

1. **Profile specific operations**: Identify which tool calls or operations cause the most growth
2. **Check object retention**: Use `objgraph` or similar tools to see what objects are being retained
3. **Review session management**: Check if MCP sessions are being properly closed
4. **Inspect connection pooling**: Verify HTTP connections are being released
5. **Check for circular references**: Look for circular references preventing garbage collection

## Output Files

All test results are saved to `debug_tools/memory_leak_test_results/` with timestamps:

- `memory_YYYYMMDD_HHMMSS.csv`: Detailed memory usage data
- `load_test_YYYYMMDD_HHMMSS.log`: Load test execution logs
- `server_YYYYMMDD_HHMMSS.log`: MCP server logs

These files can be used for:
- Creating visualizations
- Comparing different test runs
- Sharing with the development team
- Documenting the issue
