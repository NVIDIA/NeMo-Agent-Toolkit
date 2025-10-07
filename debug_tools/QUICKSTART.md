# MCP Server Memory Leak Test - Quick Start Guide

## Installation

1. Install test dependencies:
```bash
pip install -r debug_tools/requirements_memory_test.txt
```

2. Ensure MCP server dependencies are installed:
```bash
uv pip install "nvidia-nat[mcp]"
```

## Running the Test (Easiest Method)

Use the quick test script:

```bash
./debug_tools/quick_test.sh
```

This will automatically:
- Start the MCP server with the default calculator config
- Monitor memory usage
- Run 3 rounds of load tests (40 users, 10 calls per user)
- Generate a detailed report

### Using a Custom Config

```bash
./debug_tools/quick_test.sh path/to/your/config.yml
```

## Understanding the Output

The test will produce output in `debug_tools/memory_leak_test_results/`:

### Files Generated
- `memory_TIMESTAMP.csv` - Detailed memory usage over time
- `load_test_TIMESTAMP.log` - Load test execution details
- `server_TIMESTAMP.log` - MCP server logs

### Memory Analysis
The test automatically analyzes memory behavior:

```
Memory Analysis:
  Initial memory:    250.00 MB
  Final memory:      850.00 MB
  Peak memory:       870.00 MB
  Memory growth:     600.00 MB
  Growth percentage: 240.00%

⚠️  POTENTIAL MEMORY LEAK DETECTED!
   Memory increased by >50% during test
```

### Interpreting Results

**Memory Leak Indicators:**
- ⚠️ Growth >50%: Strong indication of memory leak
- ⚠️ Growth 20-50%: Possible memory leak
- ✓ Growth <20%: Normal behavior

**What to Look For:**
1. **Linear growth** - Memory increases consistently with each round
2. **No recovery** - Memory doesn't return to baseline after load
3. **Proportional growth** - Memory increase correlates with request count

## Manual Testing (Advanced)

If you need more control, run components separately:

### Terminal 1 - Start MCP Server
```bash
nat mcp serve --config_file examples/getting_started/simple_calculator/configs/config.yml
```

### Terminal 2 - Monitor Memory
```bash
python debug_tools/monitor_memory.py --name uvicorn
```

### Terminal 3 - Run Load Tests
```bash
python debug_tools/mcp_load_test.py --users 40 --calls 10 --rounds 3
```

## Customizing the Test

### More Aggressive Load
```bash
python debug_tools/run_memory_leak_test.py \
  --config_file your_config.yml \
  --users 100 \
  --calls 20 \
  --rounds 5
```

### Extended Duration Test
```bash
python debug_tools/run_memory_leak_test.py \
  --config_file your_config.yml \
  --users 20 \
  --calls 50 \
  --rounds 10 \
  --delay 30
```

### Different Server Port
```bash
python debug_tools/run_memory_leak_test.py \
  --config_file your_config.yml \
  --port 8080
```

## Analyzing Results

### View Memory Data
```bash
# Quick view of memory data
head -20 debug_tools/memory_leak_test_results/memory_*.csv

# View with column formatting
column -s, -t < debug_tools/memory_leak_test_results/memory_*.csv | less
```

### Plot Memory Usage (requires matplotlib)
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load the most recent memory data
df = pd.read_csv('debug_tools/memory_leak_test_results/memory_TIMESTAMP.csv')

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['rss_total_mb'])
plt.xlabel('Time (seconds)')
plt.ylabel('Memory (MB)')
plt.title('MCP Server Memory Usage Over Time')
plt.grid(True)
plt.savefig('memory_leak_analysis.png')
```

## Troubleshooting

### "Error getting tools" or "No tools available"
The load tester can't retrieve tools from the MCP server.

**Diagnosis:**
```bash
# Test the MCP endpoint directly
python debug_tools/test_mcp_endpoint.py

# Or with custom URL
python debug_tools/test_mcp_endpoint.py --url http://localhost:9901/mcp
```

This will show exactly what the server is returning and help identify the issue.

**Common causes:**
- Server not fully started (wait a few more seconds)
- Wrong URL or port
- Server returning unexpected format

### "No process found"
The memory monitor couldn't find the MCP server process.

**Solution:**
```bash
# Find the process manually
ps aux | grep "nat mcp serve"

# Use the specific PID
python debug_tools/monitor_memory.py --pid <PID>
```

### "Connection refused"
The load tester can't connect to the MCP server.

**Solution:**
```bash
# Check if server is running
curl http://localhost:9901/health

# Or use the diagnostic tool
python debug_tools/test_mcp_endpoint.py

# Check if port is correct
nat mcp serve --config_file your_config.yml --port 9901
```

### "Port already in use"
Another process is using port 9901.

**Solution:**
```bash
# Find what's using the port
lsof -i :9901

# Kill the process or use a different port
python debug_tools/run_memory_leak_test.py --config_file your_config.yml --port 9902
```

## What to Report

When reporting memory leak issues, include:

1. **Test configuration:**
   - Number of users
   - Calls per user
   - Number of rounds

2. **Memory statistics:**
   - Initial memory
   - Final memory
   - Peak memory
   - Growth percentage

3. **Attach files:**
   - Memory CSV file
   - Load test log
   - Server log

4. **Environment:**
   - OS and version
   - Python version
   - NAT version (`nat --version`)
   - Available system memory

## Next Steps

Once you've reproduced the issue:

1. **Try different workloads** - Test with different numbers of users/calls
2. **Identify patterns** - Which operations cause the most growth?
3. **Profile code** - Use Python profilers for detailed analysis
4. **Share results** - Provide test results to the development team

For detailed documentation, see: `debug_tools/README_MEMORY_LEAK_TEST.md`
