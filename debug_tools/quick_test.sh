#!/bin/bash
# Quick test script using Python load tester with CLI calls (no 406 errors)

set -e

CONFIG_FILE="${1:-examples/getting_started/simple_calculator/configs/config.yml}"

echo "=========================================="
echo "MCP Server Memory Leak Quick Test"
echo "=========================================="
echo ""

# Check if dependencies are installed
echo "Checking dependencies..."
python3 -c "import psutil" 2>/dev/null || {
    echo "Error: psutil not installed. Run: pip install -r debug_tools/requirements_memory_test.txt"
    exit 1
}

python3 -c "import aiohttp" 2>/dev/null || {
    echo "Error: aiohttp not installed. Run: pip install -r debug_tools/requirements_memory_test.txt"
    exit 1
}

python3 -c "import requests" 2>/dev/null || {
    echo "Error: requests not installed. Run: pip install -r debug_tools/requirements_memory_test.txt"
    exit 1
}

echo "✓ All dependencies installed"
echo ""

# Validate config file
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Usage: $0 [config_file]"
    exit 1
fi

echo "Using config file: $CONFIG_FILE"
echo ""

# Run the Python-based integrated test with CLI mode
echo "Starting integrated memory leak test..."
echo "This will:"
echo "  1. Start MCP server"
echo "  2. Monitor memory usage"
echo "  3. Run 3 rounds of load tests (40 users, 10 CLI calls each)"
echo "  4. Use proper MCP protocol (no 406 errors)"
echo "  5. Generate analysis report"
echo ""
echo "Press Ctrl+C to stop the test early"
echo ""

python3 debug_tools/run_memory_leak_test.py \
    --config_file "$CONFIG_FILE" \
    --users 40 \
    --calls 100 \
    --rounds 3 \
    --delay 10

echo ""
echo "=========================================="
echo "Test completed!"
echo "Check debug_tools/memory_leak_test_results/ for detailed results"
echo "=========================================="
