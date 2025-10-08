#!/bin/bash
# Automated test runner for text2sql MCP server load testing and memory leak detection
#
# This script:
# 1. Starts the MCP server in the background
# 2. Waits for server to be ready
# 3. Runs the load test with memory monitoring
# 4. Generates a report
# 5. Cleans up the server

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONFIG_FILE="${CONFIG_FILE:-examples/text2sql/config_text2sql_mcp.yml}"
SERVER_HOST="${SERVER_HOST:-localhost}"
SERVER_PORT="${SERVER_PORT:-9901}"
SERVER_URL="http://${SERVER_HOST}:${SERVER_PORT}/mcp"

# Load test parameters
NUM_USERS="${NUM_USERS:-20}"
CALLS_PER_USER="${CALLS_PER_USER:-10}"
NUM_ROUNDS="${NUM_ROUNDS:-3}"
DELAY_BETWEEN_ROUNDS="${DELAY_BETWEEN_ROUNDS:-5.0}"

# Output files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="examples/text2sql/logs"
SERVER_LOG="${LOG_DIR}/server_${TIMESTAMP}.log"
TEST_LOG="${LOG_DIR}/loadtest_${TIMESTAMP}.log"
MEMORY_LOG="${LOG_DIR}/memory_${TIMESTAMP}.log"

# Create log directory
mkdir -p "${LOG_DIR}"

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}  Text2SQL MCP Server Load Test & Memory Leak Detection${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""
echo -e "Configuration:"
echo -e "  Config file:        ${CONFIG_FILE}"
echo -e "  Server URL:         ${SERVER_URL}"
echo -e "  Concurrent users:   ${NUM_USERS}"
echo -e "  Calls per user:     ${CALLS_PER_USER}"
echo -e "  Test rounds:        ${NUM_ROUNDS}"
echo -e "  Delay between:      ${DELAY_BETWEEN_ROUNDS}s"
echo ""
echo -e "Logs:"
echo -e "  Server log:         ${SERVER_LOG}"
echo -e "  Test log:           ${TEST_LOG}"
echo -e "  Memory log:         ${MEMORY_LOG}"
echo ""

# Function to cleanup on exit
cleanup() {
    local exit_code=$?
    echo ""
    echo -e "${YELLOW}Cleaning up...${NC}"

    if [ -n "${SERVER_PID}" ]; then
        echo "Stopping MCP server (PID: ${SERVER_PID})"
        kill ${SERVER_PID} 2>/dev/null || true
        wait ${SERVER_PID} 2>/dev/null || true
    fi

    if [ -n "${MEMORY_PID}" ]; then
        echo "Stopping memory monitor (PID: ${MEMORY_PID})"
        kill ${MEMORY_PID} 2>/dev/null || true
        wait ${MEMORY_PID} 2>/dev/null || true
    fi

    if [ ${exit_code} -eq 0 ]; then
        echo -e "${GREEN}✓ Test completed successfully${NC}"
    else
        echo -e "${RED}✗ Test failed with exit code ${exit_code}${NC}"
    fi

    exit ${exit_code}
}

trap cleanup EXIT INT TERM

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

# Check if nat command is available
if ! command -v nat &> /dev/null; then
    echo -e "${RED}Error: 'nat' command not found${NC}"
    echo "Please install NeMo Agent Toolkit: pip install -e .[mcp]"
    exit 1
fi

# Check if config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo -e "${RED}Error: Config file not found: ${CONFIG_FILE}${NC}"
    exit 1
fi

# Check environment variables
if [ -z "${NVIDIA_API_KEY}" ]; then
    echo -e "${YELLOW}Warning: NVIDIA_API_KEY not set${NC}"
    echo "Please set it: export NVIDIA_API_KEY='your-api-key'"
    echo "Continuing anyway in case it's configured differently..."
fi

echo -e "${GREEN}✓ Prerequisites checked${NC}"
echo ""

# Step 1: Start MCP server
echo -e "${BLUE}Step 1: Starting MCP server...${NC}"

nat mcp serve \
    --config_file "${CONFIG_FILE}" \
    --host "${SERVER_HOST}" \
    --port "${SERVER_PORT}" \
    --log_level INFO \
    > "${SERVER_LOG}" 2>&1 &

SERVER_PID=$!

echo "Server PID: ${SERVER_PID}"
echo "Server log: ${SERVER_LOG}"

# Wait for server to be ready
echo -n "Waiting for server to start"
MAX_WAIT=60
WAIT_COUNT=0

while [ ${WAIT_COUNT} -lt ${MAX_WAIT} ]; do
    if curl -s -f "http://${SERVER_HOST}:${SERVER_PORT}/health" > /dev/null 2>&1; then
        echo ""
        echo -e "${GREEN}✓ Server is ready${NC}"
        break
    fi

    # Check if server process is still running
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo ""
        echo -e "${RED}✗ Server process died${NC}"
        echo "Last 20 lines of server log:"
        tail -n 20 "${SERVER_LOG}"
        exit 1
    fi

    echo -n "."
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
done

if [ ${WAIT_COUNT} -eq ${MAX_WAIT} ]; then
    echo ""
    echo -e "${RED}✗ Server failed to start within ${MAX_WAIT} seconds${NC}"
    echo "Last 20 lines of server log:"
    tail -n 20 "${SERVER_LOG}"
    exit 1
fi

# Verify text2sql_standalone tool is available
echo "Verifying text2sql_standalone tool..."
if curl -s "http://${SERVER_HOST}:${SERVER_PORT}/debug/tools/list" | grep -q "text2sql_standalone"; then
    echo -e "${GREEN}✓ text2sql_standalone tool is available${NC}"
else
    echo -e "${RED}✗ text2sql_standalone tool not found${NC}"
    exit 1
fi

echo ""

# Step 2: Start memory monitoring (optional, if monitor script exists)
MONITOR_SCRIPT="examples/text2sql/monitor_server_memory.py"
if [ -f "${MONITOR_SCRIPT}" ]; then
    echo -e "${BLUE}Step 2: Starting memory monitor...${NC}"

    python "${MONITOR_SCRIPT}" \
        --pid ${SERVER_PID} \
        --interval 2 \
        > "${MEMORY_LOG}" 2>&1 &

    MEMORY_PID=$!
    echo "Memory monitor PID: ${MEMORY_PID}"
    echo "Memory log: ${MEMORY_LOG}"
    echo -e "${GREEN}✓ Memory monitor started${NC}"
    echo ""
else
    echo -e "${YELLOW}Skipping memory monitor (${MONITOR_SCRIPT} not found)${NC}"
    echo ""
fi

# Step 3: Run load test
echo -e "${BLUE}Step 3: Running load test...${NC}"
echo ""

python examples/text2sql/text2sql_load_test.py \
    --url "${SERVER_URL}" \
    --users ${NUM_USERS} \
    --calls ${CALLS_PER_USER} \
    --rounds ${NUM_ROUNDS} \
    --delay ${DELAY_BETWEEN_ROUNDS} \
    2>&1 | tee "${TEST_LOG}"

echo ""
echo -e "${GREEN}✓ Load test completed${NC}"
echo ""

# Step 4: Generate report
echo -e "${BLUE}Step 4: Generating report...${NC}"
echo ""

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}  TEST SUMMARY${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Extract key metrics from test log
if [ -f "${TEST_LOG}" ]; then
    echo "Load Test Results:"
    grep -E "(Total calls|Successful calls|Failed calls|Success rate|Throughput|response time|Performance|Round)" "${TEST_LOG}" | tail -n 20
    echo ""
fi

# Show memory usage if available
if [ -f "${MEMORY_LOG}" ] && [ -s "${MEMORY_LOG}" ]; then
    echo "Memory Usage:"
    tail -n 10 "${MEMORY_LOG}"
    echo ""

    # Check for memory leak indicators
    FIRST_MEM=$(head -n 5 "${MEMORY_LOG}" | grep -oE "[0-9]+\.[0-9]+ MB" | head -n 1 | grep -oE "[0-9]+\.[0-9]+")
    LAST_MEM=$(tail -n 5 "${MEMORY_LOG}" | grep -oE "[0-9]+\.[0-9]+ MB" | tail -n 1 | grep -oE "[0-9]+\.[0-9]+")

    if [ -n "${FIRST_MEM}" ] && [ -n "${LAST_MEM}" ]; then
        MEM_INCREASE=$(echo "${LAST_MEM} - ${FIRST_MEM}" | bc)
        MEM_PERCENT=$(echo "scale=2; (${MEM_INCREASE} / ${FIRST_MEM}) * 100" | bc)

        echo -e "Memory Analysis:"
        echo -e "  Initial:    ${FIRST_MEM} MB"
        echo -e "  Final:      ${LAST_MEM} MB"
        echo -e "  Increase:   ${MEM_INCREASE} MB (${MEM_PERCENT}%)"

        # Warn if memory increased significantly
        if (( $(echo "${MEM_PERCENT} > 20" | bc -l) )); then
            echo -e "  ${YELLOW}⚠️  Significant memory increase detected!${NC}"
            echo -e "  This may indicate a memory leak."
        elif (( $(echo "${MEM_PERCENT} > 10" | bc -l) )); then
            echo -e "  ${YELLOW}⚠️  Moderate memory increase detected.${NC}"
        else
            echo -e "  ${GREEN}✓ Memory usage appears stable.${NC}"
        fi
        echo ""
    fi
fi

# Show server errors if any
if [ -f "${SERVER_LOG}" ]; then
    ERROR_COUNT=$(grep -c -i "error" "${SERVER_LOG}" || true)
    if [ ${ERROR_COUNT} -gt 0 ]; then
        echo -e "${YELLOW}Server Errors: ${ERROR_COUNT} errors found${NC}"
        echo "Last 10 errors:"
        grep -i "error" "${SERVER_LOG}" | tail -n 10
        echo ""
    else
        echo -e "${GREEN}✓ No server errors found${NC}"
        echo ""
    fi
fi

echo -e "${BLUE}======================================================================${NC}"
echo ""
echo "Log files saved to:"
echo "  Server:     ${SERVER_LOG}"
echo "  Load test:  ${TEST_LOG}"
echo "  Memory:     ${MEMORY_LOG}"
echo ""

# Offer to view logs
echo -e "View logs? (s=server, t=test, m=memory, n=no)"
read -p "Choice: " -n 1 -r
echo ""

case "$REPLY" in
    s|S)
        less "${SERVER_LOG}"
        ;;
    t|T)
        less "${TEST_LOG}"
        ;;
    m|M)
        if [ -f "${MEMORY_LOG}" ]; then
            less "${MEMORY_LOG}"
        else
            echo "Memory log not available"
        fi
        ;;
    *)
        echo "Skipping log view"
        ;;
esac

echo ""
echo -e "${GREEN}Test complete! 🎉${NC}"
