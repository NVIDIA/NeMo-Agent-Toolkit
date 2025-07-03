#!/bin/bash

# Test script for Code Execution Sandbox
# This script tests various scenarios using cURL requests

set -e  # Exit on any error

# Configuration
SANDBOX_URL="http://127.0.0.1:6000/execute"
TIMEOUT=30

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to test a code execution request
test_code_execution() {
    local test_name="$1"
    local code="$2"
    local expected_status="$3"
    
    print_status "Testing: $test_name"
    
    # Create JSON payload
    local json_payload=$(cat << EOF
{
    "generated_code": $(echo "$code" | jq -R -s .),
    "timeout": $TIMEOUT,
    "language": "python"
}
EOF
)
    
    # Make the request
    local response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$json_payload" \
        "$SANDBOX_URL")
    
    # Parse response
    local status=$(echo "$response" | jq -r '.process_status // "unknown"')
    local stdout=$(echo "$response" | jq -r '.stdout // ""')
    local stderr=$(echo "$response" | jq -r '.stderr // ""')
    
    echo "Response: $response"
    echo "Status: $status"
    echo "Stdout: $stdout"
    echo "Stderr: $stderr"
    
    # Check if status matches expected
    if [ "$status" = "$expected_status" ]; then
        print_success "$test_name - Status matches expected: $status"
    else
        print_error "$test_name - Status mismatch. Expected: $expected_status, Got: $status"
    fi
    
    echo "----------------------------------------"
}

# Function to check if sandbox is running
check_sandbox_running() {
    print_status "Checking if sandbox server is running..."
    
    if curl -s --connect-timeout 5 "$SANDBOX_URL" > /dev/null 2>&1; then
        print_success "Sandbox server is running at $SANDBOX_URL"
        return 0
    else
        print_error "Sandbox server is not running at $SANDBOX_URL"
        print_warning "Please start the sandbox server first:"
        print_warning "cd src/aiq/tool/code_execution/local_sandbox && ./start_local_sandbox.sh"
        return 1
    fi
}

# Main test function
run_tests() {
    print_status "Starting Code Execution Sandbox Tests"
    echo "========================================"
    
    # Check if jq is available
    if ! command -v jq &> /dev/null; then
        print_error "jq is required but not installed. Please install jq first."
        exit 1
    fi
    
    # Check if sandbox is running
    if ! check_sandbox_running; then
        exit 1
    fi
    
    # Test 1: Simple print statement
    test_code_execution "Simple Print" \
        "print('Hello, World!')" \
        "completed"
    
    # Test 2: Basic arithmetic
    test_code_execution "Basic Arithmetic" \
        "result = 2 + 3
print(f'Result: {result}')" \
        "completed"
    
    # Test 3: Using numpy (test dependencies)
    test_code_execution "Numpy Operations" \
        "import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(f'Array: {arr}')
print(f'Mean: {np.mean(arr)}')" \
        "completed"
    
    # Test 4: Using pandas (test dependencies)
    test_code_execution "Pandas Operations" \
        "import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df)
print(f'Sum of column A: {df[\"A\"].sum()}')" \
        "completed"
    
    # Test 5: Using plotly (test new dependency)
    test_code_execution "Plotly Import" \
        "import plotly.graph_objects as go
print('Plotly imported successfully')
fig = go.Figure()
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
print('Plot created successfully')" \
        "completed"
    
    # Test 6: Error handling - syntax error
    test_code_execution "Syntax Error" \
        "print('Hello World'
# Missing closing parenthesis" \
        "error"
    
    # Test 7: Error handling - runtime error
    test_code_execution "Runtime Error" \
        "x = 1 / 0
print('This should not print')" \
        "error"
    
    # Test 8: Error handling - import error
    test_code_execution "Import Error" \
        "import nonexistent_module
print('This should not print')" \
        "error"
    
    # Test 9: Code with both stdout and stderr
    test_code_execution "Mixed Output" \
        "import sys
print('This goes to stdout')
print('This goes to stderr', file=sys.stderr)
print('Back to stdout')" \
        "completed"
    
    # Test 10: Long running code (should complete within timeout)
    test_code_execution "Long Running Code" \
        "import time
for i in range(3):
    print(f'Iteration {i}')
    time.sleep(0.5)
print('Completed')" \
        "completed"
    
    # Test 11: Test dictionary input format (as mentioned in the code)
    test_code_execution "Dictionary Input Format" \
        "{'generated_code': 'print(\"Hello from dict format\")'}" \
        "completed"
    
    # Test 12: Test code with backticks (markdown format)
    test_code_execution "Markdown Code Format" \
        "\`\`\`python
print('Hello from markdown format')
\`\`\`" \
        "completed"
    
    # Test 13: File operations (test working directory)
    test_code_execution "File Operations" \
        "import os
print(f'Current directory: {os.getcwd()}')
with open('test_file.txt', 'w') as f:
    f.write('Hello, World!')
with open('test_file.txt', 'r') as f:
    content = f.read()
print(f'File content: {content}')
os.remove('test_file.txt')
print('File operations completed')" \
        "completed"
    
    # Test 14: File persistence - Create multiple file types
    test_code_execution "File Persistence - Create Files" \
        "import os
import pandas as pd
import numpy as np
print('Current directory:', os.getcwd())
print('Directory contents:', os.listdir('.')) 

# Create a test file
with open('persistence_test.txt', 'w') as f:
    f.write('Hello from sandbox persistence test!')

# Create a CSV file
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
df.to_csv('persistence_test.csv', index=False)

# Create a numpy array file
arr = np.array([1, 2, 3, 4, 5])
np.save('persistence_test.npy', arr)

print('Files created:')
for file in os.listdir('.'):
    if 'persistence_test' in file:
        print('  -', file)" \
        "completed"
    
    # Test 15: File persistence - Read back created files
    test_code_execution "File Persistence - Read Files" \
        "import pandas as pd
import numpy as np

# Read back the files we created
print('=== Reading persistence_test.txt ===')
with open('persistence_test.txt', 'r') as f:
    content = f.read()
    print(f'Content: {content}')

print('\\n=== Reading persistence_test.csv ===')
df = pd.read_csv('persistence_test.csv')
print(df)
print(f'DataFrame shape: {df.shape}')

print('\\n=== Reading persistence_test.npy ===')
arr = np.load('persistence_test.npy')
print(f'Array: {arr}')
print(f'Array sum: {np.sum(arr)}')

print('\\n=== File persistence test PASSED! ===')" \
        "completed"
    
    # Test 16: File persistence - JSON operations
    test_code_execution "File Persistence - JSON Operations" \
        "import json
import os

# Create a complex JSON file
data = {
    'test_name': 'sandbox_persistence',
    'timestamp': '2024-07-03',
    'results': {
        'numpy_test': True,
        'pandas_test': True,
        'file_operations': True
    },
    'metrics': [1.5, 2.3, 3.7, 4.1],
    'metadata': {
        'working_dir': os.getcwd(),
        'python_version': '3.x'
    }
}

# Save JSON file
with open('persistence_test.json', 'w') as f:
    json.dump(data, f, indent=2)

# Read it back
with open('persistence_test.json', 'r') as f:
    loaded_data = json.load(f)

print('JSON file created and loaded successfully')
print(f'Test name: {loaded_data[\"test_name\"]}')
print(f'Results count: {len(loaded_data[\"results\"])}')
print(f'Metrics: {loaded_data[\"metrics\"]}')
print('JSON persistence test completed!')" \
        "completed"
    
    print_status "All tests completed!"
}

# Function to test malformed requests
test_malformed_requests() {
    print_status "Testing malformed requests..."
    
    # Test missing generated_code field
    print_status "Testing missing generated_code field"
    response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d '{"timeout": 10, "language": "python"}' \
        "$SANDBOX_URL")
    echo "Response: $response"
    
    # Test missing timeout field
    print_status "Testing missing timeout field"
    response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d '{"generated_code": "print(\"test\")", "language": "python"}' \
        "$SANDBOX_URL")
    echo "Response: $response"
    
    # Test invalid JSON
    print_status "Testing invalid JSON"
    response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d '{"generated_code": "print("test")", "timeout": 10}' \
        "$SANDBOX_URL")
    echo "Response: $response"
    
    # Test non-JSON request
    print_status "Testing non-JSON request"
    response=$(curl -s -X POST \
        -H "Content-Type: text/plain" \
        -d 'This is not JSON' \
        "$SANDBOX_URL")
    echo "Response: $response"
    
    echo "----------------------------------------"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -u, --url URL     Sandbox URL (default: $SANDBOX_URL)"
    echo "  -t, --timeout N   Timeout in seconds (default: $TIMEOUT)"
    echo "  -m, --malformed   Test malformed requests"
    echo "  -h, --help        Show this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--url)
            SANDBOX_URL="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -m|--malformed)
            test_malformed_requests
            exit 0
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Run the tests
run_tests 