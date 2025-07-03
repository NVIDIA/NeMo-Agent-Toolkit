# Code Execution Sandbox

A secure, containerized Python code execution environment that allows safe execution of Python code with comprehensive error handling and debugging capabilities.

## Overview

The Code Execution Sandbox provides:
- **Secure code execution** in isolated Docker containers
- **Comprehensive error handling** with detailed stdout/stderr capture
- **Multiple input formats** including raw code, dictionary format, and markdown
- **Dependency management** with pre-installed libraries (numpy, pandas, plotly, etc.)
- **Flexible configuration** with customizable timeouts and output limits
- **Robust debugging** with extensive logging and error reporting

## Quick Start

### Step 1: Start the Sandbox Server

Navigate to the local sandbox directory and start the server:

```bash
cd src/aiq/tool/code_execution/local_sandbox
./start_local_sandbox.sh
```

The script will:
- Build the Docker image if it doesn't exist
- Start the sandbox server on port 6000
- Mount your working directory for file operations

#### Advanced Usage:
```bash
# Custom container name
./start_local_sandbox.sh my-sandbox

# Custom output directory
./start_local_sandbox.sh my-sandbox /path/to/output

# Using environment variable
export OUTPUT_DATA_PATH=/path/to/output
./start_local_sandbox.sh
```

### Step 2: Test the Installation

Run the comprehensive test suite to verify everything is working:

```bash
cd src/aiq/tool/code_execution
./test_code_execution_sandbox.sh
```

## Using the Code Execution Tool

### Basic Usage

The sandbox accepts HTTP POST requests to `http://localhost:6000/execute` with JSON payloads:

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "generated_code": "print(\"Hello, World!\")",
    "timeout": 30,
    "language": "python"
  }' \
  http://localhost:6000/execute
```

### Supported Input Formats

#### 1. Raw Python Code
```json
{
  "generated_code": "import numpy as np\nprint(np.array([1, 2, 3]))",
  "timeout": 30,
  "language": "python"
}
```

#### 2. Dictionary Format
```json
{
  "generated_code": "{'generated_code': 'print(\"Hello from dict format\")'}",
  "timeout": 30,
  "language": "python"
}
```

#### 3. Markdown Code Blocks
```json
{
  "generated_code": "```python\nprint('Hello from markdown')\n```",
  "timeout": 30,
  "language": "python"
}
```

### Response Format

The sandbox returns JSON responses with the following structure:

```json
{
  "process_status": "completed|error|timeout",
  "stdout": "Standard output content",
  "stderr": "Standard error content"
}
```

### Available Libraries

The sandbox comes pre-installed with:
- **numpy** - Numerical computing
- **pandas** - Data manipulation and analysis
- **scipy** - Scientific computing
- **ipython** - Enhanced interactive Python
- **plotly** - Interactive visualizations

## Configuration Options

### Sandbox Configuration

- **URI**: Default `http://127.0.0.1:6000`
- **Timeout**: Default 10 seconds (configurable)
- **Max Output Characters**: Default 1000 characters
- **Memory Limit**: 10GB (configurable in Docker)
- **Working Directory**: Mounted volume for file operations

### Environment Variables

- `OUTPUT_DATA_PATH`: Custom path for file operations
- `SANDBOX_HOST`: Custom sandbox host
- `SANDBOX_PORT`: Custom sandbox port

## Testing and Validation

### Comprehensive Test Suite

The repository includes a comprehensive test suite with 13 different scenarios:

```bash
# Run all tests
./test_code_execution_sandbox.sh

# Run with custom settings
./test_code_execution_sandbox.sh -u http://localhost:8000/execute -t 60

# Test error handling only
./test_code_execution_sandbox.sh -m

# Show help
./test_code_execution_sandbox.sh -h
```

### Test Categories

1. **Basic Operations**
   - Simple print statements
   - Arithmetic operations
   - String manipulation

2. **Library Testing**
   - Numpy array operations
   - Pandas DataFrame operations
   - Plotly visualization creation

3. **Error Handling**
   - Syntax errors
   - Runtime exceptions
   - Import errors

4. **Advanced Features**
   - Mixed stdout/stderr output
   - Long-running code execution
   - File system operations
   - Input format variations

### Specific Test Examples

Run specific test scenarios:

```bash
# Test plotly visualization
./curl_request.sh

# Test malformed requests
./test_code_execution_sandbox.sh -m
```

## Troubleshooting

### Common Issues

#### 1. Server Not Starting
```bash
# Check if Docker is running
docker ps

# Check port availability
lsof -i :6000

# View Docker logs
docker logs local-sandbox
```

#### 2. Permission Errors
```bash
# Make scripts executable
chmod +x test_code_execution_sandbox.sh curl_request.sh
chmod +x local_sandbox/start_local_sandbox.sh
```

#### 3. Missing Dependencies
```bash
# Install jq for JSON processing
brew install jq  # macOS
sudo apt-get install jq  # Ubuntu/Debian
```

#### 4. Timeout Issues
- Increase timeout values for complex operations
- Check system resources (CPU, memory)
- Verify Docker container health

### Debug Information

The sandbox provides extensive debugging information:

```bash
# Check debug logs in the sandbox output
docker logs local-sandbox

# Enable verbose logging
export DEBUG=1
./start_local_sandbox.sh
```

## Integration Examples

### Python Integration

```python
import requests
import json

def execute_code(code, timeout=30):
    payload = {
        "generated_code": code,
        "timeout": timeout,
        "language": "python"
    }
    
    response = requests.post(
        "http://localhost:6000/execute",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    return response.json()

# Example usage
result = execute_code("print('Hello from Python!')")
print(f"Status: {result['process_status']}")
print(f"Output: {result['stdout']}")
```

### Shell Integration

```bash
#!/bin/bash

execute_python_code() {
    local code="$1"
    local timeout="${2:-30}"
    
    curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "{\"generated_code\": $(echo "$code" | jq -R -s .), \"timeout\": $timeout, \"language\": \"python\"}" \
        "http://localhost:6000/execute"
}

# Example usage
result=$(execute_python_code "import numpy as np; print(np.version.version)")
echo "$result" | jq -r '.stdout'
```

## Security Considerations

- **Isolated execution**: All code runs in Docker containers
- **Resource limits**: Memory and CPU limits prevent resource exhaustion
- **Network isolation**: Containers have limited network access
- **File system isolation**: Mounted volumes provide controlled file access
- **Process isolation**: Each execution runs in a separate process

## Performance Optimization

- **Container reuse**: Docker containers are reused for better performance
- **Connection pooling**: HTTP connections are pooled for efficiency
- **Memory management**: Automatic cleanup of execution environments
- **Timeout handling**: Configurable timeouts prevent hanging processes

## Contributing

When adding new functionality:

1. **Follow existing patterns** for error handling and logging
2. **Add comprehensive tests** for new features
3. **Update documentation** with usage examples
4. **Test edge cases** and error conditions
5. **Maintain backward compatibility** where possible

### Adding New Tests

```bash
# Add test function to test_code_execution_sandbox.sh
test_code_execution "New Feature Test" \
    "your_test_code_here" \
    "expected_status"
```

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.
