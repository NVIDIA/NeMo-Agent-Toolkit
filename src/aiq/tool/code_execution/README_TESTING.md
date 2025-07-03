# Code Execution Sandbox Testing

This directory contains test scripts for validating the code execution sandbox functionality.

## Test Scripts

### 1. `test_code_execution_sandbox.sh`
A comprehensive test suite that validates various aspects of the code execution sandbox.

#### Features:
- **13 different test scenarios** covering:
  - Basic Python operations
  - Dependency testing (numpy, pandas, plotly)
  - Error handling (syntax, runtime, import errors)
  - Mixed stdout/stderr output
  - Long-running code execution
  - Dictionary and markdown input formats
  - File operations
- **Malformed request testing**
- **Colored output** for easy result interpretation
- **Configurable timeout and URL**

#### Usage:
```bash
# Run all tests
./test_code_execution_sandbox.sh

# Run with custom URL
./test_code_execution_sandbox.sh -u http://localhost:8000/execute

# Run with custom timeout
./test_code_execution_sandbox.sh -t 60

# Test malformed requests only
./test_code_execution_sandbox.sh -m

# Show help
./test_code_execution_sandbox.sh -h
```

#### Prerequisites:
- `jq` command-line JSON processor
- Code execution sandbox server running
- `curl` command

### 2. `curl_request.sh`
A specific test script for testing complex code execution scenarios with real-world examples.

#### Features:
- Tests complex nested JSON payloads
- Demonstrates plotly visualization code execution
- Shows both wrapper and direct code execution approaches

#### Usage:
```bash
./curl_request.sh
```

## Running the Tests

### Step 1: Start the Sandbox Server
```bash
cd src/aiq/tool/code_execution/local_sandbox
./start_local_sandbox.sh
```

### Step 2: Run Tests
```bash
cd src/aiq/tool/code_execution
./test_code_execution_sandbox.sh
```

## Test Results Interpretation

The comprehensive test script provides colored output:
- 🔵 **[INFO]** - General information and test progress
- 🟢 **[SUCCESS]** - Test passed successfully
- 🔴 **[ERROR]** - Test failed or error occurred
- 🟡 **[WARNING]** - Warning or important notice

Each test displays:
- Full JSON response from the sandbox
- Execution status (`completed`, `error`, `timeout`)
- Standard output content
- Standard error content
- Pass/fail status based on expected results

## Expected Test Results

When all systems are working correctly:
- Tests 1-5, 9-13: Should return `completed` status
- Tests 6-8: Should return `error` status (intentional error testing)
- All malformed request tests should return appropriate error messages

## Troubleshooting

### Common Issues:

1. **"jq is required but not installed"**
   ```bash
   # macOS
   brew install jq
   
   # Ubuntu/Debian
   sudo apt-get install jq
   ```

2. **"Sandbox server is not running"**
   - Make sure the sandbox server is started
   - Check if port 6000 is available
   - Verify Docker is running (for local sandbox)

3. **Tests timing out**
   - Increase timeout with `-t` option
   - Check system resources
   - Verify Docker container is healthy

4. **Permission denied**
   ```bash
   chmod +x test_code_execution_sandbox.sh curl_request.sh
   ```

## Test Coverage

The test suite covers:
- ✅ Basic code execution
- ✅ Python standard library usage
- ✅ Third-party dependencies (numpy, pandas, plotly)
- ✅ Error handling and recovery
- ✅ Input format variations
- ✅ Output stream handling
- ✅ File system operations
- ✅ Timeout behavior
- ✅ JSON validation
- ✅ HTTP error responses

## Contributing

When adding new tests:
1. Follow the existing test function pattern
2. Include expected status in test calls
3. Add appropriate documentation
4. Test both success and failure scenarios
5. Update this README with new test descriptions
