<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Code Execution Sandbox

A secure, containerized Python code execution environment that allows safe execution of Python code with comprehensive error handling and debugging capabilities.

## Overview

The Code Execution Sandbox provides:
- **Secure code execution** in isolated Docker containers
- **Multiple input formats** including raw code, dictionary format, and markdown
- **Dependency management** with pre-installed libraries
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
pytest test_code_execution_sandbox.py
```

Note: a running instance of a local sandbox is required.

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

## Security Considerations

- **Isolated execution**: All code runs in Docker containers
- **Resource limits**: Memory and CPU limits prevent resource exhaustion
- **Network isolation**: Containers have limited network access
- **File system isolation**: Mounted volumes provide controlled file access
- **Process isolation**: Each execution runs in a separate process
