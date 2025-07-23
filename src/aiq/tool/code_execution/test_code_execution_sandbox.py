# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test suite for Code Execution Sandbox using pytest.

This module provides comprehensive testing for the code execution sandbox service,
replacing the original bash script with a more maintainable Python implementation.
"""

import os
from typing import Any

import pytest
import requests
from requests.exceptions import ConnectionError
from requests.exceptions import RequestException
from requests.exceptions import Timeout


class TestCodeExecutionSandbox:
    """Test suite for the Code Execution Sandbox service."""

    @pytest.fixture(scope="class")
    def sandbox_config(self):
        """Configuration for sandbox testing."""
        return {
            "url": os.environ.get("SANDBOX_URL", "http://127.0.0.1:6000/execute"),
            "timeout": int(os.environ.get("SANDBOX_TIMEOUT", "30")),
            "connection_timeout": 5
        }

    @pytest.fixture(scope="class", autouse=True)
    def check_sandbox_running(self, sandbox_config):
        """Check if sandbox server is running before running tests."""
        try:
            _ = requests.get(sandbox_config["url"], timeout=sandbox_config["connection_timeout"])
            print(f"✓ Sandbox server is running at {sandbox_config['url']}")
        except (ConnectionError, Timeout, RequestException):
            pytest.skip(
                f"Sandbox server is not running at {sandbox_config['url']}. "
                "Please start it with: cd src/aiq/tool/code_execution/local_sandbox && ./start_local_sandbox.sh")

    def execute_code(self, sandbox_config: dict[str, Any], code: str, language: str = "python") -> dict[str, Any]:
        """
        Execute code in the sandbox and return the response.

        Args:
            sandbox_config: Configuration dictionary
            code: Code to execute
            language: Programming language (default: python)

        Returns:
            dictionary containing the response from the sandbox
        """
        payload = {"generated_code": code, "timeout": sandbox_config["timeout"], "language": language}

        response = requests.post(
            sandbox_config["url"],
            json=payload,
            timeout=sandbox_config["timeout"] + 5  # Add buffer to request timeout
        )

        # Ensure we got a response
        response.raise_for_status()
        return response.json()

    def test_simple_print(self, sandbox_config):
        """Test simple print statement execution."""
        code = "print('Hello, World!')"
        result = self.execute_code(sandbox_config, code)

        assert result["process_status"] == "completed"
        assert "Hello, World!" in result["stdout"]
        assert result["stderr"] == ""

    def test_basic_arithmetic(self, sandbox_config):
        """Test basic arithmetic operations."""
        code = """
result = 2 + 3
print(f'Result: {result}')
"""
        result = self.execute_code(sandbox_config, code)

        assert result["process_status"] == "completed"
        assert "Result: 5" in result["stdout"]
        assert result["stderr"] == ""

    def test_numpy_operations(self, sandbox_config):
        """Test numpy dependency availability and operations."""
        code = """
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(f'Array: {arr}')
print(f'Mean: {np.mean(arr)}')
"""
        result = self.execute_code(sandbox_config, code)

        assert result["process_status"] == "completed"
        assert "Array: [1 2 3 4 5]" in result["stdout"]
        assert "Mean: 3.0" in result["stdout"]
        assert result["stderr"] == ""

    def test_pandas_operations(self, sandbox_config):
        """Test pandas dependency availability and operations."""
        code = """
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df)
print(f'Sum of column A: {df["A"].sum()}')
"""
        result = self.execute_code(sandbox_config, code)

        assert result["process_status"] == "completed"
        assert "Sum of column A: 6" in result["stdout"]
        assert result["stderr"] == ""

    def test_plotly_import(self, sandbox_config):
        """Test plotly dependency availability."""
        code = """
import plotly.graph_objects as go
print('Plotly imported successfully')
fig = go.Figure()
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
print('Plot created successfully')
"""
        result = self.execute_code(sandbox_config, code)

        assert result["process_status"] == "completed"
        assert "Plotly imported successfully" in result["stdout"]
        assert "Plot created successfully" in result["stdout"]
        assert result["stderr"] == ""

    def test_syntax_error_handling(self, sandbox_config):
        """Test handling of syntax errors."""
        code = """
print('Hello World'
# Missing closing parenthesis
"""
        result = self.execute_code(sandbox_config, code)

        assert result["process_status"] == "error"
        assert "SyntaxError" in result["stderr"] or "SyntaxError" in result["stdout"]

    def test_runtime_error_handling(self, sandbox_config):
        """Test handling of runtime errors."""
        code = """
x = 1 / 0
print('This should not print')
"""
        result = self.execute_code(sandbox_config, code)

        assert result["process_status"] == "error"
        assert "ZeroDivisionError" in result["stderr"] or "ZeroDivisionError" in result["stdout"]

    def test_import_error_handling(self, sandbox_config):
        """Test handling of import errors."""
        code = """
import nonexistent_module
print('This should not print')
"""
        result = self.execute_code(sandbox_config, code)

        assert result["process_status"] == "error"
        assert "ModuleNotFoundError" in result["stderr"] or "ImportError" in result["stderr"]

    def test_mixed_output(self, sandbox_config):
        """Test code that produces both stdout and stderr output."""
        code = """
import sys
print('This goes to stdout')
print('This goes to stderr', file=sys.stderr)
print('Back to stdout')
"""
        result = self.execute_code(sandbox_config, code)

        assert result["process_status"] == "completed"
        assert "This goes to stdout" in result["stdout"]
        assert "Back to stdout" in result["stdout"]
        assert "This goes to stderr" in result["stderr"]

    def test_long_running_code(self, sandbox_config):
        """Test code that takes some time to execute but completes within timeout."""
        code = """
import time
for i in range(3):
    print(f'Iteration {i}')
    time.sleep(0.5)
print('Completed')
"""
        result = self.execute_code(sandbox_config, code)

        assert result["process_status"] == "completed"
        assert "Iteration 0" in result["stdout"]
        assert "Iteration 1" in result["stdout"]
        assert "Iteration 2" in result["stdout"]
        assert "Completed" in result["stdout"]
        assert result["stderr"] == ""

    def test_file_operations(self, sandbox_config):
        """Test basic file operations in the sandbox."""
        code = """
import os
print(f'Current directory: {os.getcwd()}')
with open('test_file.txt', 'w') as f:
    f.write('Hello, World!')
with open('test_file.txt', 'r') as f:
    content = f.read()
print(f'File content: {content}')
os.remove('test_file.txt')
print('File operations completed')
"""
        result = self.execute_code(sandbox_config, code)

        assert result["process_status"] == "completed"
        assert "File content: Hello, World!" in result["stdout"]
        assert "File operations completed" in result["stdout"]
        assert result["stderr"] == ""

    def test_file_persistence_create(self, sandbox_config):
        """Test file persistence - create various file types."""
        code = """
import os
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
        print('  -', file)
"""
        result = self.execute_code(sandbox_config, code)

        assert result["process_status"] == "completed"
        assert "persistence_test.txt" in result["stdout"]
        assert "persistence_test.csv" in result["stdout"]
        assert "persistence_test.npy" in result["stdout"]
        assert result["stderr"] == ""

    def test_file_persistence_read(self, sandbox_config):
        """Test file persistence - read back created files."""
        code = """
import pandas as pd
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

print('\\n=== File persistence test PASSED! ===')
"""
        result = self.execute_code(sandbox_config, code)

        assert result["process_status"] == "completed"
        assert "Content: Hello from sandbox persistence test!" in result["stdout"]
        assert "DataFrame shape: (3, 2)" in result["stdout"]
        assert "Array: [1 2 3 4 5]" in result["stdout"]
        assert "Array sum: 15" in result["stdout"]
        assert "File persistence test PASSED!" in result["stdout"]
        assert result["stderr"] == ""

    def test_json_operations(self, sandbox_config):
        """Test JSON file operations for persistence."""
        code = """
import json
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
print(f'Test name: {loaded_data["test_name"]}')
print(f'Results count: {len(loaded_data["results"])}')
print(f'Metrics: {loaded_data["metrics"]}')
print('JSON persistence test completed!')
"""
        result = self.execute_code(sandbox_config, code)

        assert result["process_status"] == "completed"
        assert "JSON file created and loaded successfully" in result["stdout"]
        assert "Test name: sandbox_persistence" in result["stdout"]
        assert "Results count: 3" in result["stdout"]
        assert "JSON persistence test completed!" in result["stdout"]
        assert result["stderr"] == ""

    def test_missing_generated_code_field(self, sandbox_config):
        """Test request missing the generated_code field."""
        payload = {"timeout": 10, "language": "python"}

        response = requests.post(sandbox_config["url"], json=payload)

        # Should return an error status code or error in response
        assert response.status_code != 200 or "error" in response.json()

    def test_missing_timeout_field(self, sandbox_config):
        """Test request missing the timeout field."""
        payload = {"generated_code": "print('test')", "language": "python"}

        response = requests.post(sandbox_config["url"], json=payload)

        # Should return error for missing timeout field
        result = response.json()
        assert response.status_code == 400 and result["process_status"] == "error"

    def test_invalid_json(self, sandbox_config):
        """Test request with invalid JSON."""
        invalid_json = '{"generated_code": "print("test")", "timeout": 10}'

        response = requests.post(sandbox_config["url"], data=invalid_json, headers={"Content-Type": "application/json"})

        # Should return error for invalid JSON
        assert response.status_code != 200

    def test_non_json_request(self, sandbox_config):
        """Test request with non-JSON content."""
        response = requests.post(sandbox_config["url"], data="This is not JSON", headers={"Content-Type": "text/plain"})

        # Should return error for non-JSON content
        assert response.status_code != 200

    def test_timeout_too_low(self, sandbox_config):
        """Test request with timeout too low."""
        code = """
import time
time.sleep(2.0)
"""
        payload = {"generated_code": code, "timeout": 1, "language": "python"}
        response = requests.post(sandbox_config["url"], json=payload)
        assert response.json()["process_status"] == "timeout"
        assert response.status_code == 200


# Pytest configuration and fixtures for command-line options
def pytest_addoption(parser):
    """Add custom command-line options for pytest."""
    parser.addoption("--sandbox-url",
                     action="store",
                     default="http://127.0.0.1:6000/execute",
                     help="Sandbox URL for testing")
    parser.addoption("--sandbox-timeout",
                     action="store",
                     type=int,
                     default=30,
                     help="Timeout in seconds for sandbox operations")


@pytest.fixture(scope="session", autouse=True)
def setup_environment(request):
    """Setup environment variables from command-line options."""
    os.environ["SANDBOX_URL"] = request.config.getoption("--sandbox-url", "http://127.0.0.1:6000/execute")
    os.environ["SANDBOX_TIMEOUT"] = str(request.config.getoption("--sandbox-timeout", 30))


if __name__ == "__main__":
    # Allow running as a script
    pytest.main([__file__, "-v"])
