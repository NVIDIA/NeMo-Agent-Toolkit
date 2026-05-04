# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest
import requests

CODE_BLOCKS = {
    "hello_world": {
        "code": "print('Hello, World!')", "expected_output": "Hello, World!"
    },
    "simple_addition": {
        "code": """
         result = 2 + 3
         print(f'Result: {result}')
         """,
        "expected_output": "Result: 5"
    },
    "numpy_mean": {
        "code":
            """
         import numpy as np
         arr = np.array([1, 2, 3, 4, 5])
         print(f'Array: {arr}')
         print(f'Mean: {np.mean(arr)}')
         """,
        "expected_output":
            "Mean: 3.0"
    },
    "pandas_operations": {
        "code":
            """
         import pandas as pd
         df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
         print(df)
         print(f'Sum of column A: {df["A"].sum()}')
         """,
        "expected_output":
            "Sum of column A: 6"
    },
    "plotly_import": {
        "code":
            """
         import plotly.graph_objects as go
         print('Plotly imported successfully')
         fig = go.Figure()
         fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
         print('Plot created successfully')
         """,
        "expected_output":
            "Plot created successfully"
    },
    "file_operations": {
        "code":
            """
         import os
         print(f'Current directory: {os.getcwd()}')
         with open('test_file.txt', 'w') as f:
             f.write('Hello, World!')
         with open('test_file.txt', 'r') as f:
             content = f.read()
         print(f'File content: {content}')
         os.remove('test_file.txt')
         print('File operations completed')
         """,
        "expected_output":
            "File operations completed"
    },
    "persistence_creation": {
        "code":
            """
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
         """,
        "expected_output":
            "persistence_test.npy"
    },
    "persistence_readback": {
        "code":
            """
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
         """,
        "expected_output":
            "File persistence test PASSED!"
    },
    "json_persistence": {
        "code":
            """
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
         """,
        "expected_output":
            "JSON persistence test completed!"
    }
}


@pytest.fixture(name="local_sandbox_url", scope="session", autouse=True)
def sandbox_url_fixture(local_sandbox_url: str) -> str:
    return local_sandbox_url


def _write_sandbox_workflow_config(tmp_path_factory: pytest.TempPathFactory, sandbox_url: str,
                                   sandbox_type: str) -> Path:
    config_path = tmp_path_factory.mktemp(f"{sandbox_type}_sandbox_workflow") / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(
            textwrap.dedent(f"""
            workflow:
                _type: code_execution
                uri: {sandbox_url}
                sandbox_type: {sandbox_type}
                timeout: 30
                max_output_characters: 3000
            """).strip())
    return config_path


@pytest.fixture(name="local_sandbox_workflow", scope="session")
def local_sandbox_workflow_fixture(local_sandbox_url: str, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_sandbox_workflow_config(tmp_path_factory, local_sandbox_url, sandbox_type="local")


@pytest.fixture(name="piston_sandbox_workflow", scope="session")
def piston_sandbox_workflow_fixture(piston_url: str, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_sandbox_workflow_config(tmp_path_factory, f"{piston_url.rstrip('/')}/execute", sandbox_type="piston")


def _mk_request(url: str, code: str, timeout: int, language: str = "python") -> requests.Response:
    payload = {"generated_code": code, "timeout": timeout, "language": language}

    response = requests.post(
        url,
        json=payload,
        timeout=timeout + 5  # Add buffer to request timeout
    )

    # Ensure we got a response
    response.raise_for_status()
    return response


def run_sandbox_code(sandbox_config: dict[str, Any], code: str, language: str = "python") -> dict[str, Any]:
    """
    Execute code in the sandbox and return the response.

    Args:
        sandbox_config: Configuration dictionary
        code: Code to execute
        language: Programming language (default: python)

    Returns:
        dictionary containing the response from the sandbox
    """
    response = _mk_request(url=sandbox_config["execute_url"],
                           code=code,
                           timeout=sandbox_config["timeout"],
                           language=language)
    return response.json()


def run_workflow_code(config_path: Path,
                      code: str,
                      timeout: int = 30,
                      language: str = "python",
                      workflow_url: str = "http://localhost:8000") -> dict[str, Any]:
    """
    Execute a workflow using the sandbox and return the response.
    """
    workflow_cmd = ["nat", "serve", "--config_file", str(config_path.absolute())]
    proc = subprocess.Popen(workflow_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.poll() is None, f"NAT server process failed to start: {proc.stdout.read()}"

    try:
        deadline = time.time() + 30  # 30 second timeout waiting for the workflow to respond
        response = None
        while response is None and time.time() < deadline:
            try:
                response = _mk_request(url=f"{workflow_url.rstrip('/')}/generate",
                                       code=code,
                                       timeout=timeout,
                                       language=language)
            except Exception:
                time.sleep(0.1)

        assert response is not None, f"deadline exceeded waiting for workflow response: {proc.stdout.read()}"
    finally:
        # Teardown
        i = 0
        while proc.poll() is None and i < 5:
            if i == 0:
                proc.terminate()
            else:
                proc.kill()
            time.sleep(0.1)
            i += 1

        assert proc.poll() is not None, "NAT server process failed to terminate"

    return response.json()


def _test_code_execution(code_block_key: str, sandbox_type: str, config_path: Path, sandbox_config: dict[str, Any]):
    """Test simple print statement execution."""

    code_block = CODE_BLOCKS[code_block_key]
    code = code_block["code"]
    expected_output = code_block["expected_output"]

    code = textwrap.dedent(code).strip()

    if sandbox_type == "local":
        result = run_sandbox_code(sandbox_config, code)
        result_value = result
    else:

        result = run_workflow_code(config_path=config_path, code=code)
        result_value = result["value"]

    assert "process_status" in result_value, f"Sandbox execution failed: {result}"
    assert result_value["process_status"] == "completed", f"Sandbox execution did not complete: {result}"
    assert expected_output in result_value["stdout"], f"Expected output not found in stdout: {result}"
    assert result_value["stderr"] == ""


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("code_block_key",
                         [
                             "hello_world",
                             "simple_addition",
                             "numpy_mean",
                             "pandas_operations",
                             "plotly_import",
                             "file_operations",
                             "persistence_creation",
                             "persistence_readback",
                             "json_persistence"
                         ])
@pytest.mark.parametrize("sandbox_type", ["local", "local_workflow"])
def test_local_code_execution(code_block_key: str,
                              sandbox_type: str,
                              local_sandbox_workflow: Path,
                              sandbox_config: dict[str, Any]):

    _test_code_execution(code_block_key, sandbox_type, local_sandbox_workflow, sandbox_config)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("code_block_key",
                         [
                             "hello_world",
                             "simple_addition",
                             "numpy_mean",
                             "pandas_operations",
                             "file_operations",
                             "persistence_creation",
                             "json_persistence"
                         ])
def test_piston_code_execution(code_block_key: str, piston_sandbox_workflow: Path, sandbox_config: dict[str, Any]):
    _test_code_execution(code_block_key, "piston_workflow", piston_sandbox_workflow, sandbox_config)


@pytest.mark.integration
def test_syntax_error_handling(sandbox_config: dict[str, Any]):
    """Test handling of syntax errors."""
    code = """
print('Hello World'
# Missing closing parenthesis
"""
    result = run_sandbox_code(sandbox_config, code)

    assert result["process_status"] == "error"
    assert "SyntaxError" in result["stderr"] or "SyntaxError" in result["stdout"]


@pytest.mark.integration
def test_runtime_error_handling(sandbox_config: dict[str, Any]):
    """Test handling of runtime errors."""
    code = """
x = 1 / 0
print('This should not print')
"""
    result = run_sandbox_code(sandbox_config, code)

    assert result["process_status"] == "error"
    assert "ZeroDivisionError" in result["stderr"] or "ZeroDivisionError" in result["stdout"]


@pytest.mark.integration
def test_import_error_handling(sandbox_config: dict[str, Any]):
    """Test handling of import errors."""
    code = """
import nonexistent_module
print('This should not print')
"""
    result = run_sandbox_code(sandbox_config, code)

    assert result["process_status"] == "error"
    assert "ModuleNotFoundError" in result["stderr"] or "ImportError" in result["stderr"]


@pytest.mark.integration
def test_mixed_output(sandbox_config: dict[str, Any]):
    """Test code that produces both stdout and stderr output."""
    code = """
import sys
print('This goes to stdout')
print('This goes to stderr', file=sys.stderr)
print('Back to stdout')
"""
    result = run_sandbox_code(sandbox_config, code)

    assert result["process_status"] == "completed"
    assert "This goes to stdout" in result["stdout"]
    assert "Back to stdout" in result["stdout"]
    assert "This goes to stderr" in result["stderr"]


@pytest.mark.integration
def test_long_running_code(sandbox_config: dict[str, Any]):
    """Test code that takes some time to execute but completes within timeout."""
    code = """
import time
for i in range(3):
    print(f'Iteration {i}')
    time.sleep(0.5)
print('Completed')
"""
    result = run_sandbox_code(sandbox_config, code)

    assert result["process_status"] == "completed"
    assert "Iteration 0" in result["stdout"]
    assert "Iteration 1" in result["stdout"]
    assert "Iteration 2" in result["stdout"]
    assert "Completed" in result["stdout"]
    assert result["stderr"] == ""


@pytest.mark.integration
def test_missing_generated_code_field(sandbox_config: dict[str, Any]):
    """Test request missing the generated_code field."""
    payload = {"timeout": 10, "language": "python"}

    response = requests.post(sandbox_config["execute_url"], json=payload, timeout=sandbox_config["timeout"] + 5)

    # Should return an error status code or error in response
    assert response.status_code != 200 or "error" in response.json()


@pytest.mark.integration
def test_missing_timeout_field(sandbox_config: dict[str, Any]):
    """Test request missing the timeout field."""
    payload = {"generated_code": "print('test')", "language": "python"}

    response = requests.post(sandbox_config["execute_url"], json=payload, timeout=sandbox_config["timeout"] + 5)

    # Should return error for missing timeout field
    result = response.json()
    assert response.status_code == 400 and result["process_status"] == "error"


@pytest.mark.integration
def test_invalid_json(sandbox_config: dict[str, Any]):
    """Test request with invalid JSON."""
    invalid_json = '{"generated_code": "print("test")", "timeout": 10}'

    response = requests.post(sandbox_config["execute_url"],
                             data=invalid_json,
                             headers={"Content-Type": "application/json"},
                             timeout=sandbox_config["timeout"] + 5)

    # Should return error for invalid JSON
    assert response.status_code != 200


@pytest.mark.integration
def test_non_json_request(sandbox_config: dict[str, Any]):
    """Test request with non-JSON content."""
    response = requests.post(sandbox_config["execute_url"],
                             data="This is not JSON",
                             headers={"Content-Type": "text/plain"},
                             timeout=sandbox_config["timeout"] + 5)

    # Should return error for non-JSON content
    assert response.status_code != 200


@pytest.mark.integration
def test_timeout_too_low(sandbox_config: dict[str, Any]):
    """Test request with timeout too low."""
    code = """
import time
time.sleep(2.0)
"""
    payload = {"generated_code": code, "timeout": 1, "language": "python"}
    response = requests.post(sandbox_config["execute_url"], json=payload, timeout=sandbox_config["timeout"] + 5)
    assert response.json()["process_status"] == "timeout"
    assert response.status_code == 200
