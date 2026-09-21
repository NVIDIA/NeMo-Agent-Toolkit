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
"""Integration tests for Piston-backed code execution."""

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


def _test_code_execution(code_block_key: str, config_path: Path):
    """Test simple print statement execution."""

    code_block = CODE_BLOCKS[code_block_key]
    code = code_block["code"]
    expected_output = code_block["expected_output"]

    code = textwrap.dedent(code).strip()

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
                             "file_operations",
                             "persistence_creation",
                             "json_persistence"
                         ])
def test_piston_code_execution(code_block_key: str, piston_sandbox_workflow: Path):
    _test_code_execution(code_block_key, piston_sandbox_workflow)
