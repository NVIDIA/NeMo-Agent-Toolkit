<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Code Execution

NVIDIA NeMo Agent Toolkit supports Python code execution in a remote sandbox environment through the
`code_execution` function. This function sends Python code to a remote code execution server and returns the result,
status, and any errors.

## Usage

Code execution requires a running [Piston server](https://github.com/engineer-man/piston). Follow the Piston setup
instructions or connect to an existing server, then configure the function with the server's API URL.

The config object for the `code_execution` function is shown below:
```python
class CodeExecutionToolConfig(FunctionBaseConfig, name="code_execution"):
    """
    Tool for executing python code in a remotely hosted sandbox environment.
    """
    uri: HttpUrl = Field(default="http://127.0.0.1:6000", description="URI for the code execution sandbox server")
    sandbox_type: str = Field(default="piston", description="The type of code execution sandbox")
    timeout: float = Field(default=10.0, description="Number of seconds to wait for a code execution request")
    max_output_characters: int = Field(default=1000, description="Maximum number of characters that can be returned")
```
By default, the function uses the Piston client, waits up to 10 seconds, and returns at most 1000 characters. Configure
`uri` for the Piston server before running the workflow:
```yaml
functions:
    code_execution_tool:
      _type: code_execution
      uri: "http://my-piston-server/api/v2/"
```

Below is an example config that connects to a Piston server with a timeout of 30s and a maximum of 3000 characters returned:
```yaml
functions:
    code_execution_tool:
      _type: code_execution
      uri: "http://my-piston-server/api/v2/"
      timeout: 30
      max_output_characters: 3000
```

This remote code execution servers return JSON object containing the execution status, `stdout`, and `stderr`. For example:

```json
{
    "process_status": "completed",
    "stdout": "Hello World\n\n",
    "stderr": ""
}
```
If code execution results in an error, this will show up in `stderr`:
```json
{
    "process_status": "error",
    "stdout": "",
    "stderr": "Traceback (most recent call last):\n  File \"<string>\", line 19, in <module>\n  File \"<string>\", line 1, in <module>\nZeroDivisionError: division by zero\n\n"
}
```
Lastly, it is worth noting that the only thing returned to the function calling the `code_execution` function is (assuming no errors) whatever is printed out to `stdout`. No other artifacts, such as files or in memory objects, are returned from the sandbox, so it is important that the desired result of the code execution is printed out.
