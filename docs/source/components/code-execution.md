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
# Code Execution

AgentIQ supports python code execution in a remote sandbox environment through use of the `code_execution` function. This function sends a string of python code to a remote code execution server where code is executed, and the result, status, and any errors are returned

## Usage
Currently AgentIQ supports code execution through the included `local_sandbox` (a locally run code execution docker container) and via a remote [Piston Server](https://github.com/engineer-man/piston). In order to utilize `code_execution` as part of your workflow this server must be running and accepting requests.

To start the `local_sandbox`you must have docker installed. If docker is not installed on your machine, follow the appropriate instructions [here](https://docs.docker.com/get-started/get-docker/) to install docker on your machine. Once docker is installed and running, navigate to the `local_sandbox` directory and run the `start_local_sandbox.sh` script.

```bash
# from the root of the repository
$ cd src/aiq/tool/code_execution/local_sandbox
$ source start_local_sandbox.sh
```
It will take a bit of time for the container to build and initialize, but once you see the following, the server is ready:
```bash
*** uWSGI is running in multiple interpreter mode ***
spawned uWSGI master process (pid: 9)
spawned uWSGI worker 1 (pid: 11, cores: 1)
spawned uWSGI worker 2 (pid: 12, cores: 1)
spawned uWSGI worker 3 (pid: 13, cores: 1)
spawned uWSGI worker 4 (pid: 14, cores: 1)
spawned uWSGI worker 5 (pid: 15, cores: 1)
spawned uWSGI worker 6 (pid: 16, cores: 1)
spawned uWSGI worker 7 (pid: 17, cores: 1)
spawned uWSGI worker 8 (pid: 18, cores: 1)
spawned uWSGI worker 9 (pid: 19, cores: 1)
spawned uWSGI worker 10 (pid: 20, cores: 1)
running "unix_signal:15 gracefully_kill_them_all" (master-start)...
2025-03-14 02:02:11,060 INFO success: quit_on_failure entered RUNNING state, process has stayed up for > than 1 seconds (startsecs)
```

For Piston servers, follow the instructions [here](https://github.com/engineer-man/piston) to set up a Piston server, or connect to an existing Piston server if you have access to one. Once the server is running you can run your workflow.

The config object for the `code_execution` function is shown below:
```python
class CodeExecutionToolConfig(FunctionBaseConfig, name="code_execution"):
    """
    Tool for executing python code in a remotely hosted sandbox environment.
    """
    uri: HttpUrl = Field(default="http://127.0.0.1:6000", description="URI for the code execution sandbox server")
    sandbox_type: str = Field(default="local", description="The type of code execution sandbox")
    timeout: float = Field(default=10.0, description="Number of seconds to wait for a code execution request")
    max_output_characters: int = Field(default=1000, description="Maximum number of characters that can be returned")
```
The defaults for this config are set use the `local_sandbox`server with a default timeout of 10s and a maximum output of 1000 characters. Below is an example of how this would look in the config file:
```yaml
functions:
    code_execution_tool:
      _type: code_execution
```

Below is an example config that connects to a Piston server with a timeout of 30s and a maximum of 3000 characters returned:
```yaml
functions:
    code_execution_tool:
      _type: code_execution
      uri: "http://my-piston-server"
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
