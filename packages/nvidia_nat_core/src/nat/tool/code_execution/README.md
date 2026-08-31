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

# Code Execution Sandbox

The code execution tool sends Python code to a remote
[Piston](https://github.com/engineer-man/piston) server and returns the execution status, standard output, and
standard error.

## Setup

Follow the Piston deployment instructions or connect to an existing Piston server. The Piston server must include
the Python 3.10.0 runtime expected by the client.

## Using the Code Execution Tool

Configure a workflow with the Piston API base URL:

```yaml
functions:
  code_execution_tool:
    _type: code_execution
    sandbox_type: piston
    uri: http://my-piston-server/api/v2/
    timeout: 30
    max_output_characters: 3000
```

The `sandbox_type` field remains part of the configuration API so additional sandbox implementations can be added
without changing the function configuration shape.

## Response Format

The tool returns a dictionary with `process_status`, `stdout`, and `stderr` fields:

```json
{
  "process_status": "completed",
  "stdout": "Hello, World!\n",
  "stderr": ""
}
```

Only printed output is returned. Files and in-memory objects created by executed code are not returned to the
workflow.

## Security Considerations

Executing untrusted code carries risk. Configure authentication, authorization, network policies, and resource limits
on the remote execution service as appropriate for the deployment.
