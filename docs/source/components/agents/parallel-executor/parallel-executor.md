<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Configure the NVIDIA NeMo Agent Toolkit Parallel Executor
Configure the NVIDIA NeMo Agent Toolkit parallel executor as a [workflow](../../../build-workflows/about-building-workflows.md) or a [function](../../../build-workflows/functions-and-function-groups/functions.md). The parallel executor fans out a shared input to all configured tools, executes branches concurrently, and then fans in branch outputs as appended text blocks.

## Requirements
The parallel executor requires the `nvidia-nat[langchain]` plugin to be installed, which can be installed with one of the following commands.

If you have performed a source code checkout:

```bash
uv pip install -e '.[langchain]'
```

If you have installed the NVIDIA NeMo Agent Toolkit from a package:

```bash
uv pip install "nvidia-nat[langchain]"
```

## Configuration

The parallel executor can be used as either a workflow or a function.

### Example 1: Parallel Executor as a Workflow to Configure `config.yml`
To use the parallel executor as a workflow, configure the YAML file as follows:

```yaml
functions:
  topic_agent:
    _type: chat_completion
    llm_name: nim_llm
  urgency_agent:
    _type: chat_completion
    llm_name: nim_llm
  risk_agent:
    _type: chat_completion
    llm_name: nim_llm

workflow:
  _type: parallel_executor
  tool_list: [topic_agent, urgency_agent, risk_agent]
  detailed_logs: true
  return_error_on_exception: false
```

### Example 2: Parallel Executor as a Function to Configure `config.yml`
To use the parallel executor as a function, configure the YAML file as follows:

```yaml
functions:
  topic_agent:
    _type: chat_completion
    llm_name: nim_llm
  urgency_agent:
    _type: chat_completion
    llm_name: nim_llm
  risk_agent:
    _type: chat_completion
    llm_name: nim_llm
  parallel_analysis:
    _type: parallel_executor
    tool_list: [topic_agent, urgency_agent, risk_agent]
    detailed_logs: true
    return_error_on_exception: true
  final_synthesis_agent:
    _type: chat_completion
    llm_name: nim_llm

workflow:
  _type: sequential_executor
  tool_list: [parallel_analysis, final_synthesis_agent]
  raise_type_incompatibility: false
```

### Configurable Options

* `description`: Defaults to "Parallel Executor Workflow". When the parallel executor is configured as a function, this config option allows control of the tool description.
* `tool_list`: **Required**. A list of functions ([tools](../../../build-workflows/functions-and-function-groups/functions.md#agents-and-tools)) to execute in parallel.
* `detailed_logs`: Defaults to `False`. Enables detailed logs for fan-out start, per-branch start and completion, and fan-in summary.
* `return_error_on_exception`: Defaults to `False`. If `True`, branch exceptions are captured and appended as error text blocks. If `False`, the first branch exception is raised.

## Output

The parallel executor returns text where each branch output is appended in order as a separate block.

When `return_error_on_exception` is `True`, failed branches are appended as `ERROR:` blocks.

```text
topic_agent:
{"topic":"product"}

urgency_agent:
{"urgency":"medium"}

risk_agent:
ERROR: RuntimeError: branch failed
```

## Use Cases

The parallel executor is well-suited for:

* Running independent branch analyses in parallel and appending outputs into a single text payload.
* Reducing latency for workflows with independent tool calls.
* Fan-out and fan-in orchestration patterns where each branch can operate on the same input.

## Limitations

The following are the limitations of parallel executors:

* **Shared Input Model**: Every branch receives the same input payload.
* **No Inter-branch Communication**: Branches execute independently and do not communicate during execution.
* **Appended Output Contract**: Downstream tools receive a text payload containing concatenated branch blocks.
