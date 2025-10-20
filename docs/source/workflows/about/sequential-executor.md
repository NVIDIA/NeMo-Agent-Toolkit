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

# Sequential Executor

A sequential executor is a control flow component that chains multiple functions together, where each function's output becomes the input for the next function. This creates a linear tool execution pipeline that executes functions in a predetermined sequence without requiring LLMs or agents for orchestration.

## Features
- **Sequential Function Chaining**: Chain multiple functions together where each function's output becomes the input for the next function
- **Type Compatibility Checking**: Optionally validate that the output type of one function is compatible with the input type of the next function in the chain
- **Streaming Support**: Configure individual functions to use streaming or non-streaming execution modes
- **Error Handling**: Handle errors gracefully throughout the sequential execution process
- **No LLM Required**: Execute deterministic workflows without the overhead of LLM calls
- **Configurable Workflows**: Fully configurable via YAML for flexibility and productivity

---

## Requirements
The sequential executor is part of the core NeMo Agent toolkit and does not require additional plugin installations.

If you have performed a source code checkout, install this with the following command:

```bash
uv pip install -e .
```

If you have installed the NeMo Agent toolkit from a package, you can install this with the following command:

```bash
uv pip install "nvidia-nat"
```

## Configuration

The sequential executor may be utilized as a workflow or a function.

### Example `config.yml`
In your YAML file, to use the sequential executor as a workflow:
```yaml
functions:
  text_processor:
    _type: text_processor
  data_analyzer:
    _type: data_analyzer
  report_generator:
    _type: report_generator

workflow:
  _type: sequential_executor
  tool_list: [text_processor, data_analyzer, report_generator]
  raise_type_incompatibility: false
```

In your YAML file, to use the sequential executor as a function:
```yaml
functions:
  text_processor:
    _type: text_processor
  data_analyzer:
    _type: data_analyzer
  report_generator:
    _type: report_generator
  processing_pipeline:
    _type: sequential_executor
    tool_list: [text_processor, data_analyzer, report_generator]
    description: 'A pipeline that processes text through multiple stages'
    raise_type_incompatibility: false
```

### Configuration with Tool Execution Settings
```yaml
functions:
  text_processor:
    _type: text_processor
  data_analyzer:
    _type: data_analyzer
  report_generator:
    _type: report_generator

workflow:
  _type: sequential_executor
  tool_list: [text_processor, data_analyzer, report_generator]
  tool_execution_config:
    text_processor:
      use_streaming: false
    data_analyzer:
      use_streaming: false
    report_generator:
      use_streaming: true
  raise_type_incompatibility: false
```

### Configurable Options

* `tool_list`: **Required**. A list of functions to execute sequentially. Each function's output becomes the input for the next function in the chain.

* `raise_type_incompatibility`: Defaults to `False`. Whether to raise an exception if the type compatibility check fails. The type compatibility check runs before executing the tool list, based on the type annotations of the functions. When set to `True`, any incompatibility immediately raises an exception. When set to `False`, incompatibilities generate warning messages and the sequential executor continues execution. Set this to `False` when functions in the tool list include custom type converters, as the type compatibility check may fail even though the sequential executor can still execute the tool list.

* `tool_execution_config`: Optional configuration for each tool in the sequential execution tool list. Keys must match the tool names from the `tool_list`.
  - `use_streaming`: Defaults to `False`. Whether to use streaming output for the tool.

---

## How the Sequential Executor Works

A **sequential executor** is a deterministic workflow orchestrator that executes functions in a predefined linear order. Unlike agents that use LLMs to make decisions, the sequential executor follows a fixed execution path where each function's output directly becomes the input for the next function.

### **Step-by-Step Breakdown of a Sequential Executor**
The sequential executor follows a fixed execution path where each function's output directly becomes the input for the next function.

<div align="center">
<img src="../../_static/sequential_executor.png" alt="Sequential Executor Graph Structure" width="800" style="max-width: 100%; height: auto;">
</div>

### Type Compatibility Checking

The sequential executor can optionally validate type compatibility between adjacent functions in the chain:

- **Before Execution**: The executor checks if the output type of each function is compatible with the input type of the next function
- **Compatibility Validation**: Uses Python type annotations to determine compatibility
- **Error Handling**: Can either raise exceptions or generate warnings based on configuration
- **Streaming Consideration**: Takes into account whether functions use streaming or single output modes

---

## Use Cases

The sequential executor is best suited for:

* The workflow is deterministic and follows a fixed sequence
* No decision-making is required between steps
* Functions have clear input/output dependencies

---

## Limitations

While sequential executors are efficient and predictable, they have several limitations:

* **No Dynamic Decision Making**

  Sequential executors follow a fixed execution path and cannot make decisions based on intermediate results. All functions in the tool list will always execute in the same order.

* **No Parallel Execution**

  Functions execute sequentially, which means they cannot take advantage of parallel processing opportunities. This can be inefficient for independent operations that could run simultaneously.

In summary, sequential executors are best suited for deterministic workflows with well-defined data flow requirements. For more complex orchestration needs, consider using agents or other workflow types.
