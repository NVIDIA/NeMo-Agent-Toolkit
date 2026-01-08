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

# Configure With the Sequential Executor
A sequential executor is a deterministic [workflow](../../../build-workflows/about-building-workflows.md) orchestrator that executes functions in a predefined linear order. This section explores ways you can configure using the sequential executor.

The sequential executor is part of the core NeMo Agent toolkit and does not require additional plugin installations.

## Configuration

The sequential executor may be used as a workflow or a function.

### Example 1: Sequential Executor as a Workflow to Configure `config.yml`
To use the sequential executor as a workflow, configure the YAML file as follows:
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
  return_error_on_exception: false
```
### Example 2: Sequential Executor as a Function to Configure `config.yml`
To use the sequential executor as a function, configure the YAML file as follows:
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
    return_error_on_exception: false
```

### Example 3: Configure with Tool Execution Settings
Configure the YAML file with tool execution settings as follows:
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
  return_error_on_exception: false
```

### Configurable Options

* `description`: Defaults to "Sequential Executor Workflow". When the sequential executor is configured as a function, this config option allows control of the tool description (for example, when used as a tool within another agent).

* `tool_list`: **Required**. A list of functions ([tools](../../../build-workflows/functions-and-function-groups/functions.md#agents-and-tools)) to execute sequentially. Each function's output becomes the input for the next function in the chain.

* `raise_type_incompatibility`: Defaults to `False`. Whether to raise an exception if the type compatibility check fails. The type compatibility check runs before executing the tool list, based on the type annotations of the functions. When set to `True`, any incompatibility immediately raises an exception. When set to `False`, incompatibilities generate warning messages and the sequential executor continues execution. Set this to `False` when functions in the tool list include custom type converters, as the type compatibility check may fail even though the sequential executor can still execute the tool list.

* `return_error_on_exception`: Defaults to `False`. Whether to return an error message instead of raising an exception when a tool fails during execution. When set to `True`, the sequential executor exits early and returns an error message as the workflow output instead of raising the exception. When set to `False`, exceptions are re-raised. Set this to `True` when you want the workflow to gracefully handle uncaught tool failures and immediately return error information to the user.

* `tool_execution_config`: Optional configuration for each tool in the sequential execution tool list. Keys must match the tool names from the `tool_list`.
  - `use_streaming`: Defaults to `False`. Whether to use streaming output for the tool.

### Exceptions

* **{py:class}`~nat.control_flow.sequential_executor.SequentialExecutorExit`**: Raised by a tool to exit the chain early and return a custom message as the workflow output. Unlike `return_error_on_exception` which handles unexpected errors, this exception is for intentional early termination.

## The Sequential Executor Workflow

The sequential executor follows a fixed execution path where each function's output directly becomes the input for the next function.

<div align="center">
<img src="../../../_static/sequential_executor.png" alt="Sequential Executor Graph Structure" width="800" style="max-width: 100%; height: auto;">
</div>

### Type Compatibility Validation

The sequential executor can optionally use the Python type annotations to validate the compatibility between adjacent functions in the chain:

1. Before execution, the executor checks if the output type of each function is compatible with the input type of the next function.
2. The execution then raises exceptions or generates warnings based on configuration.

:::{note} 
The validation considers whether functions use streaming or single output modes.
:::

## Use Cases

The sequential executor is well-suited for:

* The workflow is deterministic and follows a fixed sequence
* No decision-making is required between steps
* Functions have clear input and output dependencies

---

## Limitations

While sequential executors are efficient and predictable, they have several limitations:

* **No Dynamic Decision-Making** - Sequential executors follow a fixed execution path and cannot make decisions based on intermediate results. All functions in the tool list will always execute in the same order.

* **No Parallel Execution** - Functions execute sequentially, which means they cannot take advantage of parallel processing opportunities. This can be inefficient for independent operations that could run simultaneously.

In summary, sequential executors are best suited for deterministic workflows with well-defined data flow requirements. For more complex orchestration needs, consider using agents or other workflow types.
