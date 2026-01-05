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

# Sequential Executor

This example demonstrates how to use the sequential executor functionality with the NVIDIA NeMo Agent toolkit. The sequential executor is a control flow component that chains multiple functions together, where each function's output becomes the input for the next function. This creates a linear tool execution pipeline that executes functions in a predetermined sequence.

The NeMo Agent toolkit provides a [`sequential_executor`](../../../src/nat/control_flow/sequential_executor.py) tool to implement this functionality.

## Table of Contents

- [Key Features](#key-features)
- [Graph Structure](#graph-structure)
- [Configuration](#configuration)
  - [Required Configuration Options](#required-configuration-options)
  - [Optional Configuration Options](#optional-configuration-options)
  - [Exceptions](#exceptions)
  - [Example Configuration](#example-configuration)
- [Installation and Setup](#installation-and-setup)
  - [Install this Workflow](#install-this-workflow)
- [Run the Workflow](#run-the-workflow)
  - [Expected Output](#expected-output)

## Key Features

The sequential executor provides the following capabilities:

- **Sequential function chaining**: Chain multiple functions together where each function's output becomes the input for the next function
- **Type compatibility checking**: Optionally validate that the output type of one function is compatible with the input type of the next function in the chain
- **Error handling**: Handle errors gracefully throughout the sequential execution process

## Graph Structure

The sequential executor uses a linear graph structure where functions execute in a predetermined order. The following diagram illustrates the sequential executor's workflow:

<div align="center">
<img src="../../../docs/source/_static/sequential_executor.png" alt="Sequential Executor Graph Structure" width="750" style="max-width: 100%; height: auto;">
</div>

During execution, each function receives the output from the previous function as its input. The sequential executor supports type compatibility checking between adjacent functions, which you can configure as described in the [Configuration](#configuration) section.

**Note**: The sequential executor does not use agents or LLMs during execution.

## Configuration

Configure the sequential executor through the `config.yml` file. The configuration defines individual functions and chains them together using the `sequential_executor` tool.

### Required Configuration Options

The following options are required for the sequential executor:

- **`_type`**: Set to `sequential_executor` to use the sequential executor tool
- **`tool_list`**: List of functions to execute in order (such as `[text_processor, data_analyzer, report_generator]`)
- **`raise_type_incompatibility`**: Whether to raise an exception if the type compatibility check fails (default: `false`). The type compatibility check runs before executing the tool list, based on the type annotations of the functions. When set to `true`, any incompatibility immediately raises an exception. When set to `false`, incompatibilities generate warning messages and the sequential executor continues execution. Set this to `false` when functions in the tool list include custom type converters, as the type compatibility check may fail even though the sequential executor can still execute the tool list.
- **`return_error_on_exception`**: Whether to return an error message instead of raising an exception when a tool fails during execution (default: `false`). When set to `true`, the sequential executor exits early and returns an error message as the workflow output instead of raising the exception. When set to `false`, exceptions are re-raised. Set this to `true` when you want the workflow to gracefully handle uncaught tool failures and immediately return error information to the user.

### Optional Configuration Options

- **`description`**: Description of the workflow (default: "Sequential Executor Workflow")
- **`tool_execution_config`**: Configuration for each tool in the sequential execution tool list. Keys must match the tool names from the `tool_list`
  - **`use_streaming`**: Whether to use streaming output for the tool (default: `false`)

### Exceptions

- **`SequentialExecutorExit`**: Raised by a tool to exit the chain early and return a custom message as the workflow output. Unlike `return_error_on_exception` which handles unexpected errors, this exception is for intentional early termination. Import from `nat.control_flow.sequential_executor`.

### Example Configuration

The following examples show different configuration approaches:

#### Basic Configuration
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

#### Configuration with Tool Execution Settings
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
      use_streaming: false
  raise_type_incompatibility: false
  return_error_on_exception: false
```

## Installation and Setup

Before running this example, follow the instructions in the [Install Guide](../../../docs/source/get-started/installation.md#install-from-source) to create the development environment and install the NeMo Agent toolkit.

### Install this Workflow

From the root directory of the NeMo Agent toolkit repository, run the following command:

```bash
uv pip install -e examples/control_flow/sequential_executor
```
## Run the Workflow

This workflow demonstrates sequential executor functionality by processing raw text through a three-stage pipeline. Each function's output becomes the input for the next function in the chain.

Run the following command from the root of the NeMo Agent toolkit repository to execute this workflow:

```bash
nat run --config_file=examples/control_flow/sequential_executor/configs/config.yml --input "The quick brown fox jumps over the lazy dog. This is a simple test sentence to demonstrate text processing capabilities."
```

### Expected Output

```console
nemo-agent-toolkit % nat run --config_file=examples/control_flow/sequential_executor/configs/config.yml --input "The quick brown fox jumps over the lazy dog. This is a simple test sentence to demonstrate text processing capabilities."
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file / data utilities can be used.
2025-09-17 15:34:57,004 - nat.cli.commands.start - INFO - Starting NAT from config file: 'examples/control_flow/sequential_executor/configs/config.yml'

Configuration Summary:
--------------------
Workflow Type: sequential_executor
Number of Functions: 3
Number of LLMs: 0
Number of Embedders: 0
Number of Memory: 0
Number of Object Stores: 0
Number of Retrievers: 0
Number of TTC Strategies: 0
Number of Authentication Providers: 0

2025-09-17 15:34:57,571 - nat.front_ends.console.console_front_end_plugin - INFO -
--------------------------------------------------
Workflow Result:
['=== TEXT ANALYSIS REPORT ===\n\nText Statistics:\n  - Word Count: 20\n  - Sentence Count: 2\n  - Average Words per Sentence: 10.0\n  - Text Complexity: Moderate\n\nTop Words:\n  1. quick\n  2. brown\n  3. jumps\n  4. over\n  5. lazy\n\nReport generated successfully.\n==========================']
--------------------------------------------------
```

This output demonstrates how the sequential executor processes raw text input through multiple functions, creating a complete data processing pipeline that generates a formatted analysis report.
