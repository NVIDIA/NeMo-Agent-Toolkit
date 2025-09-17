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

# Sequential Execution Example

This example demonstrates how to use the sequential execution functionality with the NeMo Agent toolkit. Sequential execution allows you to chain multiple functions together where the output of one function becomes the input of the next, creating a data processing pipeline. For this purpose, NeMo Agent toolkit provides a [`sequential_execution`](../../../src/nat/tool/sequential_execution.py) tool.

## Table of Contents

- [Key Features](#key-features)
- [Graph Structure](#graph-structure)
- [Installation and Setup](#installation-and-setup)
  - [Install this Workflow](#install-this-workflow)
  - [Set Up API Keys](#set-up-api-keys)
- [Run the Workflow](#run-the-workflow)
  - [Starting the NeMo Agent Toolkit Server](#starting-the-nemo-agent-toolkit-server)
  - [Making Requests to the NeMo Agent Toolkit Server](#making-requests-to-the-nemo-agent-toolkit-server)

## Key Features

- **Sequential Function Chaining:** Demonstrates how to chain multiple functions together where the output of one becomes the input of the next.
- **Data Processing Pipeline:** Shows a complete text processing pipeline with three stages: text processing, data analysis, and report generation.
- **Type Compatibility Checking:** Optionally validates that the output type of one function is compatible with the input type of the next function in the chain.
- **Error Handling:** Demonstrates proper error handling throughout the sequential execution process.

## Pipeline Structure

The Sequential Execution example demonstrates a linear data processing pipeline where each function processes the output from the previous function:

1. **Text Processor** - Cleans raw text input and extracts basic statistics
2. **Data Analyzer** - Analyzes the processed text and generates insights about complexity and content
3. **Report Generator** - Creates a formatted report from the analysis data

```
Raw Text Input → Text Processor → Data Analyzer → Report Generator → Final Report
```

## Configuration

The Sequential Execution example is configured through the `config.yml` file. The configuration defines individual functions and then chains them together using the `sequential_execution` tool.

### Function Configuration

Each function in the pipeline is configured individually:

- **`text_processor`**: Processes raw text input and returns structured JSON data
- **`data_analyzer`**: Analyzes processed text data and generates insights
- **`report_generator`**: Creates a formatted report from analysis data

### Sequential Execution Configuration

- **`_type`**: Set to `sequential_execution` to use the sequential execution tool
- **`sequential_tool_list`**: List of functions to execute in order (e.g., `[text_processor, data_analyzer, report_generator]`)
- **`check_type_compatibility`**: Whether to validate type compatibility between adjacent functions (default: false)
- **`tool_execution_config`**: Optional configuration for individual tools in the pipeline (e.g., streaming options)

### Workflow Configuration

- **`_type`**: Set to `function` to execute a single function
- **`function_name`**: Name of the sequential execution function to run

### Example Configuration

**Basic Configuration:**
```yaml
functions:
  text_processor:
    _type: text_processor
  data_analyzer:
    _type: data_analyzer
  report_generator:
    _type: report_generator
  text_processing_pipeline:
    _type: sequential_execution
    sequential_tool_list: [text_processor, data_analyzer, report_generator]
    check_type_compatibility: false

workflow:
  _type: function
  function_name: text_processing_pipeline
```

**Configuration with Type Checking:**
```yaml
functions:
  text_processor:
    _type: text_processor
  data_analyzer:
    _type: data_analyzer
  report_generator:
    _type: report_generator
  text_processing_pipeline:
    _type: sequential_execution
    sequential_tool_list: [text_processor, data_analyzer, report_generator]
    check_type_compatibility: true  # Enable type compatibility checking

workflow:
  _type: function
  function_name: text_processing_pipeline
```

The pipeline will automatically execute each function in sequence, passing the output of each function as input to the next function in the chain.

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install NeMo Agent toolkit.

### Install this Workflow

From the root directory of the NeMo Agent toolkit library, run the following commands:

```bash
uv pip install -e examples/agents/sequential_execution
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

## Run the Workflow

This workflow showcases the Sequential Execution functionality by processing raw text through a three-stage pipeline. The example demonstrates how the output of each function becomes the input for the next function in the chain.

Run the following command from the root of the NeMo Agent toolkit repo to execute this workflow with the specified input:

```bash
nat run --config_file=examples/agents/sequential_execution/configs/config.yml --input "The quick brown fox jumps over the lazy dog. This is a simple test sentence to demonstrate text processing capabilities."
```

**Additional Example Commands:**
```bash
# Test with a longer, more complex text
nat run --config_file=examples/agents/sequential_execution/configs/config.yml --input "Natural language processing is a fascinating field that combines computational linguistics with machine learning and artificial intelligence. It enables computers to understand, interpret, and generate human language in a valuable way."

# Test with shorter text
nat run --config_file=examples/agents/sequential_execution/configs/config.yml --input "Hello world! This is a test."

# Test with text containing special characters
nat run --config_file=examples/agents/sequential_execution/configs/config.yml --input "This text has special characters: @#$%^&*()! Let's see how the pipeline handles them."
```

**Expected Workflow Output**
```console
nemo-agent-toolkit % nat run --config_file=examples/agents/sequential_execution/configs/config.yml --input "The quick brown fox jumps over the lazy dog. This is a simple test sentence to demonstrate text processing capabilities."

Configuration Summary:
--------------------
Workflow Type: function
Number of Functions: 4
Number of LLMs: 1
Number of Embedders: 0
Number of Memory: 0
Number of Object Stores: 0
Number of Retrievers: 0
Number of TTC Strategies: 0
Number of Authentication Providers: 0

--------------------------------------------------
Workflow Result:
=== TEXT ANALYSIS REPORT ===

Text Statistics:
  - Word Count: 18
  - Sentence Count: 2
  - Average Words per Sentence: 9.0
  - Text Complexity: Simple

Top Words:
  1. quick
  2. brown
  3. jumps
  4. simple
  5. test

Report generated successfully.
==========================
--------------------------------------------------
```

This demonstrates the Sequential Execution functionality where each function processes the output from the previous function, creating a complete data processing pipeline from raw text input to a formatted report.

### Starting the NeMo Agent Toolkit Server

You can start the NeMo Agent toolkit server using the `nat serve` command with the appropriate configuration file.

**Starting the Sequential Execution Example Workflow**

```bash
nat serve --config_file=examples/agents/sequential_execution/configs/config.yml
```

### Making Requests to the NeMo Agent Toolkit Server

Once the server is running, you can make HTTP requests to interact with the workflow.

#### Non-Streaming Requests

**Non-Streaming Request to the Sequential Execution Example Workflow**

```bash
curl --request POST \
  --url http://localhost:8000/generate \
  --header 'Content-Type: application/json' \
  --data '{"input_message": "The quick brown fox jumps over the lazy dog. This is a simple test sentence to demonstrate text processing capabilities."}'
```

#### Streaming Requests

**Streaming Request to the Sequential Execution Example Workflow**

```bash
curl --request POST \
  --url http://localhost:8000/generate/stream \
  --header 'Content-Type: application/json' \
  --data '{"input_message": "The quick brown fox jumps over the lazy dog. This is a simple test sentence to demonstrate text processing capabilities."}'
```
---
