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

# Mixture of Control Flows Example

**Complexity:** 🟡 Intermediate

This example demonstrates how to compose [router agent](../router_agent/README.md) and [sequential executor](../sequential_executor/README.md) control flow patterns in the NeMo Agent Toolkit. 

## Table of Contents

- [Graph Structure](#graph-structure)
- [Configuration](#configuration)
  - [Example Configuration](#example-configuration)
- [Installation and Setup](#installation-and-setup)
  - [Install this Workflow](#install-this-workflow)
  - [Set Up API Keys](#set-up-api-keys)
- [Run the Workflows](#run-the-workflows)
  - [Router Agent to Sequential Executor](#router-agent-to-sequential-executor)
  - [Router Agent to Sequential Executor with Router Agent](#router-agent-to-sequential-executor-with-router-agent)
  - [Router Agent to Router Agent](#router-agent-to-router-agent)

## Graph Structure

The following diagram illustrates an example workflow demonstrating three distinct patterns: routing to a sequential executor, routing to a sequential executor with an embedded router agent, and routing to a nested router agent for specialized tasks:

<div align="center">
<img src="../../../docs/source/_static/mixture_of_control_flows.png" alt="Mixture of Control Flows Graph Structure" width="750" style="max-width: 100%; height: auto;">
</div>

## Configuration

The mixture of control flows is configured through the `config.yml` file. This example demonstrates how to combine multiple control flow components in a single workflow by reusing existing functions from other examples.

### Example Configuration

```yaml
llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    temperature: 0.0
    max_tokens: 4096

functions:
  mock_input_validator:
    _type: mock_input_validator
  mock_uppercase_converter:
    _type: mock_uppercase_converter
  mock_lowercase_converter:
    _type: mock_lowercase_converter
  text_processor:
    _type: text_processor
  data_analyzer:
    _type: data_analyzer
  report_generator:
    _type: report_generator
  mock_result_formatter:
    _type: mock_result_formatter
  fruit_advisor:
    _type: mock_fruit_advisor
  city_advisor:
    _type: mock_city_advisor
  
  # Router Agent -> Sequential Executor
  text_analysis_pipeline:
    _type: sequential_executor
    tool_list: [text_processor, data_analyzer, report_generator]
    raise_type_incompatibility: false
    description: "Processes text, analyzes it, and generates a report"
  
  # Router Agent -> Sequential Executor -> Router Agent
  input_formatter:
    _type: router_agent
    branches: [mock_uppercase_converter, mock_lowercase_converter]
    llm_name: nim_llm
  
  text_formatting_pipeline:
    _type: sequential_executor
    tool_list: [mock_input_validator, input_formatter, mock_result_formatter]
    raise_type_incompatibility: false
    description: "Formats text by converting to uppercase or lowercase"
  
  # Router Agent -> Router Agent
  general_advisor:
    _type: router_agent
    branches: [fruit_advisor, city_advisor]
    llm_name: nim_llm
    description: "Provides advice about fruits or cities"

workflow:
  _type: router_agent
  branches: [text_analysis_pipeline, text_formatting_pipeline, general_advisor]
  llm_name: nim_llm
  detailed_logs: true
```

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/get-started/installation.md#install-from-source) to create the development environment and install NeMo Agent Toolkit.

### Install this Workflow

From the root directory of the NeMo Agent Toolkit library, run the following command:

```bash
uv pip install -e examples/control_flow/mixture_of_control_flows
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../../docs/source/get-started/quick-start.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

## Run the Workflows

This example demonstrates the power of combining different control flow patterns in the NeMo Agent Toolkit. The workflow routes different types of requests to appropriate handlers, where the main router agent intelligently determines whether to execute a text analysis pipeline, a text formatting pipeline with embedded routing, or route to a nested router agent for specialized recommendations based on the request content.

Run the following commands from the root of the NeMo Agent Toolkit repository to execute this workflow with different inputs:

### Router Agent to Sequential Executor

Test the text analysis sequential pipeline:

```bash
nat run --config_file=examples/control_flow/mixture_of_control_flows/configs/config.yml --input "Process this text: The NeMo Agent Toolkit provides powerful control flow capabilities for building sophisticated AI workflows"
```

**Expected Workflow Output:**

```console
Workflow Result:
['=== TEXT ANALYSIS REPORT ===

Text Statistics:
  - Word Count: 17
  - Sentence Count: 0
  - Average Words per Sentence: 0
  - Text Complexity: Simple

Top Words:
  1. process
  2. this
  3. text
  4. nemo
  5. agent

Text Preview:
  Process this text The NeMo Agent Toolkit provides powerful control flow capabilities for building so...

Report generated successfully.
==========================']
```

### Router Agent to Sequential Executor with Router Agent

Test the text formatting pipeline — these examples demonstrate flows from a sequential executor to a router agent:

```bash
# Test uppercase conversion
nat run --config_file=examples/control_flow/mixture_of_control_flows/configs/config.yml --input "Convert this text to uppercase"

# Test lowercase conversion
nat run --config_file=examples/control_flow/mixture_of_control_flows/configs/config.yml --input "CONVERT THIS TEXT TO LOWERCASE"
```

**Expected Workflow Output:**

```console
Workflow Result:
['=== PROCESSED RESULT ===
[VALIDATED] CONVERT THIS TEXT TO UPPERCASE
========================']

...

Workflow Result:
['=== PROCESSED RESULT ===
[validated] convert this text to lowercase
========================']
```

### Router Agent to Router Agent

Test the nested router agent for advisory tasks:

```bash
# Test fruit advisor
nat run --config_file=examples/control_flow/mixture_of_control_flows/configs/config.yml --input "What yellow fruit would you recommend?"

# Test city advisor
nat run --config_file=examples/control_flow/mixture_of_control_flows/configs/config.yml --input "What city should I visit in Canada?"
```

**Expected Workflow Output:**
```console
Workflow Result:
['banana']

...

Workflow Result:
['toronto']
```