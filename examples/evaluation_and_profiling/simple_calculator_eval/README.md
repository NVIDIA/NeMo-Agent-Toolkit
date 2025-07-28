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

# Simple Calculator - Evaluation and Profiling

This example demonstrates how to evaluate and profile AI agent performance using the NVIDIA NeMo Agent toolkit. You'll learn to systematically measure your agent's accuracy and analyze its behavior using the Simple Calculator workflow.

## Key Features

- **Tunable RAG Evaluator Integration:** Demonstrates the `aiq eval` command with Tunable RAG Evaluator to measure agent response accuracy against ground truth datasets.
- **Performance Analysis Framework:** Shows systematic evaluation of agent behavior, accuracy, and response quality using standardized test datasets.
- **Question-by-Question Analysis:** Provides detailed breakdown of individual responses with comprehensive metrics for identifying failure patterns and areas for improvement.
- **Evaluation Dataset Management:** Demonstrates how to work with structured evaluation datasets (`simple_calculator.json`) for consistent and reproducible testing.
- **Results Interpretation:** Shows how to analyze evaluation metrics and generate comprehensive performance reports for agent optimization.

## What You'll Learn

- **Accuracy Evaluation**: Measure and validate agent responses using the Tunable RAG Evaluator
- **Performance Analysis**: Understand agent behavior through systematic evaluation
- **Dataset Management**: Work with evaluation datasets for consistent testing
- **Results Interpretation**: Analyze evaluation metrics to improve agent performance

## Prerequisites

1. **Agent toolkit**: Ensure you have the Agent toolkit installed. If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install NeMo Agent Toolkit.
2. **Base workflow**: This example builds upon the Getting Started [Simple Calculator](../../getting_started/simple_calculator/) example. Make sure you are familiar with the example before proceeding.

## Installation

Install this evaluation example:

```bash
uv pip install -e examples/evaluation_and_profiling/simple_calculator_eval
```

## Run the Workflow

### Running Evaluation

Evaluate the Simple Calculator agent's accuracy against a test dataset:

```bash
aiq eval --config_file examples/evaluation_and_profiling/simple_calculator_eval/configs/config-tunable-rag-eval.yml
```

The configuration file specified above contains configurations for the NeMo Agent Toolkit `evaluation` and `profiler` capabilities. Additional documentation for evaluation configuration can be found in the [evaluation guide](../../../docs/source/workflows/evaluate.md). Furthermore, similar documentation for profiling configuration can be found in the [profiling guide](../../../docs/source/workflows/profiler.md).

This command:
- Uses the test dataset from `examples/getting_started/simple_web_query/data/simple_calculator.json`
- Applies the Tunable RAG Evaluator to measure response accuracy
- Saves detailed results to `.tmp/aiq/examples/getting_started/simple_web_query/tuneable_eval_output.json`

### Understanding Results

The evaluation generates comprehensive metrics including:

- **Accuracy Scores**: Quantitative measures of response correctness
- **Question-by-Question Analysis**: Detailed breakdown of individual responses
- **Performance Metrics**: Overall quality assessments
- **Error Analysis**: Identification of common failure patterns
