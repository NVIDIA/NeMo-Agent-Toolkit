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

## What You'll Learn

- **Accuracy Evaluation**: Measure and validate agent responses using the Tunable RAG Evaluator
- **Performance Analysis**: Understand agent behavior through systematic evaluation
- **Dataset Management**: Work with evaluation datasets for consistent testing
- **Results Interpretation**: Analyze evaluation metrics to improve agent performance

## Prerequisites

This example builds upon the [basic Simple Calculator](../../../basic/functions/simple_calculator/). Install it first:

```bash
uv pip install -e examples/basic/functions/simple_calculator
```

## Installation

Install this evaluation example:

```bash
uv pip install -e examples/intermediate/evaluation_and_profiling/simple_calculator_eval
```

## Usage

### Running Evaluation

Evaluate the Simple Calculator agent's accuracy against a test dataset:

```bash
aiq eval --config_file examples/intermediate/evaluation_and_profiling/simple_calculator_eval/configs/config-tunable-rag-eval.yml
```

This command:
- Uses the test dataset from `examples/basic/functions/simple_calculator/data/simple_calculator.json`
- Applies the Tunable RAG Evaluator to measure response accuracy
- Saves detailed results to `.tmp/aiq/examples/basic/functions/simple_calculator/tuneable_eval_output.json`

### Understanding Results

The evaluation generates comprehensive metrics including:

- **Accuracy Scores**: Quantitative measures of response correctness
- **Question-by-Question Analysis**: Detailed breakdown of individual responses
- **Performance Metrics**: Overall quality assessments
- **Error Analysis**: Identification of common failure patterns
