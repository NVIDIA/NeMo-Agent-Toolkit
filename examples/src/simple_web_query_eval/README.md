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

# Simple LangSmith-Documentation Agent - Evaluation and Profiling

This example demonstrates how to evaluate and profile AI agent performance using the NVIDIA AIQ toolkit. You'll learn to systematically measure your agent's accuracy and analyze its behavior using the Simple LangSmith-Documentation Agent workflow.

## What You'll Learn

- **Accuracy Evaluation**: Measure and validate agent responses using various evaluation methods
- **Performance Analysis**: Understand agent behavior through systematic evaluation
- **Multi-Model Testing**: Compare performance across different LLM providers (OpenAI, Llama 3.1, Llama 3.3)
- **Dataset Management**: Work with evaluation datasets for consistent testing
- **Results Interpretation**: Analyze evaluation metrics to improve agent performance

## Prerequisites

This example builds upon the [basic Simple LangSmith-Documentation Agent](../../../basic/functions/simple/). Install it first:

```bash
uv pip install -e examples/basic/functions/simple
```

## Installation

Install this evaluation example:

```bash
uv pip install -e examples/intermediate/evaluation_and_profiling/simple_eval
```

## Usage

### Set Up API Keys

Follow the [Obtaining API Keys](../../../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to set up your API keys:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>  # For OpenAI evaluations
```

### Running Evaluation

Evaluate the Simple LangSmith-Documentation agent's accuracy using different configurations:

#### Basic Evaluation
```bash
aiq eval --config_file examples/intermediate/evaluation_and_profiling/simple_eval/configs/eval_config.yml
```

#### OpenAI Model Evaluation
```bash
aiq eval --config_file examples/intermediate/evaluation_and_profiling/simple_eval/configs/eval_config_openai.yml
```

#### Llama 3.1 Model Evaluation
```bash
aiq eval --config_file examples/intermediate/evaluation_and_profiling/simple_eval/configs/eval_config_llama31.yml
```

#### Llama 3.3 Model Evaluation
```bash
aiq eval --config_file examples/intermediate/evaluation_and_profiling/simple_eval/configs/eval_config_llama33.yml
```

#### Evaluation-Only Mode
```bash
aiq eval --config_file examples/intermediate/evaluation_and_profiling/simple_eval/configs/eval_only_config.yml
```

#### Evaluation with Upload
```bash
aiq eval --config_file examples/intermediate/evaluation_and_profiling/simple_eval/configs/eval_upload.yml
```

### Understanding Results

The evaluation generates comprehensive metrics including:

- **Response Quality**: Measures how well the agent answers LangSmith-related questions
- **Accuracy Scores**: Quantitative measures of response correctness
- **Question-by-Question Analysis**: Detailed breakdown of individual responses
- **Performance Metrics**: Overall quality assessments across different models
- **Error Analysis**: Identification of common failure patterns in documentation retrieval and response generation

### Available Configurations

| Configuration | Description |
|--------------|-------------|
| `eval_config.yml` | Standard evaluation with default settings |
| `eval_config_openai.yml` | Evaluation using OpenAI models |
| `eval_config_llama31.yml` | Evaluation using Llama 3.1 model |
| `eval_config_llama33.yml` | Evaluation using Llama 3.3 model |
| `eval_only_config.yml` | Evaluation-only mode without running the workflow |
| `eval_upload.yml` | Evaluation with automatic result upload |

This helps you systematically improve your LangSmith documentation agent by understanding its strengths and areas for improvement across different model configurations.
