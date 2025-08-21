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

This example demonstrates how to evaluate and profile AI agent performance using the NVIDIA NeMo Agent toolkit. You'll learn to systematically measure your agent's accuracy and analyze its behavior using the Simple LangSmith-Documentation Agent workflow.

## Table of Contents

- [Key Features](#key-features)
- [What You'll Learn](#what-youll-learn)
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
  - [Install this Workflow](#install-this-workflow)
  - [Set Up API Keys](#set-up-api-keys)
- [Run the Workflow](#run-the-workflow)
  - [Running Evaluation](#running-evaluation)
  - [Understanding Results](#understanding-results)
  - [Available Configurations](#available-configurations)

## Key Features

- **Web Query Agent Evaluation:** Demonstrates comprehensive evaluation of the `simple_web_query` agent that retrieves and processes LangSmith documentation using `webpage_query` tools and `react_agent` reasoning.
- **Multi-Model Performance Testing:** Shows systematic comparison across different LLM providers including OpenAI models, Llama 3.1, and Llama 3.3 to identify optimal configurations for documentation retrieval tasks.
- **Evaluation Framework Integration:** Uses the NeMo Agent toolkit `nat eval` command with various evaluation configurations to measure response quality, accuracy scores, and documentation retrieval effectiveness.
- **Question-by-Question Analysis:** Provides detailed breakdown of individual agent responses with comprehensive metrics for identifying failure patterns in LangSmith documentation queries.
- **Dataset Management Workflow:** Demonstrates working with evaluation datasets for consistent testing and performance tracking over time, including evaluation-only modes and result upload capabilities.

## What You'll Learn

- **Accuracy Evaluation**: Measure and validate agent responses using various evaluation methods
- **Performance Analysis**: Understand agent behavior through systematic evaluation
- **Multi-Model Testing**: Compare performance across different LLM providers (OpenAI, Llama 3.1, Llama 3.3)
- **Dataset Management**: Work with evaluation datasets for consistent testing
- **Results Interpretation**: Analyze evaluation metrics to improve agent performance

## Prerequisites

1. **Agent toolkit**: Ensure you have the Agent toolkit installed. If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install NeMo Agent Toolkit.
2. **Base workflow**: This example builds upon the Getting Started [Simple Web Query](../../getting_started/simple_web_query/) example. Make sure you are familiar with the example before proceeding.

## Installation and Setup

### Install this Workflow

Install this evaluation example:

```bash
uv pip install -e examples/evaluation_and_profiling/simple_web_query_eval
```

### Set Up API Keys

Follow the [Obtaining API Keys](../../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to set up your API keys:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>  # For OpenAI evaluations
```

## Run the Workflow

### Running Evaluation

Evaluate the Simple LangSmith-Documentation agent's accuracy using different configurations:

#### Basic Evaluation

The configuration files specified below contain configurations for the NeMo Agent Toolkit `evaluation` and `profiler` capabilities. Additional documentation for evaluation configuration can be found in the [evaluation guide](../../../docs/source/workflows/evaluate.md). Furthermore, similar documentation for profiling configuration can be found in the [profiling guide](../../../docs/source/workflows/profiler.md).

```bash
nat eval --config_file examples/evaluation_and_profiling/simple_web_query_eval/configs/eval_config.yml
```

#### OpenAI Model Evaluation
```bash
nat eval --config_file examples/evaluation_and_profiling/simple_web_query_eval/configs/eval_config_openai.yml
```

#### Llama 3.1 Model Evaluation
```bash
nat eval --config_file examples/evaluation_and_profiling/simple_web_query_eval/configs/eval_config_llama31.yml
```

#### Llama 3.3 Model Evaluation
```bash
nat eval --config_file examples/evaluation_and_profiling/simple_web_query_eval/configs/eval_config_llama33.yml
```

#### Evaluation-Only Mode
```bash
nat eval --skip_workflow --config_file examples/evaluation_and_profiling/simple_web_query_eval/configs/eval_only_config.yml --dataset ./.tmp/nat/examples/evaluation_and_profiling/simple_web_query_eval/eval/workflow_output.json
```


#### Evaluation with Upload

#### Setting up S3 Bucket for Upload

To enable the `eval_upload.yml` workflow, you must configure an S3-compatible bucket for both dataset input and result output. You can use AWS S3, MinIO, or another S3-compatible service.

**Using AWS S3**
1. Create a bucket by following instructions [here](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html).
2. Configure your AWS credentials:
   ```bash
   export AWS_ACCESS_KEY_ID=<YOUR_ACCESS_KEY_ID>
   export AWS_SECRET_ACCESS_KEY=<YOUR_SECRET_ACCESS_KEY>
   export AWS_DEFAULT_REGION=<your-region>
   ```
3. In `eval_upload.yml`, update the `bucket`, `endpoint_url` (if using a custom endpoint), and credentials under both `eval.general.output.s3` and `eval.general.dataset.s3`.

**Using MinIO**
1. Start a local MinIO server or cloud instance.
2. Create a bucket via the MinIO console or client by following instructions [here](https://min.io/docs/minio/linux/reference/minio-mc/mc-mb.html).
3. Set environment variables:
   ```bash
   export AWS_ACCESS_KEY_ID=<MINIO_ACCESS_KEY>
   export AWS_SECRET_ACCESS_KEY=<MINIO_SECRET_KEY>
   export S3_ENDPOINT=http://<minio-host>:<port>
   ```
4. In `eval_upload.yml`, configure `endpoint_url` to point to `$S3_ENDPOINT`, and set the `bucket`, `access_key`, and `secret_key` accordingly.

For more information about using remote files for evaluation, refer to the [evaluation guide](../../../docs/source/reference/evaluate.md).

#### Upload dataset to the S3 bucket
To use the sample config file `eval_upload.yml`, you need to upload the following dataset files to the S3 bucket at path `input/`:
- `examples/evaluation_and_profiling/simple_web_query_eval/data/langsmith.json`

#### Running Evaluation
```bash
nat eval --config_file examples/evaluation_and_profiling/simple_web_query_eval/configs/eval_upload.yml
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
