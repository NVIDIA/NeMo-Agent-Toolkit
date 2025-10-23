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

The configuration files specified below contain configurations for the NeMo Agent Toolkit `evaluation` and `profiler` capabilities. For detailed information about evaluation configuration and output files, refer to the [evaluation guide](../../../docs/source/workflows/evaluate.md). For profiling configuration and metrics, see the [profiling guide](../../../docs/source/workflows/profiler.md).

```bash
nat eval --config_file examples/evaluation_and_profiling/simple_web_query_eval/configs/eval_config.yml
```

> [!NOTE]
> If you encounter rate limiting (`[429] Too Many Requests`) during evaluation, try setting the `eval.general.max_concurrency` value either in the YAML directly or via the command line with: `--override eval.general.max_concurrency 1`.

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

##### Setting up S3 Bucket for Upload

To enable the `eval_upload.yml` workflow, you must configure an S3-compatible bucket for both dataset input and result output. You can use AWS S3, MinIO, or another S3-compatible service.

We recommend installing the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) to create and manage your S3 buckets, regardless of the S3-compatible service you use.

**Set the bucket name:**

```bash
export S3_BUCKET_NAME=nat-simple-bucket
```

**Using AWS S3**
1. Configure your AWS credentials:
   ```bash
   export AWS_ACCESS_KEY_ID=<YOUR_ACCESS_KEY_ID>
   export AWS_SECRET_ACCESS_KEY=<YOUR_SECRET_ACCESS_KEY>
   export AWS_DEFAULT_REGION=<your-region>
   ```

**Using MinIO**
1. Start a local MinIO server or cloud instance. To start a local MinIO server, consult the [MinIO section](../../deploy/README.md#running-services) of the deployment guide.
2. Set environment variables:
   ```bash
   export AWS_ACCESS_KEY_ID=minioadmin
   export AWS_SECRET_ACCESS_KEY=minioadmin
   export S3_ENDPOINT_URL=http://localhost:9000
   ```

**Creating the S3 bucket:**
```bash
aws s3 mb \
  s3://${S3_BUCKET_NAME}
  ${S3_ENDPOINT_URL:+--endpoint-url=${S3_ENDPOINT_URL}}
```

For more information about using remote files for evaluation, refer to the [evaluation guide](../../../docs/source/reference/evaluate.md).

##### Upload dataset to the S3 bucket
To use the sample config file `eval_upload.yml`, you need to upload the following dataset files to the S3 bucket at path `input/`:
- `examples/evaluation_and_profiling/simple_web_query_eval/data/langsmith.json`

For example, if you have the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) installed, you can use the following command to upload the dataset files to the S3 bucket:
```bash
aws s3 cp \
  examples/evaluation_and_profiling/simple_web_query_eval/data/langsmith.json \
  s3://${S3_BUCKET_NAME}/input/langsmith.json \
  ${S3_ENDPOINT_URL:+--endpoint-url=${S3_ENDPOINT_URL}}
```

##### Running Evaluation
```bash
nat eval --config_file examples/evaluation_and_profiling/simple_web_query_eval/configs/eval_upload.yml
```

### Understanding Results

After running evaluation, you'll find output files in `./.tmp/nat/examples/evaluation_and_profiling/simple_web_query_eval/eval/` (or your configured output directory). The evaluation generates comprehensive metrics including:

- **Response Quality**: Measures how well the agent answers LangSmith-related questions
- **Accuracy Scores**: Quantitative measures of response correctness
- **Question-by-Question Analysis**: Detailed breakdown of individual responses
- **Performance Metrics**: Overall quality assessments across different models
- **Error Analysis**: Identification of common failure patterns in documentation retrieval and response generation

#### Evaluation Outputs

Running `nat eval` generates several artifacts in the output directory:

**Workflow outputs (always available)**
- `workflow_output.json`: Per-sample execution results including question, expected answer, generated answer, and intermediate steps

**Evaluator outputs (when configured)**
- `trajectory_accuracy_output.json`: Trajectory evaluator scores and reasoning
- `accuracy_output.json`: Ragas AnswerAccuracy scores
- `groundedness_output.json`: Ragas ResponseGroundedness scores
- `relevance_output.json`: Ragas ContextRelevance scores

**Profiler outputs (when enabled)**
- `standardized_data_all.csv`: Per-request profiler metrics
- `workflow_profiling_metrics.json`: Aggregated profiler statistics
- `workflow_profiling_report.txt`: Human-readable profiler summary
- `gantt_chart.png`: Timeline visualization
- `all_requests_profiler_traces.json`: Full trace events
- `inference_optimization.json`: Inference optimization signals (when `compute_llm_metrics` is enabled)

For detailed descriptions of each output file, refer to the [Evaluation outputs section](../../../docs/source/workflows/evaluate.md#evaluation-outputs-what-you-will-get) in the evaluation guide.

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
