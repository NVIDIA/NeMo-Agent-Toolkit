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

This example demonstrates **evaluation and profiling capabilities** of the AIQ toolkit using the Simple Calculator workflow. Learn how to systematically evaluate your AI agent's performance and accuracy.

## 🎯 What You'll Learn

- **Accuracy Evaluation**: How to measure and validate agent responses using the Tunable RAG Evaluator
- **Performance Profiling**: Understanding agent behavior through systematic evaluation
- **Dataset Management**: Working with evaluation datasets for consistent testing
- **Metrics Analysis**: Interpreting evaluation results to improve agent performance

## 🔗 Prerequisites

This example builds upon the [basic Simple Calculator](../../../basic/functions/simple_calculator/). Install it first:

```bash
uv pip install -e examples/basic/functions/simple_calculator
```

## 📦 Installation

```bash
uv pip install -e examples/intermediate/evaluation_and_profiling/simple_calculator_eval
```

## 🚀 Usage

### Accuracy Evaluation

Run evaluation against a sample dataset to measure response accuracy:

```bash
aiq eval --config_file examples/intermediate/evaluation_and_profiling/simple_calculator_eval/configs/config-tunable-rag-eval.yml
```

The evaluation:
- Uses the dataset in `examples/basic/functions/simple_calculator/data/simple_calculator.json`
- Applies the Tunable RAG Evaluator to measure accuracy
- Outputs detailed results to `.tmp/eval/simple_calculator/tuneable_eval_output.json`

### Understanding Results

The evaluation results include:
- **Accuracy Scores**: Quantitative measures of response correctness
- **Detailed Breakdowns**: Per-question analysis
- **Performance Metrics**: Response quality assessments

## 🔍 Key Features Demonstrated

- **Systematic Evaluation**: Repeatable testing methodology
- **Quality Metrics**: Quantitative assessment of agent performance
- **Dataset Integration**: Structured evaluation data management
- **Results Analysis**: Actionable insights from evaluation runs

## 📊 Configuration

The `config-tunable-rag-eval.yml` demonstrates:
- Evaluation pipeline setup
- Dataset configuration
- Evaluator parameters
- Output format specifications

This focused example showcases how AIQ toolkit enables comprehensive evaluation workflows essential for production AI systems.
