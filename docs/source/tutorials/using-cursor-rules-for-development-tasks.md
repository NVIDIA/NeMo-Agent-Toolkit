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

# Using Cursor Rules for Development Tasks

The AIQ toolkit integrates with Cursor to streamline common development tasks through natural language requests. Cursor rules allow you to interact with your AIQ workflows and project files using simple chat commands, which are automatically translated into the appropriate CLI commands and file edits.

This tutorial describes the comprehensive set of tasks you can accomplish using Cursor rules and how to phrase your Cursor Chat Agent requests effectively.

## Overview of Available Cursor Rules

The AIQ Cursor rules are organized into several categories:
- **General CLI Rules**: Overall guidance for AIQ CLI commands
- **Workflow Management**: Creating, installing, and managing workflows
- **Running and Serving**: Executing workflows locally and as services
- **Evaluation**: Testing and benchmarking workflow performance
- **Information Discovery**: Finding components and registry information

## Supported Tasks

### 1. Workflow Management

#### Create a New Workflow
Create workflows with various customization options:

```text
create a workflow named my_rag_workflow
```

```text
create a workflow named my_rag_workflow with description "A custom RAG workflow for document processing"
```

```text
create a workflow named my_rag_workflow in the examples directory with description "Custom workflow for data analysis"
```

```text
create a workflow named my_rag_workflow without installing it immediately
```

**What Cursor will do:**
- Generate a valid `pyproject.toml` file with plugin section
- Create `register.py` file with AIQ toolkit boilerplate code
- Generate configuration files for launching the workflow
- Install the workflow package (unless `--no-install` is specified)

#### Reinstall a Workflow
After modifying workflow code, dependencies, or configuration:

```text
reinstall workflow my_rag_workflow
```

```text
rebuild and reinstall my_rag_workflow after my changes
```

**When to use:**
- After modifying the workflow's Python code
- After updating dependencies in `pyproject.toml`
- After making changes to the workflow's configuration
- After adding new tools or components

#### Delete or Uninstall a Workflow
Remove workflows completely:

```text
delete the workflow named my_rag_workflow
```

```text
uninstall workflow my_rag_workflow and clean up its files
```

### 2. Running Workflows

#### Basic Workflow Execution
Run workflows with various input options:

```text
run workflow examples/simple/configs/config.yml with input "What is LangSmith?"
```

```text
run workflow simple with input "What is LangSmith?"
```

```text
run workflow examples/simple/configs/config.yml with input file inputs/questions.json
```

#### Advanced Running with Overrides
Test different configurations without modifying files:

```text
run workflow examples/simple/configs/config.yml with input "Hello" and override llms.nim_llm.temperature to 0.7
```

```text
run workflow config.yml with input "Test query" and override llms.nim_llm.temperature to 0.7 and retriever.top_k to 10
```

**Use cases:**
- One-off testing and debugging
- Running workflows in development
- Batch processing with input files
- Quick validation of workflow configurations

### 3. Serving Workflows

#### Basic Service Deployment
Deploy workflows as FastAPI endpoints:

```text
serve workflow examples/simple/configs/config.yml
```

```text
serve workflow examples/simple/configs/config.yml on host 0.0.0.0 port 8000
```

#### Production-Ready Serving
Configure for production environments:

```text
serve workflow examples/simple/configs/config.yml with 4 workers and auto-reload enabled
```

```text
serve workflow config.yml on host 0.0.0.0 port 8000 with 4 workers using gunicorn
```

```text
serve workflow config.yml on localhost port 8000 with auto-reload for development
```

#### Serving with Configuration Overrides
Test different parameters in the served endpoint:

```text
serve workflow config.yml on port 8080 with override llms.nim_llm.max_tokens to 2048 and retriever.top_k to 5
```

**What you get:**
- FastAPI endpoint with automatic API documentation at `/docs`
- Swagger UI for testing endpoints
- Production-ready deployment options
- Development features like auto-reload

### 4. Workflow Evaluation

#### Basic Evaluation
Assess workflow accuracy and performance:

```text
evaluate workflow examples/simple/configs/config.yml with dataset path/to/dataset.json
```

```text
evaluate workflow config.yml with dataset test_questions.json and 3 repetitions
```

#### Advanced Evaluation Options
Use comprehensive evaluation features:

```text
evaluate workflow config.yml against endpoint http://localhost:8000/generate with timeout 600 seconds
```

```text
evaluate workflow config.yml skipping completed entries with dataset results.json
```

```text
evaluate workflow config.yml extracting results from $.response.answer with dataset test_set.json
```

```text
evaluate workflow config.yml with 5 repetitions for statistical significance
```

#### Evaluation-Only Mode
When you have pre-generated results:

```text
evaluate pre-generated results in dataset results_with_answers.json using config eval_config.yml
```

**Evaluation capabilities:**
- Multiple evaluation metrics (accuracy, BLEU score, semantic similarity)
- Statistical significance through repetitions
- Endpoint evaluation against running services
- Incremental evaluation of large datasets
- Custom result extraction from complex outputs

### 5. Component and Information Discovery

#### Finding Components
Discover available components for your workflows:

```text
Show all the tools that contain "webpage" in the name
```

```text
Is there a tool that can query a webpage?
```

```text
Show all the LLM providers available
```

```text
List all available retrievers
```

```text
Find components related to embedding
```

```text
Show all evaluation components
```

#### Component Details
Get detailed information about specific components:

```text
Show details for the milvus retriever component
```

```text
What parameters does the nim_llm component accept?
```

```text
List all components in the my_package package
```

#### Registry Information
Check configured registries and channels:

```text
Show all configured registry channels
```

```text
List only REST type registry channels
```

### 6. Configuration Management

#### Validate Configurations
Ensure your configurations are correct:

```text
validate config examples/simple/configs/config.yml
```

```text
check if my workflow configuration file is valid
```

#### Add Tools to Workflows
Enhance workflows with additional tools:

```text
add tool current_datetime to workflow examples/simple/configs/config.yml
```

```text
add webpage_query tool to workflow examples/simple/configs/config.yml with url https://docs.smith.langchain.com/user_guide
```

**What Cursor will do:**
- Update the YAML configuration file
- Add the tool to the `functions` section
- Update the `tool_names` list
- Validate the configuration

## Advanced Usage Patterns

### 1. Complete Development Workflow
```text
create a workflow named rag_system with description "RAG system for document QA"
add tool webpage_query to workflow rag_system/configs/config.yml
validate config rag_system/configs/config.yml
run workflow rag_system with input "What is the latest news?"
serve workflow rag_system/configs/config.yml on localhost port 8000 with auto-reload
evaluate workflow rag_system/configs/config.yml with dataset test_data.json
```

### 2. A/B Testing Configurations
```text
run workflow baseline_config.yml with input "test query" and save results
run workflow improved_config.yml with input "test query" and compare with baseline
evaluate both configurations with dataset comparison_set.json and 3 repetitions
```

### 3. Parameter Tuning
```text
run workflow config.yml with input "test" and override llms.nim_llm.temperature to 0.3
run workflow config.yml with input "test" and override llms.nim_llm.temperature to 0.7
run workflow config.yml with input "test" and override llms.nim_llm.temperature to 0.9
```

### 4. Production Deployment Pipeline
```text
validate config production_config.yml
run workflow production_config.yml with input file test_inputs.json
evaluate workflow production_config.yml with dataset validation_set.json
serve workflow production_config.yml on host 0.0.0.0 port 8000 with 4 workers using gunicorn
```

## Tips for Effective Usage

### Natural Language Patterns
- **Be specific but natural**: "serve workflow on localhost port 8000 with auto-reload" is better than just "serve workflow"
- **Chain actions**: You can combine multiple operations in one request
- **Use descriptive names**: Provide meaningful workflow names and descriptions
- **Specify environments**: Mention development vs. production requirements

### Best Practices
1. **Start with component discovery**: Always explore available components before building workflows
2. **Validate before running**: Use validation commands to catch configuration errors early
3. **Test incrementally**: Start with `run` command before moving to `serve` for endpoints
4. **Use overrides for testing**: Test different parameters without modifying configuration files
5. **Leverage evaluation**: Regularly evaluate workflows with proper datasets
6. **Document your configurations**: Use descriptive names and comments in configuration files

### Common Workflow Patterns
1. **Discovery → Creation → Testing → Deployment**
2. **Create → Develop → Test → Evaluate → Deploy**
3. **Baseline → Parameter Tuning → A/B Testing → Production**

## Dataset Formats

### Evaluation Dataset Format
```json
[
  {
    "question": "What is machine learning?",
    "ground_truth": "Machine learning is a subset of artificial intelligence...",
    "context": "AI fundamentals",
    "difficulty": "intermediate"
  }
]
```

### Input File Format for Batch Processing
```json
[
  "What is artificial intelligence?",
  "Explain machine learning",
  "How does deep learning work?"
]
```

## Error Handling and Troubleshooting

When commands don't work as expected:
- Cursor will refer to the official CLI documentation for the most up-to-date information
- Configuration validation will catch most common issues
- Use component discovery to verify exact component names and parameters
- Check service endpoints and timeouts for evaluation and serving commands

## Integration with Development Environment

The Cursor rules seamlessly integrate with your development workflow:
- **Automatic command execution**: Natural language requests are translated to proper CLI commands
- **File validation**: Configurations are checked for errors and inconsistencies
- **Service management**: Endpoints are started, monitored, and configured appropriately
- **Development feedback**: Clear error messages and suggestions when things go wrong

This comprehensive rule system enables you to manage complex AIQ workflows through simple, natural language interactions with the Cursor agent.
