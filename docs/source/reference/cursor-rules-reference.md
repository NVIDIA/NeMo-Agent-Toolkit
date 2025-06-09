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

# Cursor Rules Reference

This document provides a comprehensive reference for all available Cursor rules in AIQ Toolkit. Each rule includes a purpose description, usage prompt, and practical examples.

## Table of Contents

- [Cursor Rules Reference](#cursor-rules-reference)
  - [Table of Contents](#table-of-contents)
  - [Foundation Rules](#foundation-rules)
    - [General Development Guidelines](#general-development-guidelines)
    - [Cursor Rules Management](#cursor-rules-management)
  - [Setup and Installation Rules](#setup-and-installation-rules)
    - [General Setup Guidelines](#general-setup-guidelines)
    - [AIQ Toolkit Installation](#aiq-toolkit-installation)
  - [CLI Command Rules](#cli-command-rules)
    - [General CLI Guidelines](#general-cli-guidelines)
    - [AIQ Workflow Commands](#aiq-workflow-commands)
    - [AIQ Run and Serve Commands](#aiq-run-and-serve-commands)
    - [AIQ Evaluation Commands](#aiq-evaluation-commands)
    - [AIQ Info Commands](#aiq-info-commands)
  - [Workflow Development Rules](#workflow-development-rules)
    - [General Workflow Guidelines](#general-workflow-guidelines)
    - [Adding Functions to Workflows](#adding-functions-to-workflows)
    - [Adding Tools to Workflows](#adding-tools-to-workflows)
  - [Agent Rules](#agent-rules)
    - [AIQ Agents Integration \& Selection](#aiq-agents-integration--selection)
  - [Quick Reference](#quick-reference)
  - [Usage Tips](#usage-tips)

---

## Foundation Rules

### General Development Guidelines

**Rule ID**: `general`  
**Purpose**: Overarching standards for all source, test, documentation, and CI files.

**Prompt**:
```
Create a new Python function with proper type hints, docstrings, and formatting that follows AIQ Toolkit coding standards.
```

**Capabilities**:
- Project structure guidelines
- Code formatting standards (isort, yapf)
- Type hint requirements
- Documentation standards
- Testing practices
- CI/CD compliance

**Related Documentation**: [General Coding Guidelines](../resources/contributing.md#coding-standards)

---

### Cursor Rules Management

**Rule ID**: `cursor-rules`  
**Purpose**: Guidelines for creating and managing cursor rules themselves.

**Prompt**:
```
Create a new cursor rule file for handling database operations with proper naming conventions and structure.
```

**Capabilities**:
- Rule file naming conventions
- Directory structure for rules
- Documentation standards for rules
- Best practices for rule descriptions

**Related Documentation**: [Contributing Guidelines](../resources/contributing.md)

---

## Setup and Installation Rules

### General Setup Guidelines

**Rule ID**: `aiq-setup/general`  
**Purpose**: Guidance for AIQ toolkit installation, setup, and environment configuration.

**Prompt**:
```
Help me set up AIQ Toolkit development environment with all required dependencies and configurations.
```

**Capabilities**:
- Installation troubleshooting
- Environment setup guidance
- Dependency management
- Initial configuration steps

**Related Documentation**: [Installation Guide](../quick-start/installing.md)

---

### AIQ Toolkit Installation

**Rule ID**: `aiq-setup/aiq-toolkit-installation`  
**Purpose**: Detailed installation procedures and setup guidance.

**Prompt**:
```
Install AIQ Toolkit with all plugins and verify the installation is working correctly.
```

**Example**:
```bash
# Install AIQ Toolkit with all dependencies
pip install aiq-toolkit[all]

# Verify installation
aiq --version

# Initialize a new project
aiq init my-project
```

**Related Documentation**: [Installation Guide](../quick-start/installing.md)

---

## CLI Command Rules

### General CLI Guidelines

**Rule ID**: `aiq-cli/general`  
**Purpose**: Guidance for all AIQ CLI commands, operations, and functionality.

**Prompt**:
```
Show me how to use AIQ CLI commands to manage workflows and troubleshoot common issues.
```

**Capabilities**:
- CLI command reference
- Common usage patterns
- Error troubleshooting
- Best practices for CLI operations

**Related Documentation**: [CLI Reference](./cli.md)

---

### AIQ Workflow Commands

**Rule ID**: `aiq-cli/aiq-workflow`  
**Purpose**: Creating, reinstalling, and deleting AIQ workflows.

**Prompt**:
```
Create a workflow named demo_workflow in examples directory with description "Demo workflow for testing features".
```

**Examples**:
```bash
# Create a new workflow
aiq workflow create my_rag_workflow --description "A custom RAG workflow for document processing"

# Reinstall after code changes
aiq workflow reinstall my_rag_workflow

# Delete a workflow
aiq workflow delete my_rag_workflow
```

**Related Documentation**: [CLI Reference - Workflow Commands](./cli.md#workflow-commands)

---

### AIQ Run and Serve Commands

**Rule ID**: `aiq-cli/aiq-run-serve`  
**Purpose**: Running, serving, and executing AIQ workflows.

**Prompt**:
```
Run my workflow locally for testing and then serve it as an API endpoint on port 8080.
```

**Examples**:
```bash
# Run a workflow locally
aiq run my_workflow --config config.yaml

# Serve a workflow as an API
aiq serve my_workflow --port 8000 --host 0.0.0.0

# Run with custom parameters
aiq run my_workflow --param input_text="Hello World"
```

**Related Documentation**: 
- [CLI Reference - Run Commands](./cli.md#run-commands)
- [Running Workflows](../workflows/run-workflows.md)

---

### AIQ Evaluation Commands

**Rule ID**: `aiq-cli/aiq-eval`  
**Purpose**: Evaluating workflow performance and quality.

**Prompt**:
```
Evaluate my workflow performance using a test dataset with accuracy and precision metrics.
```

**Examples**:
```bash
# Evaluate a workflow
aiq eval my_workflow --dataset test_data.json

# Run evaluation with custom metrics
aiq eval my_workflow --metrics accuracy,precision,recall --output results.json
```

**Related Documentation**: 
- [CLI Reference - Evaluation Commands](./cli.md#evaluation-commands)
- [Workflow Evaluation](../workflows/evaluate.md)

---

### AIQ Info Commands

**Rule ID**: `aiq-cli/aiq-info`  
**Purpose**: Getting information about AIQ components and system status.

**Prompt**:
```
Show me system information and list all available AIQ components with their details.
```

**Examples**:
```bash
# Get system information
aiq info system

# List available components
aiq info components

# Get component details
aiq info component llm_provider
```

**Related Documentation**: [CLI Reference - Info Commands](./cli.md#info-commands)

---

## Workflow Development Rules

### General Workflow Guidelines

**Rule ID**: `aiq-workflows/general`  
**Purpose**: Guidance for AIQ workflows, functions, and tools.

**Prompt**:
```
Help me design a workflow architecture with proper function and tool integration following best practices.
```

**Capabilities**:
- Workflow architecture patterns
- Function and tool integration
- Best practices for workflow design
- Documentation references

**Related Documentation**: 
- [Workflow Overview](../workflows/about/index.md)
- [Functions Overview](../workflows/functions/index.md)

---

### Adding Functions to Workflows

**Rule ID**: `aiq-workflows/add-functions`  
**Purpose**: Implementing, adding, creating, or modifying functions within AIQ workflows.

**Prompt**:
```
Add a text processing function to my workflow that splits text into sentences and counts words.
```

**Example**:
```python
from aiq.data_models.function import FunctionBaseConfig
from aiq.cli.register_workflow import register_function
from aiq.builder.builder import Builder
from pydantic import Field

# 1. Define configuration
class MyFunctionConfig(FunctionBaseConfig, name="my_function"):
    """Configuration for My Function."""
    greeting: str = Field("Hello", description="The greeting to use.")
    repeat_count: int = Field(1, description="Number of times to repeat.")

# 2. Register the function
@register_function(config_type=MyFunctionConfig)
async def register_my_function(config: MyFunctionConfig, builder: Builder):
    async def _my_function(message: str) -> str:
        """My function implementation."""
        return f"{config.greeting}, {message}" * config.repeat_count
    
    yield _my_function
```

**Related Documentation**: 
- [Writing Custom Functions](../extend/functions.md)
- [Functions Overview](../workflows/functions/index.md)

---

### Adding Tools to Workflows

**Rule ID**: `aiq-workflows/add-tools`  
**Purpose**: Adding, integrating, implementing, or configuring tools for AIQ workflows.

**Prompt**:
```
Integrate a web search tool into my workflow that can fetch and process search results from the internet.
```

**Example**:
```python
from aiq.builder.tool_wrapper import ToolWrapper
from aiq.data_models.tool_wrapper import ToolWrapperConfig

# Define tool configuration
class CustomToolConfig(ToolWrapperConfig, name="custom_tool"):
    """Configuration for custom tool."""
    api_key: str = Field(description="API key for the tool")
    base_url: str = Field("https://api.example.com", description="Base URL")

# Register the tool
@register_tool(config_type=CustomToolConfig)
async def register_custom_tool(config: CustomToolConfig, builder: Builder):
    tool = CustomTool(api_key=config.api_key, base_url=config.base_url)
    yield ToolWrapper(tool=tool)
```

**Related Documentation**: [Adding Tools Tutorial](../tutorials/add-tools-to-a-workflow.md)

---

## Agent Rules

### AIQ Agents Integration & Selection

**Rule ID**: `aiq-agents/general`  
**Purpose**: Guidelines for integrating or selecting ReAct, Tool-Calling, Reasoning, or ReWOO agents within AIQ workflows.

**Prompt**:
```
Help me choose and configure the correct AIQ agent (ReAct, Tool-Calling, ReWOO, or Reasoning) for my workflow.
```

**Capabilities**:
- Integration steps for each agent type
- Configuration parameters for each agent
- Decision matrix for selecting the appropriate agent
- Best practices and known limitations

**Related Documentation**: [Agent Docs](../workflows/about/index.md)

---

## Quick Reference

| Rule Category | Rule ID | Primary Use Case |
|---------------|---------|------------------|
| Foundation | `general` | Code quality and standards |
| Foundation | `cursor-rules` | Managing cursor rules |
| Setup | `aiq-setup/general` | Environment setup |
| Setup | `aiq-setup/aiq-toolkit-installation` | Installation procedures |
| CLI | `aiq-cli/general` | General CLI usage |
| CLI | `aiq-cli/aiq-workflow` | Workflow management |
| CLI | `aiq-cli/aiq-run-serve` | Running and serving |
| CLI | `aiq-cli/aiq-eval` | Performance evaluation |
| CLI | `aiq-cli/aiq-info` | System information |
| Workflow | `aiq-workflows/general` | Workflow design |
| Workflow | `aiq-workflows/add-functions` | Function development |
| Workflow | `aiq-workflows/add-tools` | Tool integration |
| Agents | `aiq-agents/general` | Agent selection & integration |

## Usage Tips

1. **Copy Exact Prompts**: Use the provided prompts exactly as shown for best results
2. **Customize for Your Needs**: Modify prompts with specific project details
3. **Chain Rules**: Use multiple rules together for complex development tasks
4. **Reference Documentation**: Follow the "Related Documentation" links for deeper understanding
5. **Test Incrementally**: Apply one rule at a time and test the results

For tutorials and examples on using these rules, see [Using Cursor Rules for AIQ Toolkit Development](../tutorials/using-cursor-rules-for-aiq-toolkit-development.md). 