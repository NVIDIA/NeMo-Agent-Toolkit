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

# Cursor Rules for AIQ Toolkit Development

**Streamline AIQ workflow creation with intelligent prompts.** Cursor rules enable you to build complete agent workflows, integrate functions, and configure tools through natural language commands – transforming complex development tasks into simple conversational interactions.

**Looking to accelerate your AIQ development workflow?** This guide demonstrates how to harness Cursor rules for efficient, consistent AIQ Toolkit development.

## Table of Contents

- [Cursor Rules for AIQ Toolkit Development](#cursor-rules-for-aiq-toolkit-development)
  - [Table of Contents](#table-of-contents)
  - [Why Use Cursor Rules](#why-use-cursor-rules)
    - [Benefits for Newcomers](#benefits-for-newcomers)
    - [Benefits for Expert Developers](#benefits-for-expert-developers)
    - [Code Quality Benefits](#code-quality-benefits)
  - [Cursor Rules Organization](#cursor-rules-organization)
  - [Getting Started with Common Prompts](#getting-started-with-common-prompts)
  - [Practical Example: Building a Demo Agent with Cursor Rules](#practical-example-building-a-demo-agent-with-cursor-rules)
    - [Step 1: Install AIQ Toolkit](#step-1-install-aiq-toolkit)
    - [Step 2: Explore Available Tools](#step-2-explore-available-tools)
    - [Step 3: Create the Workflow](#step-3-create-the-workflow)
    - [Step 4: Configure DateTime Function](#step-4-configure-datetime-function)
    - [Step 5: Integrate React Agent](#step-5-integrate-react-agent)
    - [Step 6: Execute the Workflow](#step-6-execute-the-workflow)
  - [Summary](#summary)

## Why Use Cursor Rules

Cursor rules in AIQ Toolkit act as an intelligent development companion that offers structured assistance for developers at all experience levels:

### Benefits for Newcomers
- **Guided Learning Experience**: Provides an interactive approach to mastering AIQ Toolkit through contextual assistance
- **Progressive Guidance**: Offers comprehensive examples for typical development workflows
- **Simplified Onboarding**: Eliminates the complexity of memorizing intricate CLI syntax and configuration structures

### Benefits for Expert Developers
- **Accelerated Development**: Provides streamlined workflows with established, tested patterns
- **Enhanced Productivity**: Minimizes routine tasks while automatically applying best practices
- **Standardized Workflows**: Ensures uniform development approaches across development teams and projects

### Code Quality Benefits
- **Best Practice Enforcement**: Automatically applies standards for coding practices, documentation, and configuration
- **Improved Code Standards**: Maintains consistent formatting, type annotations, and documentation requirements
- **Streamlined Code Reviews**: Decreases review overhead by ensuring consistent patterns in submissions
- **Standards Compliance**: Assists in maintaining adherence to project conventions and requirements

## Cursor Rules Organization

AIQ Toolkit offers a comprehensive collection of Cursor rules organized into four primary categories:

- **[Foundation Rules](../reference/cursor-rules-reference.md#foundation-rules)**: Core code quality standards and cursor rules management
- **[Setup and Installation Rules](../reference/cursor-rules-reference.md#setup-and-installation-rules)**: Environment configuration and toolkit installation procedures
- **[CLI Command Rules](../reference/cursor-rules-reference.md#cli-command-rules)**: Complete AIQ CLI operations and command handling
- **[Workflow Development Rules](../reference/cursor-rules-reference.md#workflow-development-rules)**: Function and tool development for workflow creation

For a **comprehensive overview of all supported tasks**, including detailed prompts, examples, and capabilities for each rule, see the **[Cursor Rules Reference](../reference/cursor-rules-reference.md)**.

## Getting Started with Common Prompts

:::{note}
For optimal Cursor rules experience, avoid using the `Auto` mode for LLM model selection. Instead, manually choose a model from the selection menu, such as `claude-4-sonnet`.
:::

Here are frequently used prompts to begin your development:

**Installing AIQ Toolkit:**
```
Install AIQ Toolkit with all dependencies and verify the installation is working correctly.
```

**Environment setup:**
```
Help me set up AIQ Toolkit development environment with all required dependencies and configurations.
```

**Workflow creation:**
```
Create a workflow named demo_workflow in examples directory with description "Demo workflow for testing features".
```

**Function integration:**
```
Add a text processing function to my workflow that splits text into sentences and counts words.
```

**Running and serving workflows:**
```
Run my workflow locally for testing and then serve it as an API endpoint on port 8080.
```

For complete documentation with all available rules, prompts, and examples, refer to the **[Cursor Rules Reference](../reference/cursor-rules-reference.md)**.

## Practical Example: Building a Demo Agent with Cursor Rules

This comprehensive example demonstrates creating and running a functional agent workflow using Cursor rules:

### Step 1: Install AIQ Toolkit

Prompt:
```
Install AIQ Toolkit with all required dependencies and verify the installation
```

The assistant will reference and apply the [aiq-setup/aiq-toolkit-installation](../../../.cursor/rules/aiq-setup/aiq-toolkit-installation.mdc) rule to validate prerequisites and install the toolkit, followed by installation verification.

<div align="center">
  <img src="../_static/cursor_rules_demo/install.gif" width="600">
</div>

### Step 2: Explore Available Tools

Prompt:
```
Find datetime-related functions and tools available in AIQ Toolkit
```
The assistant will reference and apply the [aiq-cli/aiq-info](../../../.cursor/rules/aiq-cli/aiq-info.mdc) rule to discover available tools and functions.

<div align="center">
  <img src="../_static/cursor_rules_demo/find_tool.gif" width="600">
</div>


### Step 3: Create the Workflow

Prompt:
```
Create a new workflow named `demo_workflow` in the examples folder
```

The assistant will reference and apply the [aiq-workflows/general](../../../.cursor/rules/aiq-workflows/general.mdc) rule to generate a new workflow using the `aiq workflow create` command.

<div align="center">
  <img src="../_static/cursor_rules_demo/create_workflow.gif" width="600">
</div>

### Step 4: Configure DateTime Function

Prompt:
```
Add the current_datetime function to the demo_workflow
```

The assistant will reference and apply the [aiq-workflows/add-functions](../../../.cursor/rules/aiq-workflows/add-functions.mdc) rule to integrate the function into the workflow.

<div align="center">
  <img src="../_static/cursor_rules_demo/add_tool.gif" width="600">
</div>


### Step 5: Integrate React Agent

Prompt:
```
Integrate ReAct agent to the workflow
```
The assistant will reference and apply the [aiq-agents/general](../../../.cursor/rules/aiq-agents/general.mdc) rule to integrate a ReAct agent within the workflow.

<div align="center">
  <img src="../_static/cursor_rules_demo/react_agent.gif" width="600">
</div>

### Step 6: Execute the Workflow

Prompt:
```
Run the demo_workflow
```

The assistant will reference and apply the [aiq-cli/aiq-run-serve](../../../.cursor/rules/aiq-cli/aiq-run-serve.mdc) rule to run the workflow.

<div align="center">
  <img src="../_static/cursor_rules_demo/run_workflow.gif" width="600">
</div>

Congratulations! You have successfully created a functional demo workflow using Cursor rules with minimal manual coding!

:::{note}
Keep your prompts specific and concise. For instance, rather than stating "Create a workflow", specify "Create a workflow named `demo_workflow` in examples directory with description `Demo workflow for testing features`".
:::

## Summary

Cursor rules in AIQ Toolkit offer a powerful approach to streamline development processes, ensure consistency, and maintain superior code quality. By leveraging these guidelines and applying the appropriate rules for each development task, you can substantially enhance your development efficiency while creating robust, maintainable AIQ workflows.
