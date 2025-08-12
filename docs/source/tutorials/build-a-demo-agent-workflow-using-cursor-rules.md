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

# Build a Demo Agent Workflow Using Cursor Rules for NVIDIA NeMo Agent Toolkit

Learn how to use Cursor rules for NeMo Agent toolkit development to create and run a demo agent workflow.

## About Cursor Rules
Cursor rules in NeMo Agent toolkit act as an intelligent development that offers structured assistance for developers at all experience levels. The key functionalities of Cursor rules are as follows:
* Streamline workflow creation with intelligent prompts: You can build complete agent workflows, integrate functions, and configure tools through natural language commands. It allows you to transform complex development tasks into simple conversational interactions.
* Accelerate development workflows: You can use Cursor rules to develop NeMo Agent toolkit efficiently and consistently as it provides streamlined workflows with established and tested patterns. It also enhances productivity by minimizing routine tasks, while applying best practices for coding, documentation, and configuration.
* Learn and understand NeMo Agent toolkit quickly and simply: For less experienced developers, Cursor rules provide an interactive approach to mastering NeMo Agent toolkit through contextual assistance and comprehensive examples for typical development workflows.
* Standardization: Ensures uniform development standards, such as formatting, type annotations, and documentation requirements, across development teams and projects. Thus, decreasing code review overhead during submissions.

## Common Prompts

:::{note}
For optimal Cursor rules experience, avoid using the `Auto` mode for LLM model selection. Instead, manually choose a model from the selection menu, such as `claude-4-sonnet`.
:::

The following are frequently used prompts to begin development:

**Installing NeMo Agent Toolkit:**
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

## Building a Demo Agent with Cursor Rules

Follow the steps below for a comprehensive example that demonstrates creating and running a functional agent workflow using Cursor rules:

### Install NeMo Agent Toolkit

Before you begin, make sure you have cloned the NeMo Agent toolkit repository and opened the project in Cursor, by selecting `File > Open Workspace from File... > select the nat.code-workspace in the repository`.

Prompt:
```
Install AIQ Toolkit with all required dependencies and verify the installation
```

The assistant will reference and apply the [toolkit-installation](../../../.cursor/rules/nat-setup/nat-toolkit-installation.mdc) rule to validate prerequisites and install the toolkit, followed by installation verification.

<div align="center">
  <img src="../_static/cursor_rules_demo/install.gif" width="600">
</div>

### Explore Available Tools

Prompt:
```
Find datetime-related functions and tools available in AIQ Toolkit
```
The assistant will reference and apply the [info](../../../.cursor/rules/nat-cli/nat-info.mdc) rule to discover available tools and functions.

<div align="center">
  <img src="../_static/cursor_rules_demo/find_tool.gif" width="600">
</div>


### Create the Workflow

Prompt:
```
Create a new workflow named `demo_workflow` in the examples folder
```

The assistant will reference and apply the [general](../../../.cursor/rules/nat-workflows/general.mdc) rule to generate a new workflow using the `nat workflow create` command.

<div align="center">
  <img src="../_static/cursor_rules_demo/create_workflow.gif" width="600">
</div>

### Configure the DateTime Function

Prompt:
```
Add the current_datetime function to the demo_workflow
```

The assistant will reference and apply the [add-functions](../../../.cursor/rules/nat-workflows/add-functions.mdc) rule to integrate the function into the workflow.

<div align="center">
  <img src="../_static/cursor_rules_demo/add_tool.gif" width="600">
</div>


### Integrate the ReAct Agent

Prompt:
```
Integrate ReAct agent to the workflow
```
The assistant will reference and apply the [general](../../../.cursor/rules/nat-agents/general.mdc) rule to integrate a ReAct agent within the workflow.

<div align="center">
  <img src="../_static/cursor_rules_demo/react_agent.gif" width="600">
</div>

### Run the Workflow

Prompt:
```
Run the demo_workflow
```

The assistant will reference and apply the [run-serve](../../../.cursor/rules/nat-cli/nat-run-serve.mdc) rule to run the workflow.

<div align="center">
  <img src="../_static/cursor_rules_demo/run_workflow.gif" width="600">
</div>

Congratulations! You have successfully created a functional demo workflow using Cursor rules with minimal manual coding!

:::{note}
Keep your prompts specific and concise. For instance, rather than stating "Create a workflow", specify "Create a workflow named `demo_workflow` in examples directory with description `Demo workflow for testing features`".
:::

## Cursor Rules Organization

NeMo Agent toolkit offers a comprehensive collection of Cursor rules organized into four primary categories:

- **[Foundation Rules](../reference/cursor-rules-reference.md#foundation-rules)**: Core code quality standards and cursor rules management
- **[Setup and Installation Rules](../reference/cursor-rules-reference.md#setup-and-installation-rules)**: Environment configuration and toolkit installation procedures
- **[CLI Command Rules](../reference/cursor-rules-reference.md#cli-command-rules)**: Complete CLI operations and command handling
- **[Workflow Development Rules](../reference/cursor-rules-reference.md#workflow-development-rules)**: Function and tool development for workflow creation

For a **comprehensive overview of all supported tasks**, including detailed prompts, examples, and capabilities for each rule, refer to the **[Cursor Rules Reference](../reference/cursor-rules-reference.md)**.
