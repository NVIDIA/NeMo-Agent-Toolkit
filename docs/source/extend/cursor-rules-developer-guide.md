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

# Cursor Rules Developer Guide

This guide explains how to organize, create, and maintain Cursor rules within the NeMo Agent toolkit project.

## Overview

Cursor Rules allow you to provide system-level guidance to AI assistants, functioning as persistent context that helps them understand your project and preferences. According to the [official Cursor documentation](https://docs.cursor.com/context/rules), rules solve the problem that "Large language models do not retain memory between completions" by providing persistent, reusable context at the prompt level.

In the NeMo Agent toolkit project, Cursor rules serve as specialized documentation files that extract information from project documentation and convert it into system prompts for AI agents. They help AI assistants understand:

* Project-specific patterns and conventions
* Configuration requirements for different components
* Best practices for integration and implementation
* Decision-making criteria for choosing between alternatives

When a rule is applied, its contents are included at the start of the model context, providing consistent guidance whether the AI is generating code, interpreting edits, or helping with workflows.

## Rule Organization Structure

The NeMo Agent toolkit uses a hierarchical structure for organizing Cursor rules under `.cursor/rules/`:

```
.cursor/rules/
├── cursor-rules.mdc           # Meta-rules for creating Cursor rules
├── general.mdc                # Project-wide coding standards
├── aiq-agents/                # Agent integration and selection rules
│   └── general.mdc
├── aiq-cli/                   # CLI command rules
│   ├── general.mdc
│   ├── aiq-eval.mdc          # Evaluation commands
│   ├── aiq-info.mdc          # Info commands
│   ├── aiq-run-serve.mdc     # Run and serve commands
│   └── aiq-workflow.mdc      # Workflow management commands
├── aiq-setup/                 # Setup and installation rules
│   ├── general.mdc
│   └── aiq-toolkit-installation.mdc
└── aiq-workflows/             # Workflow development rules
    ├── general.mdc
    ├── add-functions.mdc      # Function creation and integration
    └── add-tools.mdc          # Tool integration
```

### Core Rules Files

#### Cursor Rules MDC
The foundation file (`cursor-rules.mdc`) containing meta-rules that define:
* File naming conventions (kebab-case with `.mdc` extension)
* Directory structure requirements
* YAML format specifications
* Documentation referencing patterns
* Guidelines for writing effective rule descriptions

#### General MDC
The general rules file (`general.mdc`) contains project-wide coding standards including:
* Project structure guidelines
* Code formatting and import rules
* Type hints requirements
* Documentation standards (Google-style docstrings)
* Testing practices with pytest
* CI/CD compliance rules
* Security and performance guidelines

### Topic-Based Subdirectories

Each subdirectory focuses on a specific area of the toolkit:

#### `aiq-agents/`
* **`general.mdc`**: Integration guidelines for ReAct, Tool-Calling, Reasoning, and ReWOO agents
* Includes configuration parameters, selection criteria, and best practices
* Contains decision matrix for choosing appropriate agent types

#### `aiq-cli/`
* **`general.mdc`**: Meta-rules referencing CLI documentation
* **`aiq-eval.mdc`**: Detailed rules for workflow evaluation commands
* **`aiq-info.mdc`**: System information and component querying rules
* **`aiq-run-serve.mdc`**: Local execution and API serving guidelines
* **`aiq-workflow.mdc`**: Workflow creation, installation, and deletion rules

#### `aiq-setup/`
* **`general.mdc`**: Environment setup and configuration guidance
* **`aiq-toolkit-installation.mdc`**: Comprehensive installation procedures

#### `aiq-workflows/`
* **`general.mdc`**: High-level workflow architecture guidance
* **`add-functions.mdc`**: Detailed function creation, registration, and composition rules
* **`add-tools.mdc`**: Tool integration and configuration guidelines

## Creating and Maintaining Cursor Rules

### Fundamental Principles

* **Documentation-First Approach**: After updating the codebase, always create or update documentation first, then create Cursor rules based on that documentation. This ensures Cursor rules stay aligned with the latest codebase changes and maintain consistency with the documentation.

* **Use Cursor Agent to Create Rules**: Always use the Cursor Agent to create rules. This approach is faster and more importantly, it automatically follows `@cursor/rules/cursor-rules.mdc` to ensure rules are consistent with the rule creation guidelines and maintain the proper organization structure.

### Rule Creation Process

1. **Update Documentation First**
   
   Create or update the documentation for the feature you want to add Cursor rules for. You can also create Cursor rules based on existing documentation.

2. **Use Cursor Agent to Create Rules**
   
   The most efficient way to create Cursor rules is to use the Cursor agent itself. Use a prompt like this:

   ```
   Read the @cli.md documentation and create Cursor rules for CLI command use cases including `aiq workflow create/reinstall/delete`, `aiq run/serve`, `aiq info`, and `aiq eval`. 
   
   The goal is to enable the Cursor agent to execute the correct CLI commands with proper arguments when users request these actions. For example, when a user asks to create a workflow, the agent should respond with the correct `aiq workflow create` command syntax.
   
   Please follow @cursor-rules.mdc guidelines for rule structure and formatting.
   ```

   :::{note}
   Important: To ensure the context window of the Cursor agent is large enough, DO NOT use the `Auto` mode of LLM model selection. Instead, manually select a model from the toggle list, such as `claude-4-sonnet`.
   :::

3. **Select Proper Rule Type and Add Description**
   
   According to the [official Cursor documentation](https://docs.cursor.com/context/rules), there are four types of Cursor rules, which are defined in the `.mdc` metadata header:

   | Rule Type | Description | When to Use |
   |-----------|-------------|-------------|
   | **Always** (`alwaysApply: true`) | Always included in the model context | Universal project standards that should apply to all interactions |
   | **Auto Attached** (with `globs` pattern) | Included when files matching a glob pattern are referenced | Rules specific to certain file types or directories |
   | **Agent Requested** (`alwaysApply: false` + `description`) | Available to the AI, which decides whether to include it | Task-specific rules that the AI should choose based on context |
   | **Manual** (`alwaysApply: false`, no `description`) | Only included when explicitly mentioned using @ruleName | Rules that should only be applied when explicitly requested |

### Writing Effective Agent Requested Rule Descriptions

   For **Agent Requested** rules, the description is crucial as it helps the AI determine when to apply the rule. Based on existing NeMo Agent toolkit rules, follow these patterns:

   * `"Follow these rules when the user's request involves integrating or selecting ReAct, Tool-Calling, Reasoning, or ReWOO agents within NeMo Agent workflows"`
   * `"Follow these rules when the user's request involves creating, reinstalling, or deleting NeMo Agent workflows"`
   * `"Follow these rules when the user's request involves running, serving, or executing NeMo Agent workflows"`
