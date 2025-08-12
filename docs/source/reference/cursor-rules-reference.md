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

This document provides a comprehensive reference for all available Cursor rules in NeMo Agent toolkit. Each rule includes a purpose description, usage prompt, and practical examples.

## Foundation Rules

### General Development Guidelines

**Cursor Rule file**: `.cursor/rules/general.mdc`
**Purpose**: Overarching standards for all source, test, documentation, and CI files.

**Prompt**:
```
Create a new Python function with proper type hints, docstrings, and formatting that follows NeMo Agent toolkit coding standards.
```

**Capabilities**:
- Project structure guidelines
- Code formatting standards
- Type hint requirements
- Documentation standards
- Testing practices
- CI/CD compliance

---

### Cursor Rules Management

**Cursor Rule file**: `.cursor/rules/cursor-rules.mdc`
**Purpose**: Guidelines for creating and managing cursor rules themselves.

**Prompt**:
```
Create a new Cursor rule for creating a new NeMo Agent workflow
```

**Capabilities**:
- Rule file naming conventions
- Directory structure for rules
- Documentation standards for rules
- Best practices for rule descriptions

---

## Setup and Installation Rules

### General Setup Guidelines

**Cursor Rule file**: `.cursor/rules/nat-setup/general.mdc`
**Purpose**: Guidance for NeMo Agent toolkit installation, setup, and environment configuration.

**Prompt**:
```
Help me set up NeMo Agent toolkit development environment with all required dependencies and configurations.
```

**Capabilities**:
- Installation troubleshooting
- Environment setup guidance
- Dependency management
- Initial configuration steps

**Related Documentation**: [Installation Guide](../quick-start/installing.md)

---

### NeMo Agent Toolkit Installation

**Cursor Rule file**: `.cursor/rules/nat-setup/nat-toolkit-installation.mdc`
**Purpose**: Detailed installation procedures and setup guidance.

**Prompt**:
```
Install NeMo Agent toolkit with all plugins and verify the installation is working correctly.
```



**Related Documentation**: [Installation Guide](../quick-start/installing.md)

---

## CLI Command Rules

### General CLI Guidelines

**Cursor Rule file**: `.cursor/rules/nat-cli/general.mdc`
**Purpose**: Guidance for all NeMo Agent CLI commands, operations, and functionality.

**Prompt**:
```
Show me how to use CLI commands to manage workflows
```

**Capabilities**:
- CLI command reference
- Common usage patterns
- Error troubleshooting
- Best practices for CLI operations

**Related Documentation**: [CLI Reference](./cli.md)

---

### NeMo Agent Workflow Commands

**Cursor Rule file**: `.cursor/rules/nat-cli/nat-workflow.mdc`
**Purpose**: Creating, reinstalling, and deleting NeMo Agent workflows.

**Prompt**:
```
Create a workflow named demo_workflow in examples directory with description "Demo workflow for testing features".
```



**Related Documentation**: [CLI Reference - Workflow Commands](./cli.md#workflow)

---

### NeMo Agent Run and Serve Commands

**Cursor Rule file**: `.cursor/rules/nat-cli/nat-run-serve.mdc`
**Purpose**: Running, serving, and executing NeMo Agent workflows.

**Prompt**:
```
Run my workflow locally for testing and then serve it as an API endpoint on port 8080.
```



**Related Documentation**:
- [CLI Reference - Run Commands](./cli.md#run)
- [Running Workflows](../workflows/run-workflows.md)

---

### NeMo Agent Evaluation Commands

**Cursor Rule file**: `.cursor/rules/nat-cli/nat-eval.mdc`
**Purpose**: Evaluating workflow performance and quality.

**Prompt**:
```
Evaluate my workflow performance using a test dataset with accuracy and precision metrics.
```

**Related Documentation**:
- [CLI Reference - Evaluation Commands](./cli.md#evaluation)
- [Workflow Evaluation](../workflows/evaluate.md)

---

### NeMo Agent Info Commands

**Cursor Rule file**: `.cursor/rules/nat-cli/nat-info.mdc`
**Purpose**: Getting information about NeMo Agent components and system status.

**Prompt**:
```
Show me system information and list all available NeMo Agent components with their details.
```

**Related Documentation**: [CLI Reference - Info Commands](./cli.md#information-commands)

---

## Workflow Development Rules

### General Workflow Guidelines

**Cursor Rule file**: `.cursor/rules/nat-workflows/general.mdc`
**Purpose**: Guidance for NeMo Agent workflows, functions, and tools.

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

**Cursor Rule file**: `.cursor/rules/nat-workflows/add-functions.mdc`
**Purpose**: Implementing, adding, creating, or modifying functions within NeMo Agent workflows.

**Prompt**:
```
Add a text processing function to my workflow that splits text into sentences and counts words.
```

**Related Documentation**:
- [Writing Custom Functions](../extend/functions.md)
- [Functions Overview](../workflows/functions/index.md)

---

### Adding Tools to Workflows

**Cursor Rule file**: `.cursor/rules/nat-workflows/add-tools.mdc`
**Purpose**: Adding, integrating, implementing, or configuring tools for NeMo Agent workflows.

**Prompt**:
```
Integrate a web search tool into my workflow that can fetch and process search results from the internet.
```

**Related Documentation**: [Adding Tools Tutorial](../tutorials/add-tools-to-a-workflow.md)

---

## Agent Rules

### Agent Integration and Selection

**Cursor Rule file**: `.cursor/rules/nat-agents/general.mdc`
**Purpose**: Guidelines for integrating or selecting ReAct, Tool-Calling, Reasoning, or ReWOO agents within NeMo Agent workflows.

**Prompt**:
```
Integrate ReAct agent to the workflow
```

**Related Documentation**: [Agent Docs](../workflows/about/index.md)

---

## Quick Reference

<!-- path-check-skip-begin -->
| Rule Category | Cursor Rule file | Primary Use Case |
|---------------|---------|------------------|
| Foundation | `general` | Code quality and standards |
| Foundation | `cursor-rules` | Managing cursor rules |
| Setup | `nat-setup/general` | Environment setup |
| Setup | `nat-setup/nat-toolkit-installation` | Installation procedures |
| CLI | `nat-cli/general` | General CLI usage |
| CLI | `nat-cli/nat-workflow` | Workflow management |
| CLI | `nat-cli/nat-run-serve` | Running and serving |
| CLI | `nat-cli/nat-eval` | Performance evaluation |
| CLI | `nat-cli/nat-info` | System information |
| Workflow | `nat-workflows/general` | Workflow design |
| Workflow | `nat-workflows/add-functions` | Function development |
| Workflow | `nat-workflows/add-tools` | Tool integration |
| Agents | `nat-agents/general` | Agent selection & integration |
<!-- path-check-skip-end -->

## Usage Tips

* **Copy Exact Prompts**: Use the provided prompts exactly as shown for best results
* **Customize for Your Needs**: Modify prompts with specific project details
* **Chain Rules**: Use multiple rules together for complex development tasks
* **Reference Documentation**: Follow the "Related Documentation" links for deeper understanding
* **Test Incrementally**: Apply one rule at a time and test the results

For tutorials and examples on using these rules, see [Build a Demo Agent Workflow Using Cursor Rules for NeMo Agent Toolkit](../tutorials/build-a-demo-agent-workflow-using-cursor-rules.md).
