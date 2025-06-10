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

This guide explains how Cursor rules are organized, created, and maintained in the AIQ Toolkit project. Cursor rules provide structured guidance to AI assistants for specific development tasks, ensuring consistent and accurate agent responses.

## What are Cursor Rules?

Cursor Rules allow you to provide system-level guidance to the Agent and Cmd-K AI, functioning as persistent context that helps AI assistants understand your project and preferences. According to the [official Cursor documentation](https://docs.cursor.com/context/rules), rules solve the problem that "Large language models do not retain memory between completions" by providing persistent, reusable context at the prompt level.

In the context of AIQ Toolkit, Cursor rules are specialized documentation files that extract information from project documentation and convert it into system prompts for AI agents. They serve as contextual guidelines that help AI assistants understand:

- Project-specific patterns and conventions
- Configuration requirements for different components
- Best practices for integration and implementation
- Decision-making criteria for choosing between alternatives

When a rule is applied, its contents are included at the start of the model context, giving the AI consistent guidance whether it is generating code, interpreting edits, or helping with workflows.

## Rule Organization Structure

The Cursor rules are organized in a hierarchical structure under `.cursor/rules/`:

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

#### `cursor-rules.mdc`
The most important file in the system. It contains meta-rules that define:
- File naming conventions (kebab-case with `.mdc` extension)
- Directory structure requirements
- YAML frontmatter format with description, globs, and alwaysApply fields
- Documentation referencing patterns
- Best practices for writing effective rule descriptions

#### `general.mdc`
Contains project-wide coding standards including:
- Project structure guidelines
- Code formatting and import rules (isort, yapf)
- Type hints requirements
- Documentation standards (Google-style docstrings)
- Testing practices with pytest
- CI/CD compliance rules
- Security and performance guidelines

### Topic-Based Subdirectories

Each subdirectory focuses on a specific area of the toolkit:

#### `aiq-agents/`
- **`general.mdc`**: Integration guidelines for ReAct, Tool-Calling, Reasoning, and ReWOO agents
- Includes configuration parameters, selection criteria, and best practices
- Contains decision matrix for choosing appropriate agent types

#### `aiq-cli/`
- **`general.mdc`**: Meta-rules referencing CLI documentation
- **`aiq-eval.mdc`**: Detailed rules for workflow evaluation commands
- **`aiq-info.mdc`**: System information and component querying rules
- **`aiq-run-serve.mdc`**: Local execution and API serving guidelines
- **`aiq-workflow.mdc`**: Workflow creation, installation, and deletion rules

#### `aiq-setup/`
- **`general.mdc`**: Environment setup and configuration guidance
- **`aiq-toolkit-installation.mdc`**: Comprehensive installation procedures

#### `aiq-workflows/`
- **`general.mdc`**: High-level workflow architecture guidance
- **`add-functions.mdc`**: Detailed function creation, registration, and composition rules
- **`add-tools.mdc`**: Tool integration and configuration guidelines

## Creating and Maintaining Cursor Rules

### Fundamental Principle

**Documentation-First Approach**: Always create or update documentation first, then create Cursor rules based on that documentation. This ensures rules stay aligned with the latest codebase changes.

### Rule Creation Process

1. **Update Documentation First**
   ```bash
   # Example: Adding new agent type documentation
   echo "Add documentation for new agent type" > docs/source/workflows/about/new-agent.md
   ```

2. **Create Rule Based on Documentation**
   ```bash
   # Create rule file referencing the new documentation
   touch .cursor/rules/aiq-agents/new-agent.mdc
   ```

3. **Follow Naming Conventions**
   - Use kebab-case for filenames
   - Always use `.mdc` extension
   - Make names descriptive of the rule's purpose

### Rule File Structure

Every rule file must follow this structure:

```markdown
---
description: Follow these rules when the user's request involves [specific trigger conditions]
globs: optional/path/pattern/**/*
alwaysApply: false
---
# Rule Title

Brief description of the rule's purpose.

## Referenced Documentation

- **Doc Name**: [file.md](mdc:path/to/file.md) - Brief description
- **Another Doc**: [another.md](mdc:path/to/another.md) - Description

## Rules

1. Specific guidelines
2. Code examples
3. Best practices
```

### Key Requirements

#### YAML Frontmatter
- **description**: Must start with "Follow these rules when"
- **globs**: Optional file patterns to limit scope
- **alwaysApply**: Usually `false` for targeted rules

#### Referenced Documentation Section
- List all documentation files referenced in the rule
- Use `mdc:` prefix for internal documentation links
- Provide brief descriptions for each reference

#### Effective Descriptions
- Start with "Follow these rules when"
- Use specific trigger conditions
- Include relevant action verbs (creating, modifying, implementing, configuring)
- Be comprehensive but concise
- Use consistent terminology matching the project

### Subdirectory Rules

When creating topic-based subdirectories:

1. **Always include `general.mdc`** with overarching guidelines
2. **Use kebab-case** for directory names
3. **Place specific rules** as separate `.mdc` files within the subdirectory
4. **Include Referenced Documentation sections** in all `general.mdc` files

### Maintenance Best Practices

#### Regular Updates
- Review rules when documentation changes
- Ensure rule descriptions match current functionality
- Update referenced documentation links

#### Validation
- Test rules with actual AI interactions
- Verify that generated code follows the guidelines
- Check that all referenced documentation exists and is accurate

#### Documentation Alignment
- When updating code, update documentation first
- Then update corresponding Cursor rules
- Maintain consistency between docs and rules

### Example: Adding a New Rule

Let's say you're adding support for a new type of retriever:

1. **Create Documentation**
   ```markdown
   # Adding Custom Retrievers
   
   This guide explains how to create and integrate custom retrievers...
   ```

2. **Create Rule File**
   ```markdown
   ---
   description: Follow these rules when the user's request involves creating or integrating custom retrievers
   globs: docs/source/extend/**
   alwaysApply: false
   ---
   # Custom Retriever Integration Rules
   
   ## Referenced Documentation
   
   - **Custom Retriever Guide**: [adding-a-retriever.md](mdc:docs/source/extend/adding-a-retriever.md) - Guide for creating custom retrievers
   
   ## Rules
   
   1. Always inherit from the base retriever class
   2. Implement required async methods
   3. Use proper type hints and validation
   ```

## Integration with Development Workflow

### Pre-Development
- Check existing rules for similar functionality
- Review referenced documentation
- Understand the patterns and conventions

### During Development
- Use rules as guidance for implementation
- Follow the patterns established in the rules
- Reference the documentation links for detailed information

### Post-Development
- Update documentation if new patterns emerge
- Create or update rules to reflect new capabilities
- Ensure rules accurately represent the implemented functionality

## Best Practices Summary

1. **Documentation First**: Always update docs before rules
2. **Consistency**: Follow established naming and structure conventions
3. **Clarity**: Write clear, actionable rule descriptions
4. **Maintenance**: Regularly review and update rules
5. **Testing**: Validate rules through actual usage
6. **Integration**: Ensure rules work well together across topics

By following these guidelines, you'll maintain a robust system of Cursor rules that effectively guide AI assistants in generating consistent, high-quality code for the AIQ Toolkit project. 