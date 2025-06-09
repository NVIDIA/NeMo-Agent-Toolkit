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

# Using Cursor Rules for AIQ Toolkit Development

**Create complete AIQ workflows with minimal code.** With Cursor rules, generate entire agent workflows, and function integrations using simple prompts – dramatically reducing the code you need to write while transforming hours of development into streamlined, guided execution.

**Ready to accelerate your development?** This guide shows you how to leverage Cursor rules for faster, more consistent AIQ Toolkit workflows.

## Motivation

Cursor rules in AIQ Toolkit serve as an intelligent development assistant that provides structured guidance for both new and experienced users:

### For New Users
- **Interactive Learning**: Provides an interactive and easy way to learn about the AIQ Toolkit through contextual guidance
- **Step-by-Step Guidance**: Offers detailed examples for common development tasks
- **Reduced Learning Curve**: Eliminates the need to memorize complex CLI commands and configuration patterns

### For Experienced Users
- **Streamlined Development**: Provides a way to streamline the development process with consistent, fixed patterns
- **Productivity Boost**: Reduces repetitive tasks and ensures best practices are followed automatically
- **Consistent Workflow**: Standardizes development patterns across team members and projects

### Code Quality Enhancement
- **Enforced Standards**: Automatically enforces rules on coding practices, documentation, and configuration settings
- **Enhanced Code Quality**: Ensures consistent formatting, type hints, and documentation standards
- **Simplified Code Review**: Reduces review time by ensuring submissions follow established patterns
- **Automated Compliance**: Helps maintain compliance with project standards and conventions

## Available Cursor Rules

AIQ Toolkit provides a comprehensive set of Cursor rules to assist with development tasks. The rules are organized into four main categories:

- **[Foundation Rules](../reference/cursor-rules-reference.md#foundation-rules)**: Code quality standards and cursor rules management
- **[Setup and Installation Rules](../reference/cursor-rules-reference.md#setup-and-installation-rules)**: Environment setup and toolkit installation
- **[CLI Command Rules](../reference/cursor-rules-api.md#cli-command-rules)**: All AIQ CLI operations and commands
- **[Workflow Development Rules](../reference/cursor-rules-api.md#workflow-development-rules)**: Function and tool development for workflows

For a **complete list of all tasks** that Cursor rules can handle, including detailed prompts, examples, and capabilities for each rule, refer to the **[Cursor Rules Reference](../reference/cursor-rules-reference.md)**.

### Quick Start Examples

Here are some commonly used prompts to get you started:

**Install AIQ Toolkit:**
```
Install AIQ Toolkit with all dependencies and verify the installation is working correctly.
```

**Set up development environment:**
```
Help me set up AIQ Toolkit development environment with all required dependencies and configurations.
```

**Create a new workflow:**
```
Create a workflow named demo_workflow in examples directory with description "Demo workflow for testing features".
```

**Add a function to a workflow:**
```
Add a text processing function to my workflow that splits text into sentences and counts words.
```

**Run and serve a workflow:**
```
Run my workflow locally for testing and then serve it as an API endpoint on port 8080.
```

For a complete reference with all available rules, prompts, and examples, see the **[Cursor Rules Reference](../reference/cursor-rules-reference.md)**.

## Sample Workflow: Creating a DateTime Agent

Here's a complete workflow demonstrating how to create and run a simple agent using Cursor rules:

### Step 1: Install AIQ Toolkit

```
Install AIQ Toolkit with all required dependencies and verify the installation
```

**Commands**:
```bash
# Install AIQ Toolkit with all dependencies
pip install aiq-toolkit[all]

# Verify installation
aiq --version
```

### Step 2: Check Available Tools

```
Find datetime-related functions and tools available in AIQ Toolkit
```

### Step 3: Create the Workflow

```
Set up a new workflow named datetime_workflow in the examples folder
```

### Step 4: Add DateTime Tool to Configuration

```
Include the current_datetime function in the datetime_workflow configuration
```

**Result**: Configuration file with datetime tool:
```yaml
# File: examples/datetime_workflow/config.yaml
functions:
  current_datetime:
    _type: current_datetime
```

### Step 5: Configure React Agent

```
Modify the configuration to integrate the react agent workflow
```

**Result**: Complete configuration with React agent:
```yaml
# File: examples/datetime_workflow/config.yaml
workflow:
  _type: react_agent
  tool_names: [current_datetime]
  llm_name: nim_llm
  verbose: true
  handle_parsing_errors: true
  max_retries: 2

functions:
  current_datetime:
    _type: current_datetime

llms:
  nim_llm:
    _type: nim_llm
    model: meta/llama-3.1-8b-instruct
```

### Step 6: Update Python Code

```
Implement the Python workflow registration to enable the react agent for datetime queries
```

**Result**: Agent implementation in register.py:
```python
# File: examples/datetime_workflow/register.py
from aiq.cli.register_workflow import register_workflow

@register_workflow
async def workflow(builder):
    """DateTime workflow using React agent."""
    return builder.get_workflow()
```

### Step 7: Run the Workflow

```
Execute the datetime_workflow and test its functionality
```

**Commands**:
```bash
# Install the workflow
aiq workflow reinstall datetime_workflow

# Run the workflow
aiq run datetime_workflow --param query="What is the current date and time?"
```

## Best Practices

1. **Use Prompts**: Copy the exact prompts provided in each section above and paste them into Cursor to trigger the appropriate rules automatically

2. **Be Specific**: Modify the provided prompts with your specific requirements (e.g., change "demo_workflow" to your actual workflow name)

3. **Follow the Pattern**: Each rule provides specific patterns and examples - follow them exactly for consistency

4. **Leverage Documentation**: Rules reference comprehensive documentation - use both the rules and the referenced docs

5. **Test Incrementally**: After each change, use `aiq workflow reinstall` and test your workflow

6. **Use Code Quality Rules**: Always include requests for proper formatting, type hints, and documentation standards

7. **Clean Up**: Use `aiq workflow delete` to remove unused workflows and keep your environment clean

## Troubleshooting

If you encounter issues:

1. **Check Rule Descriptions**: Ensure you're using the correct rule for your specific task
2. **Verify Dependencies**: Use `@aiq-setup/aiq-toolkit-installation` for installation issues
3. **Review Documentation**: Each rule references specific documentation for deeper understanding
4. **Follow Exact Patterns**: Copy the provided examples exactly, then modify as needed
5. **Use Info Commands**: Use `@aiq-cli/aiq-info` to get system and component information

## Conclusion

Cursor rules in AIQ Toolkit provide a powerful way to streamline development, ensure consistency, and maintain high code quality. By following these guidelines and using the appropriate rules for each task, you can significantly improve your development efficiency and create robust, maintainable AIQ workflows.
